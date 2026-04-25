#!/usr/bin/env python3
"""Create a reproducible stratified train/val/test split for the XBOW benchmark suite.

Val selection strategy (--val-size, default 10):
  - Group all benchmarks by primary vulnerability type (tag).
  - From each tag group pick the hardest example (highest level; ties broken
    by benchmark ID for stability). This maximises category coverage in val.
  - Sort the per-tag candidates by difficulty descending and take the top
    --val-size examples as the val set.

Remaining benchmarks are split into train / test:
  - Singletons / pairs → all train
  - Triples            → train + test only
  - n ≥ 4             → 1 val (already taken above) + ~20% test + rest train

Usage:
    uv run python source/dataset/create_splits.py
    uv run python source/dataset/create_splits.py --seed 42 --val-size 10
"""

import argparse
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

SEED     = 42
VAL_SIZE = 10

_REPO_ROOT            = Path(__file__).resolve().parents[2]
_DEFAULT_BENCHMARKS_DIR = _REPO_ROOT / "xbow" / "benchmarks"
_DEFAULT_OUTPUT       = Path(__file__).parent / "splits.json"


# ── Data loading ───────────────────────────────────────────────────────────────

def _collect(benchmarks_dir: Path) -> list[dict]:
    records = []
    for bench_dir in sorted(benchmarks_dir.iterdir()):
        bj = bench_dir / "benchmark.json"
        if not bj.exists():
            continue
        data = json.loads(bj.read_text(encoding="utf-8"))
        tags = data.get("tags", [])
        records.append({
            "id":          bench_dir.name,
            "level":       int(data["level"]),
            "primary_tag": tags[0] if tags else "unknown",
            "name":        data.get("name", bench_dir.name),
        })
    return records


# ── Val selection: hardest example per tag, most categories first ──────────────

def _select_val(records: list[dict], val_size: int) -> set[str]:
    """Return a set of benchmark IDs for the val split.

    Per tag: take the hardest example (level desc, id asc for ties).
    From those per-tag winners: take the top val_size by difficulty.
    """
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_tag[r["primary_tag"]].append(r)

    # Best (hardest) representative per tag
    candidates: list[dict] = []
    for tag in sorted(by_tag):
        recs = sorted(by_tag[tag], key=lambda r: (-r["level"], r["id"]))
        candidates.append(recs[0])

    # Take the val_size hardest across all tags
    candidates.sort(key=lambda r: (-r["level"], r["primary_tag"]))
    return {r["id"] for r in candidates[:val_size]}


# ── Train/test split for remaining benchmarks ──────────────────────────────────

def _alloc_remaining(n: int) -> tuple[int, int]:
    """Return (n_train, n_test) for a stratum of size n (val already removed)."""
    if n <= 0:
        return (0, 0)
    if n <= 2:
        warnings.warn(f"Stratum of size {n} after val removal — all to train.")
        return (n, 0)
    n_te = max(1, round(n * 0.20))
    return (n - n_te, n_te)


# ── Core split logic ───────────────────────────────────────────────────────────

def create_splits(benchmarks_dir: Path, seed: int, val_size: int = VAL_SIZE) -> dict:
    records = _collect(benchmarks_dir)
    if not records:
        raise RuntimeError(f"No benchmark.json files found under {benchmarks_dir}")

    val_ids = _select_val(records, val_size)

    # Split remaining by (level, tag) stratum
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r["id"] not in val_ids:
            key = f"{r['level']}_{r['primary_tag']}"
            by_stratum[key].append(r["id"])

    rng = random.Random(seed)
    train, test = [], []
    strata_info: dict[str, dict] = {}

    for stratum_key in sorted(by_stratum.keys()):
        ids = list(by_stratum[stratum_key])
        rng.shuffle(ids)
        n_tr, n_te = _alloc_remaining(len(ids))
        strata_info[stratum_key] = {"total": len(ids), "train": n_tr, "test": n_te}
        train.extend(ids[:n_tr])
        test.extend(ids[n_tr:])

    val = sorted(val_ids)

    # Reconstruct per-stratum val counts for the meta block
    for r in records:
        if r["id"] in val_ids:
            key = f"{r['level']}_{r['primary_tag']}"
            if key not in strata_info:
                strata_info[key] = {"total": 0, "train": 0, "test": 0}
            strata_info[key]["total"] = strata_info[key].get("total", 0) + 1
            strata_info[key].setdefault("val", 0)
            strata_info[key]["val"] = strata_info[key].get("val", 0) + 1

    return {
        "_meta": {
            "seed":     seed,
            "total":    len(records),
            "strategy": "hardest_per_vuln_type_val_then_stratified_train_test",
            "generated_by": "source/dataset/create_splits.py",
            "split_counts": {
                "train": len(train),
                "val":   len(val),
                "test":  len(test),
            },
            "strata": strata_info,
        },
        "train": sorted(train),
        "val":   val,
        "test":  sorted(test),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed",           type=int,  default=SEED)
    parser.add_argument("--val-size",       type=int,  default=VAL_SIZE)
    parser.add_argument("--benchmarks-dir", type=Path, default=_DEFAULT_BENCHMARKS_DIR)
    parser.add_argument("--output",         type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = create_splits(args.benchmarks_dir, args.seed, args.val_size)
    meta   = result["_meta"]
    counts = meta["split_counts"]

    level_names = {"1": "Easy", "2": "Medium", "3": "Hard"}

    diff_summary: dict[str, dict] = {}
    for key, info in meta["strata"].items():
        level = key.split("_")[0]
        diff_name = level_names.get(level, f"Level{level}")
        agg = diff_summary.setdefault(diff_name, {"total": 0, "train": 0, "val": 0, "test": 0})
        agg["total"] += info.get("total", 0)
        agg["train"] += info.get("train", 0)
        agg["val"]   += info.get("val", 0)
        agg["test"]  += info.get("test", 0)

    # Show which tags made it into val and at what level
    val_ids = set(result["val"])
    from collections import defaultdict as _dd
    records = _collect(args.benchmarks_dir)
    val_records = sorted(
        [r for r in records if r["id"] in val_ids],
        key=lambda r: (-r["level"], r["primary_tag"]),
    )

    print(f"Strategy : {meta['strategy']}")
    print(f"Seed     : {meta['seed']}, Val size: {args.val_size}")
    print(f"\nVal set ({len(val_records)} examples, {len({r['primary_tag'] for r in val_records})} unique vuln types):")
    for r in val_records:
        diff = level_names.get(str(r["level"]), str(r["level"]))
        print(f"  {r['id']}  level={diff:<6}  tag={r['primary_tag']}")

    print(f"\n{'Difficulty':<10} {'Total':>7} {'Train':>7} {'Val':>7} {'Test':>7}")
    print("-" * 44)
    for diff, agg in sorted(diff_summary.items()):
        print(f"{diff:<10} {agg['total']:>7} {agg['train']:>7} {agg['val']:>7} {agg['test']:>7}")
    print("-" * 44)
    print(f"{'TOTAL':<10} {meta['total']:>7} {counts['train']:>7} {counts['val']:>7} {counts['test']:>7}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten → {args.output}")


if __name__ == "__main__":
    main()
