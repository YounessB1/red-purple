#!/usr/bin/env python3
"""Create a reproducible stratified train/val/test split for the XBOW benchmark suite.

Stratification is by difficulty × primary vulnerability type, using a fixed
random seed. Metadata is read from each benchmark's benchmark.json (field "level"
for difficulty, first entry of "tags" for vulnerability type).

Split target: ~70-75% train / ~10% val / ~20% test
  Singletons  (n=1):  1 / 0 / 0  → all to train
  Pairs       (n=2):  2 / 0 / 0  → all to train
  Triples     (n=3):  2 / 0 / 1  → train + test only
  n≥4               : 1 val, ~20% test, rest train

Usage:
    python3 source/dataset/create_splits.py
    python3 source/dataset/create_splits.py --seed 42 --benchmarks-dir xbow/benchmarks
"""

import argparse
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

SEED = 42

_REPO_ROOT       = Path(__file__).resolve().parents[2]
_DEFAULT_BENCHMARKS_DIR = _REPO_ROOT / "xbow" / "benchmarks"
_DEFAULT_OUTPUT  = Path(__file__).parent / "splits.json"


# ── Allocation helper ──────────────────────────────────────────────────────────

def _alloc(n: int) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) for a stratum of size n."""
    if n == 0:
        return (0, 0, 0)
    if n == 1:
        warnings.warn("Stratum of size 1 — assigning to train.")
        return (1, 0, 0)
    if n == 2:
        warnings.warn("Stratum of size 2 — assigning both to train.")
        return (2, 0, 0)
    if n == 3:
        return (2, 0, 1)
    # n >= 4: exactly 1 val, ~20% test, rest train
    n_va = 1
    n_te = max(1, round(n * 0.20))
    n_tr = n - n_va - n_te
    return (n_tr, n_va, n_te)


# ── Data loading ───────────────────────────────────────────────────────────────

def _collect(benchmarks_dir: Path) -> list[dict]:
    """Load metadata from every benchmark.json, sorted by benchmark ID."""
    records = []
    for bench_dir in sorted(benchmarks_dir.iterdir()):
        bj = bench_dir / "benchmark.json"
        if not bj.exists():
            continue
        data = json.loads(bj.read_text(encoding="utf-8"))
        tags = data.get("tags", [])
        records.append({
            "id":          bench_dir.name,
            "level":       str(data["level"]),       # normalize int/str → "1"/"2"/"3"
            "primary_tag": tags[0] if tags else "unknown",
            "all_tags":    tags,
            "name":        data.get("name", bench_dir.name),
        })
    return records


# ── Core split logic ───────────────────────────────────────────────────────────

def create_splits(benchmarks_dir: Path, seed: int) -> dict:
    records = _collect(benchmarks_dir)
    if not records:
        raise RuntimeError(f"No benchmark.json files found under {benchmarks_dir}")

    # Group IDs by (difficulty level, primary vulnerability type)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for r in records:
        key = f"{r['level']}_{r['primary_tag']}"
        by_stratum[key].append(r["id"])

    rng = random.Random(seed)
    train, val, test = [], [], []
    strata_info: dict[str, dict] = {}

    for stratum_key in sorted(by_stratum.keys()):
        ids = list(by_stratum[stratum_key])
        rng.shuffle(ids)
        n = len(ids)
        n_tr, n_va, n_te = _alloc(n)

        strata_info[stratum_key] = {
            "total": n, "train": n_tr, "val": n_va, "test": n_te
        }

        train.extend(ids[:n_tr])
        val.extend(ids[n_tr : n_tr + n_va])
        test.extend(ids[n_tr + n_va :])

    return {
        "_meta": {
            "seed":           seed,
            "total":          len(records),
            "strategy":       "difficulty_x_vuln_type_stratified",
            "generated_by":   "source/dataset/create_splits.py",
            "split_counts": {
                "train": len(train),
                "val":   len(val),
                "test":  len(test),
            },
            "strata": strata_info,
        },
        "train": sorted(train),
        "val":   sorted(val),
        "test":  sorted(test),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed",           type=int,  default=SEED)
    parser.add_argument("--benchmarks-dir", type=Path, default=_DEFAULT_BENCHMARKS_DIR)
    parser.add_argument("--output",         type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = create_splits(args.benchmarks_dir, args.seed)
    meta   = result["_meta"]
    counts = meta["split_counts"]

    level_names = {"1": "Easy", "2": "Medium", "3": "Hard"}

    # Aggregate totals per difficulty for a readable summary
    diff_summary: dict[str, dict] = {}
    for key, info in meta["strata"].items():
        level = key.split("_")[0]
        diff_name = level_names.get(level, f"Level{level}")
        agg = diff_summary.setdefault(diff_name, {"total": 0, "train": 0, "val": 0, "test": 0})
        for k in ("total", "train", "val", "test"):
            agg[k] += info[k]

    print(f"Strategy : {meta['strategy']}")
    print(f"Seed     : {meta['seed']}")
    print(f"Strata   : {len(meta['strata'])} (difficulty × primary vuln type)\n")
    print(f"{'Difficulty':<10} {'Total':>7} {'Train':>7} {'Val':>7} {'Test':>7}")
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
