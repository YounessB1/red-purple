#!/usr/bin/env python3
"""Evaluate an experiment's seed and best candidates against the held-out test split.

The GEPA optimization loop only ever sees train/val benchmarks (see
`source/optimize_anything/dataset.py::load_dataset`). This script reuses the
exact same `evaluator.evaluate()` call GEPA itself uses, run against the
`test` split of the experiment's own splits file, for the seed agent
(`source/seed/`) and/or the run's `best_candidate.json`.

Usage:
    python3 -m source.optimize_anything.eval_test_set experiments/split30
    python3 -m source.optimize_anything.eval_test_set experiments/split30 --candidates best --workers 10
    python3 -m source.optimize_anything.eval_test_set experiments/split30 --limit 3   # cheap dry run

Output: <experiment_dir>/test/{seed,best}/{agent/, results.json, experiment_summary.json,
iteration_000/test/<bench_id>/...} and <experiment_dir>/test/summary.json when both
candidates are evaluated.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEED_DIR = _REPO_ROOT / "source" / "seed"
_DATASET_DIR = _REPO_ROOT / "source" / "dataset"

# Standalone entrypoint — unlike main.py, doesn't otherwise inherit the main
# GEPA run's dotenv loading, so OPENROUTER_API_KEY (needed by the scorer,
# which runs as a local `opencode run` subprocess) would only come from a
# real shell env var otherwise, even when it's set in repo-root .env.
load_dotenv()

# Many benchmark base images (e.g. mysql:5.7) were never published for
# arm64. Force amd64 + emulation instead of requiring every benchmark's
# docker-compose.yml to be patched with `platform: linux/amd64`. Scoped to
# this process's subprocess calls only (source.benchmark's docker/make
# invocations inherit it) — doesn't touch the caller's shell environment,
# and is a no-op on native linux/amd64 hosts (see xbow/test_benchmarks.py).
os.environ.setdefault("DOCKER_DEFAULT_PLATFORM", "linux/amd64")

from source.optimize_anything import cache, candidate_store, evaluator
from source.optimize_anything.dataset import load_test_split
from source.optimize_anything.logger import Logger
from source.optimize_anything.utils import candidate_hash, dict_to_folder, folder_to_dict


def _resolve_named_candidate(name: str, experiment_dir: Path) -> dict:
    if name == "seed":
        seed_files = folder_to_dict(_SEED_DIR)
        seed_hash = candidate_hash(seed_files)
        candidate_store.store(seed_hash, seed_files)
        return {"files": seed_hash}
    if name == "best":
        best_path = experiment_dir / "best_candidate.json"
        if not best_path.exists():
            raise FileNotFoundError(f"{best_path} not found — has this experiment finished a run?")
        return json.loads(best_path.read_text(encoding="utf-8"))
    raise ValueError(f"unknown candidate name {name!r} (expected 'seed' or 'best')")


def _aggregate(per_bench: list[dict]) -> dict:
    n = len(per_bench)
    solved = sum(1 for r in per_bench if r["success"])
    by_level: dict[str, dict] = {}
    by_family: dict[str, dict] = {}
    for r in per_bench:
        lvl = str(r["level"])
        by_level.setdefault(lvl, {"n": 0, "solved": 0})
        by_level[lvl]["n"] += 1
        by_level[lvl]["solved"] += int(r["success"])
        fam = r["family"] or "unknown"
        by_family.setdefault(fam, {"n": 0, "solved": 0})
        by_family[fam]["n"] += 1
        by_family[fam]["solved"] += int(r["success"])
    return {
        "n": n,
        "solved": solved,
        "pass_rate": round(solved / n, 4) if n else 0.0,
        "avg_score": round(sum(r["score"] for r in per_bench) / n, 4) if n else 0.0,
        "by_level": by_level,
        "by_family": by_family,
    }


def _run_sweep(
    name: str,
    candidate: dict,
    examples: list[dict],
    test_root: Path,
    *,
    agent_model: str,
    agent_max_iter: int,
    judge_model: str,
    gt: bool,
    workers: int,
    splits_meta: dict,
) -> dict:
    candidate_dir = test_root / name

    files_ref = candidate.get("files")
    files = candidate_store.load(files_ref) if isinstance(files_ref, str) else files_ref
    if files is None:
        raise RuntimeError(f"could not resolve files for candidate {name!r} (hash {files_ref!r} not in store)")
    dict_to_folder(candidate_dir / "agent", files)

    logger = Logger(
        reflector_model="",
        judge_model=judge_model,
        agent_model=agent_model,
        diagnoser_model="",
        log_dir=candidate_dir,
    )
    logger.start_logger()

    evaluator.configure_runtime(
        experiment_dir=candidate_dir,
        agent_max_iter=agent_max_iter,
        agent_model=agent_model,
        judge_model=judge_model,
        diagnoser_model="",
        reflector_model="",
        train_size=0,
        gt=gt,
        diagnoser_gt=False,
        logger=logger,
    )
    evaluator.set_gepa_role("val")
    evaluator.set_gepa_iteration(0)

    def _eval_one(example: dict) -> tuple[str, float, dict]:
        bench_id = example.get("benchmark_id", "unknown")
        try:
            score, side_info = evaluator.evaluate(candidate, example)
            return bench_id, score, side_info
        except Exception as e:
            print(f"[test-eval:{name}] {bench_id} — evaluation error: {e}")
            return bench_id, 0.0, {"benchmark_id": bench_id, "error": str(e)}

    if workers > 1 and len(examples) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            raw_results = list(ex.map(_eval_one, examples))
    else:
        raw_results = [_eval_one(e) for e in examples]

    logger.stop_logger()

    per_bench = []
    for bench_id, score, side_info in raw_results:
        meta = splits_meta.get(bench_id, {})
        per_bench.append({
            "benchmark_id": bench_id,
            "level": meta.get("level"),
            "family": meta.get("family"),
            "tags": meta.get("tags", []),
            "score": score,
            "success": bool(side_info.get("success")) if "error" not in side_info else False,
            "stop_reason": side_info.get("stop_reason"),
        })

    aggregate = _aggregate(per_bench)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "results.json").write_text(
        json.dumps({"candidate": name, "results": per_bench, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    print(f"[test-eval:{name}] pass_rate={aggregate['pass_rate']} ({aggregate['solved']}/{aggregate['n']})")
    return {"per_bench": per_bench, "aggregate": aggregate}


def _write_top_summary(test_root: Path, seed_result: dict, best_result: dict) -> None:
    seed_by_id = {r["benchmark_id"]: r for r in seed_result["per_bench"]}
    best_by_id = {r["benchmark_id"]: r for r in best_result["per_bench"]}
    diff = {"solved_by_both": [], "solved_by_seed_only": [], "solved_by_best_only": [], "solved_by_neither": []}
    for bench_id, seed_row in seed_by_id.items():
        s = seed_row["success"]
        b = best_by_id.get(bench_id, {}).get("success", False)
        if s and b:
            diff["solved_by_both"].append(bench_id)
        elif s and not b:
            diff["solved_by_seed_only"].append(bench_id)
        elif b and not s:
            diff["solved_by_best_only"].append(bench_id)
        else:
            diff["solved_by_neither"].append(bench_id)
    summary = {
        "seed": seed_result["aggregate"],
        "best": best_result["aggregate"],
        "delta_pass_rate": round(best_result["aggregate"]["pass_rate"] - seed_result["aggregate"]["pass_rate"], 4),
        "diff": diff,
    }
    test_root.mkdir(parents=True, exist_ok=True)
    (test_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[test-eval] summary written to {test_root / 'summary.json'} (delta_pass_rate={summary['delta_pass_rate']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", type=Path, help="e.g. experiments/split30")
    parser.add_argument("--candidates", default="seed,best", help="comma-separated: seed,best")
    parser.add_argument("--workers", type=int, default=None, help="defaults to the experiment's own config.yaml workers")
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N test benchmarks (dry-run)")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    experiments_dir = experiment_dir.parent
    config = yaml.safe_load((experiment_dir / "config.yaml").read_text(encoding="utf-8"))

    splits_name = config["splits"]
    agent_model = config["ctf_agent"]["md"]["model"]
    agent_max_iter = config["ctf_agent"]["md"]["steps"]
    judge_model = config["scorer"]["md"].get("model", "")
    gt = config["scorer"].get("gt", False)
    workers = args.workers or config.get("workers", 10)

    candidate_store.configure(experiment_dir / ".candidates")
    cache.CACHE_DIR = experiments_dir / ".eval_cache"

    test_examples = load_test_split(splits_name=splits_name)
    if args.limit:
        test_examples = test_examples[: args.limit]

    splits_meta = json.loads((_DATASET_DIR / f"{splits_name}.json").read_text(encoding="utf-8"))["_meta"]["assignments"]

    wanted = [c.strip() for c in args.candidates.split(",") if c.strip()]
    test_root = experiment_dir / "test"

    print(f"[test-eval] experiment={experiment_dir.name} splits={splits_name} test_n={len(test_examples)} "
          f"candidates={wanted} workers={workers}")

    results: dict[str, dict] = {}
    for name in wanted:
        candidate = _resolve_named_candidate(name, experiment_dir)
        results[name] = _run_sweep(
            name, candidate, test_examples, test_root,
            agent_model=agent_model, agent_max_iter=agent_max_iter,
            judge_model=judge_model, gt=gt, workers=workers,
            splits_meta=splits_meta,
        )

    if "seed" in results and "best" in results:
        _write_top_summary(test_root, results["seed"], results["best"])


if __name__ == "__main__":
    main()
