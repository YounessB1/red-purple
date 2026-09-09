#!/usr/bin/env python3
"""Standalone evaluation harness for the GEPA seed CTF agent (source/seed/).

Same lifecycle as source/baseline/evaluate.py (Docker benchmark for XBOW +
containerized agent-server for the OpenCode agent run, strict flag-match
success) but evaluates the unmodified seed candidate GEPA starts from, and
repeats the full benchmark set across N independent runs so pass-rate
variance can be measured.

The agent-server (source/agent/) must already be running:
    docker compose -f source/agent/docker-compose.yml up -d --build

Usage:
    python3 -m source.seed_eval.evaluate                        # 5 runs, all 104 benchmarks
    python3 -m source.seed_eval.evaluate --runs 3
    python3 -m source.seed_eval.evaluate --workers 15
    python3 -m source.seed_eval.evaluate --benchmarks XBEN-001-24 --runs 1
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from source.baseline.evaluate import (
    _BENCHMARKS_DIR,
    _agent_model,
    _aggregate_by,
    _check_agent_server,
    _print_result,
    evaluate_one,
)
from source.optimize_anything.utils import folder_to_dict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEED_DIR = _REPO_ROOT / "source" / "seed"


def _load_candidate() -> dict:
    # No exclude filter: source/seed/ holds only candidate files (matches
    # how core_loop._build_seed_candidate() reads this same directory).
    return {"files": folder_to_dict(_SEED_DIR)}


def run_once(benchmark_ids: list[str], candidate: dict, workers: int, out_dir: Path) -> dict:
    """Run every benchmark once and write <out_dir>/summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(evaluate_one, bid, candidate, out_dir): bid for bid in benchmark_ids}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                _print_result(r)
    else:
        for bid in benchmark_ids:
            r = evaluate_one(bid, candidate, out_dir)
            results.append(r)
            _print_result(r)

    passed = sum(1 for r in results if r["success"])
    total_cost = round(sum(r["total_cost_usd"] for r in results), 4)
    total_tokens = sum(r["total_tokens"] for r in results)
    by_tag = _aggregate_by(results, "tags")
    by_level = _aggregate_by(results, "level")

    summary = {
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
        "passed": passed,
        "total": len(results),
        "total_cost_usd": total_cost,
        "total_tokens": total_tokens,
        "by_tag": by_tag,
        "by_level": by_level,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _print_summary(summary: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"PASSED: {summary['passed']}/{summary['total']}  ({summary['pass_rate']} pass rate)")
    print(f"cost: ${summary['total_cost_usd']}  tokens: {summary['total_tokens']}")
    print("\nBy tag:")
    for tag, stats in sorted(summary["by_tag"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {tag:<25} {stats['solved']}/{stats['n']}  ({stats['pass_rate']})")
    print("\nBy level:")
    for level, stats in summary["by_level"].items():
        print(f"  {level:<25} {stats['solved']}/{stats['n']}  ({stats['pass_rate']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", nargs="*", default=None, help="specific benchmark IDs (default: all)")
    parser.add_argument("--workers", type=int, default=1, help="parallel workers per run (default: 1)")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N benchmarks (dry run)")
    parser.add_argument("--runs", type=int, default=5, help="number of independent full-dataset runs (default: 5)")
    parser.add_argument("--start-run", type=int, default=1, help="first run index, for resuming (default: 1)")
    args = parser.parse_args()

    _check_agent_server()

    if args.benchmarks:
        benchmark_ids = args.benchmarks
    else:
        benchmark_ids = sorted(
            p.name for p in _BENCHMARKS_DIR.iterdir()
            if p.is_dir() and (p / "benchmark.json").exists()
        )
    if args.limit:
        benchmark_ids = benchmark_ids[: args.limit]

    candidate = _load_candidate()
    model_dir = _REPO_ROOT / "experiments" / "seed" / _agent_model(candidate).split("/")[-1]
    last_run = args.start_run + args.runs - 1

    print(
        f"[seed-eval] {len(benchmark_ids)} benchmark(s) x {args.runs} run(s) "
        f"({args.workers} worker(s) each) -> {model_dir}\n"
    )

    run_summaries = {}
    for run_idx in range(args.start_run, last_run + 1):
        out_dir = model_dir / f"run{run_idx}"
        print(f"\n{'#' * 60}\n[seed-eval] run {run_idx}/{last_run} -> {out_dir}\n{'#' * 60}")
        summary = run_once(benchmark_ids, candidate, args.workers, out_dir)
        _print_summary(summary)
        run_summaries[f"run{run_idx}"] = {
            "pass_rate": summary["pass_rate"],
            "passed": summary["passed"],
            "total": summary["total"],
            "total_cost_usd": summary["total_cost_usd"],
            "total_tokens": summary["total_tokens"],
        }

    pass_rates = [s["pass_rate"] for s in run_summaries.values()]
    mean_pass_rate = round(sum(pass_rates) / len(pass_rates), 3) if pass_rates else 0.0
    print(f"\n{'=' * 60}\n[seed-eval] all {args.runs} run(s) complete -> {model_dir}")
    print(f"[seed-eval] mean pass rate across runs: {mean_pass_rate}  ({pass_rates})")

    (model_dir / "runs_summary.json").write_text(
        json.dumps({"mean_pass_rate": mean_pass_rate, "runs": run_summaries}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
