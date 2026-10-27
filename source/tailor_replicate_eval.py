#!/usr/bin/env python3
"""Run the `tailor` CTF agent (source/tailor/) N times against a single
xbow benchmark, to measure run-to-run consistency.

By default the N replicates run concurrently, each in its own isolated
Docker compose project (via `project_suffix` — see
source/benchmark.start_benchmark) so they don't collide on
container/network names or `docker ps` port lookups. Pass `--sequential`
to run them one at a time instead (removes any resource-contention
confound between replicates, at the cost of wall-clock time).

Output layout — with `--try NAME` (the normal case for the iterate-on-the-
prompt loop): experiments/tailor/<benchmark_id>/NAME/run<i>/<benchmark_id>/
{metadata,context_window}.json, experiments/tailor/<benchmark_id>/NAME/
runs_summary.json, and a snapshot of the exact agent config used at
experiments/tailor/<benchmark_id>/NAME/agent/ (prompt.md/opencode.json/
AGENTS.md/.opencode/), so each NAME is self-contained and reproducible
even after source/tailor/'s prompt.md is edited again for the next round.
Without `--try`, output goes straight to experiments/tailor/<benchmark_id>/
(no agent snapshot) — for quick one-off dry runs.

The agent-server (source/agent/) must already be running:
    docker compose -f source/agent/docker-compose.yml up -d --build

Usage:
    python3 -m source.tailor_replicate_eval --try 3
    python3 -m source.tailor_replicate_eval --try 3 --sequential
    python3 -m source.tailor_replicate_eval --benchmark XBEN-022-24 --runs 5
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AGENT_DIR = _REPO_ROOT / "source" / "tailor"
_AGENT_SERVER_URL = os.environ.get("AGENT_SERVER_URL", "http://localhost:8000")

load_dotenv()
os.environ.setdefault("DOCKER_DEFAULT_PLATFORM", "linux/amd64")

from source.benchmark import BENCHMARKS_DIR  # noqa: E402
from source.eval_harness import check_agent_server, evaluate_one, load_candidate, print_result  # noqa: E402

_AGENT_FILES = ("prompt.md", "opencode.json", "AGENTS.md", ".opencode")


def _snapshot_agent(out_root: Path) -> None:
    dest = out_root / "agent"
    dest.mkdir(parents=True, exist_ok=True)
    for name in _AGENT_FILES:
        src = _AGENT_DIR / name
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dest / name, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default="XBEN-022-24", help="benchmark id to replicate (default: XBEN-022-24)")
    parser.add_argument("--runs", type=int, default=5, help="number of replicate runs (default: 5)")
    parser.add_argument("--try", default=None, dest="try_n", help="save under experiments/tailor/<benchmark>/<NAME>/ with an agent-config snapshot (e.g. 'try6' or 'try6_duplicate')")
    parser.add_argument("--sequential", action="store_true", help="run replicates one at a time instead of concurrently")
    parser.add_argument("--agent-dir", default=None, help="agent config dir to run (default: source/tailor). e.g. source/seed for the baseline seed agent")
    parser.add_argument("--out-root", default=None, help="explicit output dir (overrides the default experiments/tailor/<benchmark>/...); always snapshots the agent config into <out-root>/agent/")
    args = parser.parse_args()

    check_agent_server(_AGENT_SERVER_URL)

    global _AGENT_DIR
    if args.agent_dir is not None:
        _AGENT_DIR = (_REPO_ROOT / args.agent_dir).resolve()

    candidate = load_candidate(_AGENT_DIR, frozenset())
    if args.out_root is not None:
        out_root = (_REPO_ROOT / args.out_root).resolve()
        _snapshot_agent(out_root)
    else:
        out_root = _REPO_ROOT / "experiments" / "tailor" / args.benchmark
        if args.try_n is not None:
            out_root = out_root / args.try_n
            _snapshot_agent(out_root)

    mode = "sequential" if args.sequential else "concurrent"
    print(f"[tailor-replicate-eval] {args.runs}x {args.benchmark} ({mode}) -> {out_root}\n")

    def _run(i: int) -> dict:
        r = evaluate_one(
            args.benchmark, candidate, out_root / f"run{i}",
            benchmarks_dir=BENCHMARKS_DIR, agent_server_url=_AGENT_SERVER_URL,
            project_suffix=f"-r{i}",
        )
        r["run"] = i
        return r

    results: list[dict] = []
    if args.sequential:
        for i in range(1, args.runs + 1):
            r = _run(i)
            results.append(r)
            print_result(r)
    else:
        with ThreadPoolExecutor(max_workers=args.runs) as pool:
            futures = {pool.submit(_run, i): i for i in range(1, args.runs + 1)}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                print_result(r)

    results.sort(key=lambda r: r["run"])
    passed = sum(1 for r in results if r["success"])
    pass_rate = round(passed / len(results), 3) if results else 0.0

    runs_summary = {
        "mean_pass_rate": pass_rate,
        "runs": {
            f"run{r['run']}": {
                "pass_rate": 1.0 if r["success"] else 0.0,
                "passed": 1 if r["success"] else 0,
                "total": 1,
                "total_cost_usd": r["total_cost_usd"],
                "total_tokens": r["total_tokens"],
                "stop_reason": r["stop_reason"],
            }
            for r in results
        },
    }
    (out_root / "runs_summary.json").write_text(json.dumps(runs_summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"{args.benchmark}: {passed}/{len(results)} passed ({pass_rate} pass rate)")
    print(f"[tailor-replicate-eval] done -> {out_root}")


if __name__ == "__main__":
    main()
