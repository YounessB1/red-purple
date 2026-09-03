#!/usr/bin/env python3
"""Standalone evaluation harness for the hand-authored baseline CTF agent.

Runs the agent defined in this folder (prompt.md/AGENTS.md/skills) against
XBOW benchmarks, driving the same lifecycle GEPA evaluation uses
(source.benchmark for Docker; the containerized agent-server for the
OpenCode agent run) without any of the optimizer/judge/diagnoser
machinery — this is for measuring one fixed, hand-built agent, not
evolving one.

The agent MUST run inside the `agent-server` container (source/agent/),
not as a bare `opencode` subprocess on the host: that container is a Kali
image with sqlmap/nmap/hydra/gobuster/etc. preinstalled, which several of
this agent's skills assume are available. Start it first if it isn't
already running:
    docker compose -f source/agent/docker-compose.yml up -d --build

Usage:
    python3 -m source.baseline.evaluate                       # all 104 benchmarks
    python3 -m source.baseline.evaluate --benchmarks XBEN-001-24 XBEN-004-24
    python3 -m source.baseline.evaluate --workers 6
    python3 -m source.baseline.evaluate --limit 5              # cheap dry run
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = Path(__file__).resolve().parent
_BENCHMARKS_DIR = _REPO_ROOT / "xbow" / "benchmarks"
_AGENT_SERVER_URL = os.environ.get("AGENT_SERVER_URL", "http://localhost:8000")

load_dotenv()

# mysql:5.7 and other old base images were never published for arm64.
# Force amd64 + emulation instead of requiring every benchmark's
# docker-compose.yml to be patched (see xbow/test_benchmarks.py). No-op on
# native linux/amd64 hosts, and scoped to this process only.
os.environ.setdefault("DOCKER_DEFAULT_PLATFORM", "linux/amd64")

from source.benchmark import start_benchmark, stop_benchmark  # noqa: E402
from source.optimize_anything.utils import folder_to_dict  # noqa: E402

_EXCLUDE = frozenset({"evaluate.py", "__init__.py", "results", "__pycache__"})


def _check_agent_server() -> None:
    try:
        urllib.request.urlopen(f"{_AGENT_SERVER_URL}/docs", timeout=5)
    except urllib.error.HTTPError:
        pass  # any HTTP response (even 404) means the server is up and reachable
    except Exception as e:
        raise RuntimeError(
            f"agent-server not reachable at {_AGENT_SERVER_URL} ({e}). "
            "Start it first: docker compose -f source/agent/docker-compose.yml up -d --build"
        ) from e


def _run_via_server(target: str, candidate: dict, expected_flag: str) -> dict:
    params = urllib.parse.urlencode({
        "target": target,
        "seed_json": json.dumps(candidate),
        "expected_flag": expected_flag,
    })
    req = urllib.request.Request(f"{_AGENT_SERVER_URL}/run?{params}", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"agent-server error {e.code} for {target}:\n{body}") from e


def _load_candidate() -> dict:
    files = folder_to_dict(_BASELINE_DIR, exclude=_EXCLUDE)
    return {"files": files}


def _agent_model(candidate: dict) -> str:
    """Read the `model:` frontmatter field from .opencode/agents/ctf-agent.md."""
    content = candidate["files"].get(".opencode/agents/ctf-agent.md", "")
    m = re.search(r'(?m)^model:\s*"?([^"\n]+)"?', content)
    return m.group(1).strip() if m else "unknown-model"


def _load_benchmark_meta(benchmark_id: str) -> dict:
    meta_path = _BENCHMARKS_DIR / benchmark_id / "benchmark.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"level": str(meta.get("level")), "tags": meta.get("tags", [])}


def evaluate_one(benchmark_id: str, candidate: dict, out_dir: Path) -> dict:
    """Run one benchmark and save <out_dir>/<benchmark_id>/{metadata,context_window}.json."""
    meta = _load_benchmark_meta(benchmark_id)
    start = time.monotonic()
    context_window: list = []
    try:
        port, expected_flag = start_benchmark(benchmark_id)
        try:
            artifacts = _run_via_server(f"http://localhost:{port}", candidate, expected_flag)
            run_metadata = artifacts["metadata"]
            context_window = artifacts.get("context_window", [])
        finally:
            stop_benchmark(benchmark_id)
    except Exception as e:
        run_metadata = {
            "success": False,
            "stop_reason": "error",
            "error_detail": f"{type(e).__name__}: {e}",
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }
    duration = round(time.monotonic() - start, 1)
    run_metadata = {
        **run_metadata,
        "benchmark_id": benchmark_id,
        "eval_duration_seconds": duration,
        "level": meta["level"],
        "tags": meta["tags"],
    }

    bench_dir = out_dir / benchmark_id
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (bench_dir / "context_window.json").write_text(
        json.dumps(context_window, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "benchmark_id": benchmark_id,
        "success": run_metadata.get("success", False),
        "stop_reason": run_metadata.get("stop_reason"),
        "duration_seconds": duration,
        "total_tokens": run_metadata.get("total_tokens", 0),
        "total_cost_usd": run_metadata.get("total_cost_usd", 0.0),
        "level": meta["level"],
        "tags": meta["tags"],
    }


def _print_result(r: dict) -> None:
    status = "PASS" if r["success"] else "FAIL"
    print(
        f"[{status}] {r['benchmark_id']:<14} lvl={r['level']} "
        f"tags={','.join(r['tags']):<40} ({r['duration_seconds']}s)"
        + (f"  -- {r['stop_reason']}" if not r["success"] else "")
    )


def _aggregate_by(results: list[dict], key: str) -> dict:
    buckets: dict[str, dict] = {}
    for r in results:
        values = r[key] if isinstance(r[key], list) else [r[key]]
        for v in values:
            b = buckets.setdefault(v, {"n": 0, "solved": 0})
            b["n"] += 1
            b["solved"] += 1 if r["success"] else 0
    return {
        k: {**v, "pass_rate": round(v["solved"] / v["n"], 3) if v["n"] else 0.0}
        for k, v in sorted(buckets.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", nargs="*", default=None, help="specific benchmark IDs (default: all)")
    parser.add_argument("--workers", type=int, default=1, help="parallel workers (default: 1)")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N benchmarks (dry run)")
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
    out_dir = _REPO_ROOT / "experiments" / "baseline" / _agent_model(candidate).split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[baseline-eval] {len(benchmark_ids)} benchmark(s), {args.workers} worker(s) -> {out_dir}\n")

    results: list[dict] = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
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

    print(f"\n{'=' * 60}")
    print(f"PASSED: {passed}/{len(results)}  ({round(passed / len(results), 3) if results else 0.0} pass rate)")
    print(f"cost: ${total_cost}  tokens: {total_tokens}")
    print("\nBy tag:")
    for tag, stats in sorted(by_tag.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {tag:<25} {stats['solved']}/{stats['n']}  ({stats['pass_rate']})")
    print("\nBy level:")
    for level, stats in by_level.items():
        print(f"  {level:<25} {stats['solved']}/{stats['n']}  ({stats['pass_rate']})")

    out_path = out_dir / "summary.json"
    out_path.write_text(
        json.dumps(
            {
                "pass_rate": round(passed / len(results), 3) if results else 0.0,
                "passed": passed,
                "total": len(results),
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "by_tag": by_tag,
                "by_level": by_level,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[baseline-eval] results written to {out_path}")


if __name__ == "__main__":
    main()
