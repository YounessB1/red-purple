"""Shared standalone evaluation harness — drives a fixed agent config against
a list of benchmarks via the containerized agent-server, without any of the
GEPA optimizer/judge/diagnoser machinery.

Extracted from source/train_tailored/evaluate.py so the same logic can be reused
by source/seed/evaluate.py (and any other fixed-agent-vs-benchmark-set
harness) instead of duplicating it. Every benchmark-source-specific detail
(which agent-config directory to load, which benchmarks_dir a given
benchmark id lives under, which benchmark ids to run, where to write output)
is passed in by the caller — this module owns none of it.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from source.benchmark import start_benchmark, stop_benchmark
from source.optimize_anything.utils import folder_to_dict


def check_agent_server(agent_server_url: str) -> None:
    try:
        urllib.request.urlopen(f"{agent_server_url}/docs", timeout=5)
    except urllib.error.HTTPError:
        pass  # any HTTP response (even 404) means the server is up and reachable
    except Exception as e:
        raise RuntimeError(
            f"agent-server not reachable at {agent_server_url} ({e}). "
            "Start it first: docker compose -f source/agent/docker-compose.yml up -d --build"
        ) from e


def run_via_server(
    agent_server_url: str,
    target: str,
    candidate: dict,
    expected_flag: str,
    extra_targets: list[str] | None = None,
) -> dict:
    # seed_json goes in the POST body, not the query string: a candidate's
    # combined file content routinely exceeds uvicorn/h11's ~16KB request-line
    # limit once an agent has accumulated more than a couple of skills, which
    # silently fails every run with "Invalid HTTP request received." (400)
    # rather than a clear size error.
    #
    # `extra_targets` carries any additional host-published services a
    # multi-port benchmark exposes (see benchmark.start_benchmark) as repeated
    # `extra=` params, so the agent is told about the full attack surface
    # rather than just the primary web port.
    query: dict = {"target": target, "expected_flag": expected_flag}
    if extra_targets:
        query["extra"] = extra_targets
    params = urllib.parse.urlencode(query, doseq=True)
    body = json.dumps(candidate).encode("utf-8")
    req = urllib.request.Request(
        f"{agent_server_url}/run?{params}", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"agent-server error {e.code} for {target}:\n{body}") from e


def load_candidate(agent_dir: Path, exclude: frozenset[str]) -> dict:
    files = folder_to_dict(agent_dir, exclude=exclude)
    return {"files": files}


def agent_model(candidate: dict) -> str:
    """Read the `model:` frontmatter field from .opencode/agents/ctf-agent.md."""
    content = candidate["files"].get(".opencode/agents/ctf-agent.md", "")
    m = re.search(r'(?m)^model:\s*"?([^"\n]+)"?', content)
    return m.group(1).strip() if m else "unknown-model"


def load_benchmark_meta(benchmarks_dir: Path, benchmark_id: str) -> dict:
    meta_path = benchmarks_dir / benchmark_id / "benchmark.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"level": str(meta.get("level")), "tags": meta.get("tags", [])}


def evaluate_one(
    benchmark_id: str,
    candidate: dict,
    out_dir: Path,
    *,
    benchmarks_dir: Path,
    agent_server_url: str,
    project_suffix: str = "",
) -> dict:
    """Run one benchmark and save <out_dir>/<benchmark_id>/{metadata,context_window}.json.

    `project_suffix` isolates this call's Docker containers under their own
    compose project (see benchmark.start_benchmark) — pass a distinct value
    per call when running several concurrent instances of the *same*
    benchmark_id (e.g. N replicate runs), otherwise leave it as the default.
    """
    meta = load_benchmark_meta(benchmarks_dir, benchmark_id)
    start = time.monotonic()
    context_window: list = []
    try:
        port, extra_ports, expected_flag = start_benchmark(
            benchmark_id, benchmarks_dir=benchmarks_dir, project_suffix=project_suffix
        )
        try:
            artifacts = run_via_server(
                agent_server_url, f"http://localhost:{port}", candidate, expected_flag,
                extra_targets=[f"localhost:{p}" for p in extra_ports],
            )
            run_metadata = artifacts["metadata"]
            context_window = artifacts.get("context_window", [])
        finally:
            stop_benchmark(benchmark_id, benchmarks_dir=benchmarks_dir, project_suffix=project_suffix)
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


def print_result(r: dict) -> None:
    status = "PASS" if r["success"] else "FAIL"
    print(
        f"[{status}] {r['benchmark_id']:<14} lvl={r['level']} "
        f"tags={','.join(r['tags']):<40} ({r['duration_seconds']}s)"
        + (f"  -- {r['stop_reason']}" if not r["success"] else "")
    )


def aggregate_by(results: list[dict], key: str) -> dict:
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


def run_sweep(
    benchmark_ids: list[str],
    candidate: dict,
    out_dir: Path,
    *,
    benchmarks_dir: Path,
    agent_server_url: str,
    workers: int,
    limit: int | None = None,
) -> list[dict]:
    """Run `evaluate_one` over `benchmark_ids`, sequentially or via a thread pool."""
    if limit:
        benchmark_ids = benchmark_ids[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] {len(benchmark_ids)} benchmark(s), {workers} worker(s) -> {out_dir}\n")

    results: list[dict] = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    evaluate_one, bid, candidate, out_dir,
                    benchmarks_dir=benchmarks_dir, agent_server_url=agent_server_url,
                ): bid
                for bid in benchmark_ids
            }
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                print_result(r)
    else:
        for bid in benchmark_ids:
            r = evaluate_one(
                bid, candidate, out_dir,
                benchmarks_dir=benchmarks_dir, agent_server_url=agent_server_url,
            )
            results.append(r)
            print_result(r)

    passed = sum(1 for r in results if r["success"])
    total_cost = round(sum(r["total_cost_usd"] for r in results), 4)
    total_tokens = sum(r["total_tokens"] for r in results)

    by_tag = aggregate_by(results, "tags")
    by_level = aggregate_by(results, "level")

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
    print(f"\n[eval] results written to {out_path}")

    return results
