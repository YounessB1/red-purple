"""Run the seed CTF agent against a real xbow benchmark and save the OpenCode trace.

Same setup as GEPA: seed files from source/seed/, model and steps patched from CLI,
benchmark started via Docker. Calls runner.run() directly — no agent server needed.
Output (metadata.json + context_window.json) saved to
tests/runner/output/tool_call_formatting/<bench_id>[_runN]/ for later inspection.

Run with:
    python tests/runner/test_tool_call_formatting.py --bench XBEN-001-24
    python tests/runner/test_tool_call_formatting.py --bench XBEN-001-24 --model openrouter/qwen/qwen3.7-plus --steps 20 --runs 3
"""

import argparse
import json
import re
from pathlib import Path

from source.benchmark import start_benchmark, stop_benchmark
from source.agent.runner import run as run_agent

_ROOT = Path(__file__).resolve().parents[2]
_SEED_DIR = _ROOT / "source" / "seed"
_OUT_DIR = _ROOT / "tests" / "runner" / "output" / "tool_call_formatting"

DEFAULT_MODEL = "openrouter/google/gemma-4-31b-it"
DEFAULT_STEPS = 50


def _build_candidate(model: str, steps: int) -> dict:
    files: dict[str, str] = {}
    for f in sorted(_SEED_DIR.rglob("*")):
        if f.is_file() and f.name != ".gitkeep":
            files[str(f.relative_to(_SEED_DIR))] = f.read_text(encoding="utf-8")

    agent_md_key = ".opencode/agents/ctf-agent.md"
    if agent_md_key in files:
        md = files[agent_md_key]
        md = re.sub(r'(?m)^model:.*$', f'model: "{model}"', md)
        md = re.sub(r'(?m)^steps:.*$', f'steps: {steps}', md)
        files[agent_md_key] = md

    return {"files": files}


def _run_once(bench_id: str, candidate: dict) -> tuple[dict, list]:
    port = start_benchmark(bench_id)
    target = f"http://localhost:{port}"
    print(f"[test] Benchmark started — {target}")
    try:
        metadata, context_window = run_agent(target=target, candidate=candidate)
        return metadata, context_window
    finally:
        stop_benchmark(bench_id)
        print(f"[test] Benchmark stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", required=True, help="Benchmark ID, e.g. XBEN-001-24")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier (default: %(default)s)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Max agent steps (default: %(default)s)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
    args = parser.parse_args()

    print(f"[test] Bench  : {args.bench}")
    print(f"[test] Model  : {args.model}")
    print(f"[test] Steps  : {args.steps}")
    print(f"[test] Runs   : {args.runs}")

    candidate = _build_candidate(args.model, args.steps)

    for run_num in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n── Run {run_num}/{args.runs} {'─' * 50}")

        metadata, context_window = _run_once(args.bench, candidate)

        print(f"[test] stop_reason    : {metadata.get('stop_reason')}")
        print(f"[test] success        : {metadata.get('success')}")
        print(f"[test] iterations_used: {metadata.get('iterations_used')}")
        print(f"[test] tool_counts    : {metadata.get('tool_counts', {})}")

        suffix = f"_run{run_num}" if args.runs > 1 else ""
        out_dir = _OUT_DIR / f"{args.bench}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (out_dir / "context_window.json").write_text(json.dumps(context_window, indent=2), encoding="utf-8")
        print(f"[test] Saved to {out_dir}/")


if __name__ == "__main__":
    main()
