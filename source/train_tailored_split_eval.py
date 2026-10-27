#!/usr/bin/env python3
"""Evaluate the train-set-tailored agent (source/train_tailored/) against
exactly the 30 benchmarks in source/dataset/split30.json's `train` split —
the same 30 IDs its skills were written from (see
source/train_tailored/PROVENANCE.md) — for a single run.

Output layout matches experiments/seed_think/: a `run1/` subfolder holding
per-benchmark metadata.json/context_window.json plus summary.json, and a
sibling runs_summary.json at the model-dir level (a single-entry "runs"
dict here, since this evaluates one run, not N).

The agent-server (source/agent/) must already be running:
    docker compose -f source/agent/docker-compose.yml up -d --build

Usage:
    python3 -m source.train_tailored_split_eval
    python3 -m source.train_tailored_split_eval --workers 6
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AGENT_DIR = _REPO_ROOT / "source" / "train_tailored"
_SPLIT_PATH = _REPO_ROOT / "source" / "dataset" / "split30.json"
_AGENT_SERVER_URL = os.environ.get("AGENT_SERVER_URL", "http://localhost:8000")

load_dotenv()
os.environ.setdefault("DOCKER_DEFAULT_PLATFORM", "linux/amd64")

from source.benchmark import BENCHMARKS_DIR  # noqa: E402
from source.eval_harness import agent_model, check_agent_server, load_candidate, run_sweep  # noqa: E402

_EXCLUDE = frozenset({"evaluate.py", "__init__.py", "results", "__pycache__", "PROVENANCE.md"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6, help="parallel workers (default: 6)")
    args = parser.parse_args()

    check_agent_server(_AGENT_SERVER_URL)

    train_ids = sorted(json.loads(_SPLIT_PATH.read_text())["train"])

    candidate = load_candidate(_AGENT_DIR, _EXCLUDE)
    model_dir = _REPO_ROOT / "experiments" / "train_tailored" / agent_model(candidate).split("/")[-1]
    out_dir = model_dir / "run1"

    print(f"[train-split-eval] {len(train_ids)} train-split benchmark(s), {args.workers} worker(s) -> {out_dir}\n")

    results = run_sweep(
        train_ids, candidate, out_dir,
        benchmarks_dir=BENCHMARKS_DIR, agent_server_url=_AGENT_SERVER_URL,
        workers=args.workers,
    )

    passed = sum(1 for r in results if r["success"])
    total_cost = round(sum(r["total_cost_usd"] for r in results), 4)
    total_tokens = sum(r["total_tokens"] for r in results)
    pass_rate = round(passed / len(results), 3) if results else 0.0

    runs_summary = {
        "mean_pass_rate": pass_rate,
        "runs": {
            "run1": {
                "pass_rate": pass_rate,
                "passed": passed,
                "total": len(results),
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
            }
        },
    }
    (model_dir / "runs_summary.json").write_text(json.dumps(runs_summary, indent=2), encoding="utf-8")
    print(f"\n[train-split-eval] done -> {model_dir}")


if __name__ == "__main__":
    main()
