#!/usr/bin/env python3
"""Evaluate a finished GEPA experiment's best_candidate.json against exactly
the 30 benchmarks in source/dataset/split30.json's `train` split — the same
split the experiment itself trained on, and the same 30 IDs train_tailored
was evaluated against (see source/train_tailored_split_eval.py) — for a
single run, using the plain flag-match harness (source.eval_harness), not
GEPA's own judge/scorer machinery.

This exists to compare a GEPA-evolved candidate against the seed and
train_tailored on equal footing: same benchmarks, same success criterion
(exact flag match), same single-run methodology. It deliberately does NOT
use source/optimize_anything/eval_test_set.py, which evaluates the HELD-OUT
test split (39 benchmarks) through GEPA's evaluator.evaluate() (LLM-judge
scoring path) — different split, different scoring mechanism, not what
this comparison needs.

Output layout matches experiments/seed_think/ and experiments/train_tailored/:
a `run1/` subfolder holding per-benchmark metadata.json/context_window.json
plus summary.json, and a sibling runs_summary.json.

The agent-server (source/agent/) must already be running:
    docker compose -f source/agent/docker-compose.yml up -d --build

Usage:
    python3 -m source.gepa_best_split_eval experiments/split30_think_2
    python3 -m source.gepa_best_split_eval experiments/split30_think_copy --workers 6
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPLIT_PATH = _REPO_ROOT / "source" / "dataset" / "split30.json"
_AGENT_SERVER_URL = os.environ.get("AGENT_SERVER_URL", "http://localhost:8000")

load_dotenv()
os.environ.setdefault("DOCKER_DEFAULT_PLATFORM", "linux/amd64")

from source.benchmark import BENCHMARKS_DIR  # noqa: E402
from source.eval_harness import agent_model, check_agent_server, run_sweep  # noqa: E402
from source.optimize_anything import candidate_store  # noqa: E402


def _resolve_best_candidate(experiment_dir: Path) -> dict:
    best_path = experiment_dir / "best_candidate.json"
    if not best_path.exists():
        raise FileNotFoundError(f"{best_path} not found — has this experiment finished a run?")
    candidate = json.loads(best_path.read_text(encoding="utf-8"))
    files_ref = candidate.get("files")
    if isinstance(files_ref, str):
        candidate_store.configure(experiment_dir / ".candidates")
        files = candidate_store.load(files_ref)
        if files is None:
            raise RuntimeError(f"candidate hash {files_ref!r} not found in {experiment_dir / '.candidates'}")
        candidate = {**candidate, "files": files}
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", type=Path, help="e.g. experiments/split30_think_2")
    parser.add_argument("--workers", type=int, default=6, help="parallel workers (default: 6)")
    args = parser.parse_args()

    check_agent_server(_AGENT_SERVER_URL)

    experiment_dir = args.experiment_dir.resolve()
    candidate = _resolve_best_candidate(experiment_dir)
    print(f"[gepa-best-eval] {experiment_dir.name}: resolved best_candidate.json -> {len(candidate['files'])} files")

    train_ids = sorted(json.loads(_SPLIT_PATH.read_text())["train"])

    model_dir = experiment_dir / "train_eval" / agent_model(candidate).split("/")[-1]
    out_dir = model_dir / "run1"

    print(f"[gepa-best-eval] {len(train_ids)} train-split benchmark(s), {args.workers} worker(s) -> {out_dir}\n")

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
    print(f"\n[gepa-best-eval] done -> {model_dir}")


if __name__ == "__main__":
    main()
