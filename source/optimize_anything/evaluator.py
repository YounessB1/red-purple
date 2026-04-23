"""GEPA evaluator — scores a candidate seed against a single benchmark."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import gepa.optimize_anything as oa

from source.agent.runner import run as run_agent
from source.agent.seed import TOOL_SCHEMAS, TOOLS
from source.benchmark import start_benchmark, stop_benchmark
from source.optimize_anything import cache

# Set by core_loop before optimization starts
EXPERIMENT_DIR: Path | None = None
AGENT_MAX_ITER: int = 50
AGENT_MODEL: str = ""

# Iteration tracking — each unique candidate hash = one GEPA iteration
_seen_candidates: dict[str, int] = {}
_iteration_counter: int = 0


def _get_iteration(candidate_hash: str) -> int:
    """Return the iteration number for a candidate, assigning a new one if first seen."""
    global _iteration_counter
    if candidate_hash not in _seen_candidates:
        _iteration_counter += 1
        _seen_candidates[candidate_hash] = _iteration_counter
    return _seen_candidates[candidate_hash]


def evaluate(candidate: dict[str, str], example: dict) -> tuple[float, dict]:
    """GEPA evaluator for generalization mode.

    Args:
        candidate: {"system_prompt": str, "default_task": str}
        example:   {"benchmark_id": str, "level": str, "tags": list[str], "name": str}

    Returns:
        (score, side_info) — score is 1.0 if flag found, 0.0 otherwise.
    """
    bench_id = example["benchmark_id"]
    c_hash = _candidate_hash(candidate)
    iteration = _get_iteration(c_hash)
    run_id = bench_id

    # experiments/experimentN/iteration_001/XBEN-xxx-24/
    base_dir = EXPERIMENT_DIR or Path("experiments")
    runs_dir = base_dir / f"iteration_{iteration:03d}"
    run_dir = runs_dir / run_id

    cached = cache.try_load(c_hash, bench_id, AGENT_MODEL, AGENT_MAX_ITER, run_dir)
    if cached is not None:
        return cached

    port = start_benchmark(bench_id)
    try:
        metadata = run_agent(
            target=f"http://localhost:{port}",
            run_id=run_id,
            seed=_make_seed(candidate),
            max_iter=AGENT_MAX_ITER,
            runs_dir=runs_dir,
            model=AGENT_MODEL,
        )

        score = 1.0 if metadata["success"] else 0.0

        # Feed the full context window to GEPA's reflection LLM
        context_path = run_dir / "context_window.json"
        if context_path.exists():
            context_window = context_path.read_text(encoding="utf-8")
            oa.log(f"Benchmark: {bench_id} | Success: {metadata['success']}\n\nContext window:\n{context_window}")

        side_info = {
            "benchmark_id": bench_id,
            "level": example["level"],
            "tags": example["tags"],
            "success": metadata["success"],
            "stop_reason": metadata["stop_reason"],
            "iterations": metadata["iterations_used"],
            "cost_usd": metadata["total_cost_usd"],
        }

        cache.try_save(c_hash, bench_id, AGENT_MODEL, AGENT_MAX_ITER, score, side_info, run_dir)
        return score, side_info

    finally:
        stop_benchmark(bench_id)


def _make_seed(candidate: dict[str, str]) -> SimpleNamespace:
    """Convert GEPA candidate dict into a seed namespace for the runner."""
    return SimpleNamespace(
        SYSTEM_PROMPT=candidate["system_prompt"],
        DEFAULT_TASK=candidate["default_task"],
        TOOL_SCHEMAS=TOOL_SCHEMAS,
        TOOLS=TOOLS,
    )


def _candidate_hash(candidate: dict[str, str]) -> str:
    """SHA-256 hash of candidate for unique run IDs."""
    raw = json.dumps(candidate, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()
