"""GEPA evaluator — scores a candidate seed against a single benchmark."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

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


def configure_runtime(
    *,
    experiment_dir: Path,
    agent_max_iter: int,
    agent_model: str,
) -> None:
    global EXPERIMENT_DIR, AGENT_MAX_ITER, AGENT_MODEL
    global _seen_candidates, _iteration_counter

    EXPERIMENT_DIR = experiment_dir
    AGENT_MAX_ITER = agent_max_iter
    AGENT_MODEL = agent_model
    _seen_candidates = {}
    _iteration_counter = 0


def _get_iteration(candidate_hash: str) -> int:
    global _iteration_counter
    if candidate_hash not in _seen_candidates:
        _iteration_counter += 1
        _seen_candidates[candidate_hash] = _iteration_counter
    return _seen_candidates[candidate_hash]


def evaluate(candidate: dict[str, str], example: dict) -> tuple[float, dict]:
    """Run the agent on one benchmark and return (score, side_info).

    score is 1.0 on flag capture, 0.0 otherwise.
    """
    bench_id = example["benchmark_id"]
    c_hash = _candidate_hash(candidate)
    iteration = _get_iteration(c_hash)

    base_dir = EXPERIMENT_DIR or Path("experiments")
    runs_dir = base_dir / f"iteration_{iteration:03d}"
    run_dir = runs_dir / bench_id

    cached = cache.try_load(c_hash, bench_id, AGENT_MODEL, AGENT_MAX_ITER, run_dir)
    if cached is not None:
        score, side_info = cached
        print(f"[eval] {bench_id} — cache hit (score={score})")
        return score, side_info

    print(f"[eval] {bench_id} — starting benchmark")
    port = start_benchmark(bench_id)
    try:
        seed = SimpleNamespace(
            PROMPT=candidate["prompt"],
            TOOL_SCHEMAS=TOOL_SCHEMAS,
            TOOLS=TOOLS,
        )
        metadata, context_window = run_agent(
            target=f"http://localhost:{port}",
            run_id=bench_id,
            seed=seed,
            max_iter=AGENT_MAX_ITER,
            model=AGENT_MODEL,
            runs_dir=run_dir.parent,
        )
        
        # artifacts = _run_via_server(f"http://localhost:{port}", candidate, AGENT_MAX_ITER, AGENT_MODEL); 
        # metadata, context_window = artifacts["metadata"], artifacts["context_window"]

        score = 1.0 if metadata["success"] else 0.0
        side_info = {
            "benchmark_id": bench_id,
            "success": metadata["success"],
            "stop_reason": metadata["stop_reason"],
            "iterations": metadata["iterations_used"],
            "context_window": context_window,
        }

        cache.try_save(c_hash, bench_id, AGENT_MODEL, AGENT_MAX_ITER, score, side_info, run_dir)
        return score, side_info

    finally:
        print(f"[eval] {bench_id} — stopping benchmark")
        stop_benchmark(bench_id)


def _candidate_hash(candidate: dict[str, str]) -> str:
    raw = json.dumps(candidate, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()
