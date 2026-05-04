"""GEPA evaluator — scores a candidate seed against a single benchmark."""

import hashlib
import json
import threading
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

from source.agent.runner import run as run_agent
from source.benchmark import start_benchmark, stop_benchmark
from source.optimize_anything import cache
from source.optimize_anything.LLM_as_judge import llm_judge

# Set by core_loop before optimization starts
EXPERIMENT_DIR: Path | None = None
AGENT_MAX_ITER: int = 50
AGENT_MODEL: str = ""
JUDGE_MODEL: str = ""
GT: bool = False
AGENT_SERVER_URL: str = "http://localhost:8000"
LOGGER = None

# Iteration tracking — driven by GEPA engine callbacks, not candidate hashes
_gepa_iteration: int = 0
_gepa_iteration_lock = threading.Lock()


def configure_runtime(
    *,
    experiment_dir: Path,
    agent_max_iter: int,
    agent_model: str,
    judge_model: str,
    gt: bool,
    logger=None,
) -> None:
    global EXPERIMENT_DIR, AGENT_MAX_ITER, AGENT_MODEL, JUDGE_MODEL, GT, LOGGER
    global _gepa_iteration

    EXPERIMENT_DIR = experiment_dir
    AGENT_MAX_ITER = agent_max_iter
    AGENT_MODEL = agent_model
    JUDGE_MODEL = judge_model
    GT = gt
    LOGGER = logger
    _gepa_iteration = 0


def set_gepa_iteration(n: int) -> None:
    global _gepa_iteration
    with _gepa_iteration_lock:
        _gepa_iteration = n


def _get_iteration() -> int:
    with _gepa_iteration_lock:
        return _gepa_iteration


def save_run(run_dir: Path, metadata: dict, context_window: list) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "context_window.json").write_text(
        json.dumps(context_window, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def evaluate(candidate: dict[str, str], example: dict) -> tuple[float, dict]:
    """Run the agent on one benchmark and return (score, side_info).

    score is 1.0 on flag capture, 0.0 otherwise.
    """
    bench_id = example["benchmark_id"]
    c_hash = _candidate_hash(candidate)
    iteration = _get_iteration()

    split = example.get("split", "unknown")
    base_dir = EXPERIMENT_DIR or Path("experiments")
    runs_dir = base_dir / f"iteration_{iteration:03d}" / split
    run_dir = runs_dir / bench_id

    cached = cache.try_load(c_hash, bench_id, AGENT_MODEL, AGENT_MAX_ITER, run_dir)
    if cached is not None:
        metadata, context_window = cached
        print(f"[eval] {bench_id} — cache hit")
    else:
        print(f"[eval] {bench_id} — starting benchmark")
        port = start_benchmark(bench_id)
        try:
            try:
                artifacts = _run_via_server(f"http://localhost:{port}", candidate, AGENT_MAX_ITER, AGENT_MODEL)
            except Exception:
                import traceback; traceback.print_exc()
                raise
            metadata, context_window = artifacts["metadata"], artifacts["context_window"]
            save_run(run_dir, metadata, context_window)
            cache.try_save(c_hash, bench_id, AGENT_MODEL, AGENT_MAX_ITER, run_dir)
        finally:
            print(f"[eval] {bench_id} — stopping benchmark")
            stop_benchmark(bench_id)

    LOGGER.log_agents(metadata)

    if metadata["success"]:
        score = 1.0
    elif JUDGE_MODEL:
        score = llm_judge(context_window, bench_id, model=JUDGE_MODEL, logger=LOGGER, gt=GT)
    else:
        score = 0.0
    print(f"[eval] {bench_id} — score {score:.3f} - {metadata['stop_reason']}")
    side_info = {
        "benchmark_id": bench_id,
        "success": metadata["success"],
        "stop_reason": metadata["stop_reason"],
        "iterations": metadata["iterations_used"],
        "context_window": context_window,
    }
    return score, side_info


def _run_via_server(target: str, candidate: dict, max_iter: int, model: str) -> dict:
    params = urllib.parse.urlencode({
        "target": target,
        "max_iter": max_iter,
        "seed_json": json.dumps(candidate),
        "model": model,
    })
    req = urllib.request.Request(
        f"{AGENT_SERVER_URL}/run?{params}",
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Server error {e.code} for {target}:\n{body}") from e
    except Exception as e:
        import traceback; traceback.print_exc()
        raise


def _candidate_hash(candidate: dict[str, str]) -> str:
    raw = json.dumps(candidate, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()
