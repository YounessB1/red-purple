"""GEPA evaluator — scores a candidate seed against a single benchmark."""

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
from source.optimize_anything.diagnoser import diagnose
from source.optimize_anything.utils import candidate_hash

# Set by core_loop before optimization starts
EXPERIMENT_DIR: Path | None = None
AGENT_MAX_ITER: int = 50
AGENT_MODEL: str = ""
JUDGE_MODEL: str = ""
DIAGNOSER_MODEL: str = ""
REFLECTOR_MODEL: str = ""
TRAIN_SIZE: int = 0
GT: bool = False
AGENT_SERVER_URL: str = "http://localhost:8000"
LOGGER = None

# Iteration tracking — driven by GEPA engine callbacks, not candidate hashes
_gepa_iteration: int = 0
_gepa_iteration_lock = threading.Lock()

# Role tracking — "parent" or "child" within a train iteration, "val" otherwise
_gepa_role: str = "val"
_gepa_role_lock = threading.Lock()

# Current candidate — set during evaluate() so the reflector can access it
_current_candidate: dict = {}
_current_candidate_lock = threading.Lock()


def configure_runtime(
    *,
    experiment_dir: Path,
    agent_max_iter: int,
    agent_model: str,
    judge_model: str,
    diagnoser_model: str = "",
    reflector_model: str = "",
    train_size: int = 0,
    gt: bool,
    logger=None,
) -> None:
    global EXPERIMENT_DIR, AGENT_MAX_ITER, AGENT_MODEL, JUDGE_MODEL, DIAGNOSER_MODEL, REFLECTOR_MODEL, TRAIN_SIZE, GT, LOGGER
    global _gepa_iteration

    EXPERIMENT_DIR = experiment_dir
    AGENT_MAX_ITER = agent_max_iter
    AGENT_MODEL = agent_model
    JUDGE_MODEL = judge_model
    DIAGNOSER_MODEL = diagnoser_model
    REFLECTOR_MODEL = reflector_model
    TRAIN_SIZE = train_size
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


def set_gepa_role(role: str) -> None:
    global _gepa_role
    with _gepa_role_lock:
        _gepa_role = role


def _get_role() -> str:
    with _gepa_role_lock:
        return _gepa_role


def set_current_candidate(candidate: dict) -> None:
    global _current_candidate
    with _current_candidate_lock:
        _current_candidate = candidate


def get_current_candidate() -> dict:
    with _current_candidate_lock:
        return _current_candidate


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
    set_current_candidate(candidate)
    bench_id = example["benchmark_id"]
    c_hash = candidate_hash(candidate)
    iteration = _get_iteration()

    split = example.get("split", "unknown")
    base_dir = EXPERIMENT_DIR or Path("experiments")
    role = _get_role()
    subpath = f"{split}/{role}" if split == "train" else split
    runs_dir = base_dir / f"iteration_{iteration:03d}" / subpath
    run_dir = runs_dir / bench_id

    cached_judge = None
    cached = cache.try_load(c_hash, bench_id, AGENT_MODEL, AGENT_MAX_ITER, run_dir)
    if cached is not None:
        metadata, context_window, diagnosis, cached_judge = cached
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
            if DIAGNOSER_MODEL and not metadata["success"] and _get_role() == "parent":
                diagnosis = diagnose(context_window, metadata, DIAGNOSER_MODEL, LOGGER,
                                    reflector_model=REFLECTOR_MODEL, train_size=TRAIN_SIZE)
                (run_dir / "diagnosis.json").write_text(
                    json.dumps({"diagnosis": diagnosis}, indent=2), encoding="utf-8"
                )
            else:
                diagnosis = ""
            cache.try_save(c_hash, bench_id, AGENT_MODEL, AGENT_MAX_ITER, run_dir)
        finally:
            print(f"[eval] {bench_id} — stopping benchmark")
            stop_benchmark(bench_id)

    LOGGER.log_agents(metadata)

    if metadata["success"]:
        score = 1.0
    elif JUDGE_MODEL:
        if cached_judge and cached_judge.get("model") == JUDGE_MODEL:
            score = float(cached_judge["score"])
        else:
            score, reason = llm_judge(context_window, bench_id, model=JUDGE_MODEL, logger=LOGGER, gt=GT)
            (run_dir / "judge_score.json").write_text(
                json.dumps({"model": JUDGE_MODEL, "score": score, "reason": reason}),
                encoding="utf-8",
            )
            cache.try_save(c_hash, bench_id, AGENT_MODEL, AGENT_MAX_ITER, run_dir)
    else:
        score = 0.0
    print(f"[eval] {bench_id} — score {score:.3f} - {metadata['stop_reason']}")
    side_info = {
        "benchmark_id": bench_id,
        "success": metadata["success"],
        "stop_reason": metadata["stop_reason"],
        "iterations": metadata["iterations_used"],
        "context_window": context_window,
        "diagnosis": diagnosis,
    }
    return score, side_info


_WORKSPACE_AGENT = Path(__file__).resolve().parents[2] / "workspace" / "agent"
_SEED_DIR = Path(__file__).resolve().parents[2] / "source" / "seed"


def _resolve_candidate(candidate: dict) -> dict:
    """Replace hash-string files ref with the stored files dict for that hash."""
    from source.optimize_anything import candidate_store
    files_ref = candidate.get("files")
    if isinstance(files_ref, dict):
        return candidate  # already resolved
    if isinstance(files_ref, str):
        stored = candidate_store.load(files_ref)
        if stored is not None:
            return {**candidate, "files": stored}
    # Fallback: read directly from workspace/agent/ (seed before store is populated)
    src = _WORKSPACE_AGENT if (
        _WORKSPACE_AGENT.exists()
        and any(f for f in _WORKSPACE_AGENT.rglob("*") if f.is_file() and f.name != ".gitkeep")
    ) else _SEED_DIR
    files = {}
    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.name == ".gitkeep":
            continue
        try:
            files[str(f.relative_to(src))] = f.read_text(encoding="utf-8")
        except Exception:
            pass
    return {**candidate, "files": files}


def _run_via_server(target: str, candidate: dict, max_iter: int, model: str) -> dict:
    params = urllib.parse.urlencode({
        "target": target,
        "max_iter": max_iter,
        "seed_json": json.dumps(_resolve_candidate(candidate)),
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


