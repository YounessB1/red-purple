"""Evaluation cache — only the seed candidate's runs are cached, since mutated
candidates are unique to their run.

Key = SHA-256 of (candidate_hash, bench_id, model, max_iter). Entries store
(score, side_info) plus the run artifacts (metadata.json + context_window.json)
so replays look identical to a fresh run. Shared across experiments.
"""

import hashlib
import json
import shutil
from pathlib import Path

import gepa.optimize_anything as oa

from source.agent.seed import DEFAULT_TASK, SYSTEM_PROMPT

# Set by core_loop before optimization starts
CACHE_DIR: Path | None = None

# Only seed-candidate evals are cached.
_SEED_CANDIDATE = {"system_prompt": SYSTEM_PROMPT, "default_task": DEFAULT_TASK}
SEED_CANDIDATE_HASH = hashlib.sha256(
    json.dumps(_SEED_CANDIDATE, sort_keys=True).encode()
).hexdigest()


def try_load(
    candidate_hash: str,
    bench_id: str,
    model: str,
    max_iter: int,
    run_dir: Path,
) -> tuple[float, dict] | None:
    """Return cached (score, side_info) on a seed-candidate cache hit, else None.

    On hit: restores the cached run artifacts into `run_dir` and re-emits the
    context window to GEPA's reflection log, so the cached eval is
    indistinguishable from a fresh one from GEPA's perspective.
    """
    if not _cacheable(candidate_hash):
        return None
    key = _make_key(candidate_hash, bench_id, model, max_iter)
    result_file = CACHE_DIR / key / "result.json"
    if not result_file.exists():
        return None

    data = json.loads(result_file.read_text(encoding="utf-8"))
    score, side_info = data["score"], data["side_info"]

    _restore_artifacts(key, run_dir)
    _log_context_window(run_dir, bench_id, side_info["success"])
    print(f"[eval] {bench_id} — cache hit (score={score})")
    return score, side_info


def try_save(
    candidate_hash: str,
    bench_id: str,
    model: str,
    max_iter: int,
    score: float,
    side_info: dict,
    run_dir: Path,
) -> None:
    """Save (score, side_info) + run artifacts, but only for the seed candidate."""
    if not _cacheable(candidate_hash):
        return
    key = _make_key(candidate_hash, bench_id, model, max_iter)
    entry_dir = CACHE_DIR / key
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / "result.json").write_text(
        json.dumps({"score": score, "side_info": side_info}, indent=2),
        encoding="utf-8",
    )
    artifacts_dir = entry_dir / "run"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    if run_dir.exists():
        shutil.copytree(run_dir, artifacts_dir)


# ── Internals ──────────────────────────────────────────────────────────

def _cacheable(candidate_hash: str) -> bool:
    return CACHE_DIR is not None and candidate_hash == SEED_CANDIDATE_HASH


def _make_key(candidate_hash: str, bench_id: str, model: str, max_iter: int) -> str:
    raw = f"{candidate_hash}:{bench_id}:{model}:{max_iter}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _restore_artifacts(key: str, target_dir: Path) -> None:
    source_dir = CACHE_DIR / key / "run"
    if source_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            shutil.copy2(item, target_dir / item.name)


def _log_context_window(run_dir: Path, bench_id: str, success: bool) -> None:
    context_path = run_dir / "context_window.json"
    if context_path.exists():
        ctx = context_path.read_text(encoding="utf-8")
        oa.log(f"Benchmark: {bench_id} | Success: {success}\n\nContext window:\n{ctx}")
