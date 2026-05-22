"""Evaluation cache — only the seed candidate's runs are cached, since mutated
candidates are unique to their run.

Key = SHA-256 of (candidate_hash, bench_id, model, max_iter). Entries store
only the raw run artifacts (metadata.json + context_window.json). The score is
always recomputed from the artifacts so changing the judge never stales the cache.
"""

import hashlib
import json
import shutil
from pathlib import Path

from source.optimize_anything.utils import candidate_hash

# Set by core_loop before optimization starts
CACHE_DIR: Path | None = None

_SEED_DIR = Path(__file__).resolve().parents[2] / "source" / "seed"


def _compute_seed_hash() -> str:
    seed_files = {}
    for f in sorted(_SEED_DIR.rglob("*")):
        if f.is_file() and f.name != ".gitkeep":
            seed_files[str(f.relative_to(_SEED_DIR))] = f.read_text(encoding="utf-8")
    return candidate_hash({"files": candidate_hash(seed_files)})


SEED_CANDIDATE_HASH = _compute_seed_hash()


def try_load(
    candidate_hash: str,
    bench_id: str,
    model: str,
    max_iter: int,
    run_dir: Path,
) -> tuple[dict, list, str, dict | None] | None:
    """Return cached (metadata, context_window, diagnosis, judge_score) on a seed-candidate cache hit, else None.

    diagnosis is "" when no diagnosis.json exists.
    judge_score is the raw judge_score.json dict, or None if not present.
    On hit: restores the cached run artifacts into `run_dir`.
    """
    if not _cacheable(candidate_hash):
        return None
    key = _make_key(candidate_hash, bench_id, model, max_iter)
    artifacts_dir = CACHE_DIR / key / bench_id
    metadata_file = artifacts_dir / "metadata.json"
    context_file = artifacts_dir / "context_window.json"
    if not metadata_file.exists() or not context_file.exists():
        return None

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    context_window = json.loads(context_file.read_text(encoding="utf-8"))
    diagnosis_file = artifacts_dir / "diagnosis.json"
    diagnosis = json.loads(diagnosis_file.read_text(encoding="utf-8"))["diagnosis"] if diagnosis_file.exists() else ""
    judge_score_file = artifacts_dir / "judge_score.json"
    judge_score = json.loads(judge_score_file.read_text(encoding="utf-8")) if judge_score_file.exists() else None
    _restore_artifacts(key, bench_id, run_dir)
    return metadata, context_window, diagnosis, judge_score


def try_save(
    candidate_hash: str,
    bench_id: str,
    model: str,
    max_iter: int,
    run_dir: Path,
) -> None:
    """Save run artifacts (metadata + context_window), but only for the seed candidate."""
    if not _cacheable(candidate_hash):
        return
    key = _make_key(candidate_hash, bench_id, model, max_iter)
    entry_dir = CACHE_DIR / key
    entry_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = entry_dir / bench_id
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


def _restore_artifacts(key: str, bench_id: str, target_dir: Path) -> None:
    source_dir = CACHE_DIR / key / bench_id
    if source_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            shutil.copy2(item, target_dir / item.name)
