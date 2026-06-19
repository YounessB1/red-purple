"""Shared utilities for the optimize_anything pipeline."""

import hashlib
import json
import re
from pathlib import Path


def candidate_hash(candidate: dict) -> str:
    return hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()


def folder_to_dict(d: Path, exclude: frozenset[str] = frozenset()) -> dict:
    """Read all text files under d into a {relative_path: content} dict, skipping .gitkeep."""
    result = {}
    for f in sorted(d.rglob("*")):
        if not f.is_file() or f.name == ".gitkeep":
            continue
        rel = str(f.relative_to(d))
        if any(rel == e or rel.startswith(e + "/") for e in exclude):
            continue
        try:
            result[rel] = f.read_text(encoding="utf-8")
        except Exception:
            pass
    return result


def dict_to_folder(d: Path, files: dict) -> None:
    """Write a {relative_path: content} dict to disk under d."""
    d.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        dest = d / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def get_candidates_pool(experiment_dir: Path, iteration: int) -> list[dict]:
    """Read pool.json for the given iteration. Returns the candidates list."""
    pool_path = experiment_dir / f"iteration_{iteration:03d}" / "pool.json"
    if not pool_path.exists():
        return []
    try:
        data = json.loads(pool_path.read_text(encoding="utf-8"))
        return data.get("candidates", [])
    except Exception:
        return []


def log(msg: str) -> None:
    """Print to stdout, silently ignoring BrokenPipeError on restart."""
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        pass


def next_experiment_dir(base: Path) -> Path:
    """Find the next experiment directory: experiment1, experiment2, ..."""
    base.mkdir(parents=True, exist_ok=True)
    existing = [
        int(m.group(1))
        for d in base.iterdir()
        if d.is_dir() and (m := re.match(r"experiment(\d+)$", d.name))
    ]
    n = max(existing, default=0) + 1
    return base / f"experiment{n}"
