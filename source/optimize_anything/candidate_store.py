"""Content-addressable store: maps candidate files_hash → agent file snapshot.

Each entry lives at STORE_DIR/{hash}/ with the actual agent files inside.
"""

import shutil
from pathlib import Path

STORE_DIR: Path | None = None


def configure(store_dir: Path) -> None:
    global STORE_DIR
    STORE_DIR = store_dir
    store_dir.mkdir(parents=True, exist_ok=True)


def store(files_hash: str, files: dict) -> None:
    """Persist a files snapshot keyed by hash. No-op if already stored."""
    if STORE_DIR is None or not files_hash:
        return
    entry = STORE_DIR / files_hash
    if entry.exists():
        return
    entry.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        dest = entry / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def load(files_hash: str) -> dict | None:
    """Return stored files dict for the given hash, or None if not found."""
    if STORE_DIR is None or not files_hash:
        return None
    entry = STORE_DIR / files_hash
    if not entry.exists():
        return None
    result = {}
    for f in sorted(entry.rglob("*")):
        if f.is_file():
            try:
                result[str(f.relative_to(entry))] = f.read_text(encoding="utf-8")
            except Exception:
                pass
    return result or None


def restore_workspace(files_hash: str, agent_dir: Path) -> bool:
    """Overwrite agent_dir with stored files for files_hash. Returns True on success."""
    files = load(files_hash)
    if files is None:
        return False
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
    agent_dir.mkdir(parents=True)
    for rel, content in files.items():
        dest = agent_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    # Ensure skills/ dir exists even when snapshot has no skill files yet
    (agent_dir / "skills").mkdir(exist_ok=True)
    return True
