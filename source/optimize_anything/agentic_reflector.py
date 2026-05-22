"""Agentic prompt reflector — spawns an OpenCode agent that edits agent files in workspace/."""

import json
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

from source.optimize_anything import candidate_store
from source.optimize_anything.logger import Logger
from source.optimize_anything.evaluator import _get_iteration, get_current_candidate
from source.optimize_anything.utils import candidate_hash

_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _ROOT / "workspace"
_WORKSPACE_AGENT = _WORKSPACE / "agent"
_REFLECTOR_MD = _ROOT / ".opencode" / "agents" / "reflector.md"


def _snapshot_dir(d: Path, exclude: frozenset[str] = frozenset()) -> dict:
    result = {}
    for f in sorted(d.rglob("*")):
        if not f.is_file():
            continue
        rel = str(f.relative_to(d))
        if any(rel == e or rel.startswith(e + "/") for e in exclude):
            continue
        try:
            result[rel] = f.read_text(encoding="utf-8")
        except Exception:
            pass
    return result


def _materialize_dir(d: Path, files: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        dest = d / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def _update_reflector_model(model: str) -> None:
    """Patch the model field in root's reflector.md in-place."""
    content = _REFLECTOR_MD.read_text(encoding="utf-8")
    new_content = re.sub(r'(?m)^model:.*$', f'model: "{model}"', content)
    if new_content != content:
        _REFLECTOR_MD.write_text(new_content, encoding="utf-8")


def _copy_artifacts(iter_dir: Path) -> None:
    artifacts_dst = _WORKSPACE / "artifacts"
    shutil.rmtree(artifacts_dst, ignore_errors=True)
    artifacts_src = iter_dir / "train" / "parent"
    if artifacts_src.exists():
        shutil.copytree(artifacts_src, artifacts_dst)
    else:
        artifacts_dst.mkdir(parents=True)


# ── Agentic reflector ──────────────────────────────────────────────────

class AgenticReflector:
    """Spawns an OpenCode agent that reads artifacts and edits agent files in workspace/ in-place."""

    def __init__(self, model: str, logger: Logger, experiment_dir: Path) -> None:
        self._model = model
        self._logger = logger
        self._experiment_dir = experiment_dir

    def __call__(self, _prompt: str | list[dict]) -> str:
        iteration = _get_iteration()
        iter_dir = self._experiment_dir / f"iteration_{iteration:03d}"

        # Patch model in root reflector.md; reflector runs from root so no workspace copy needed
        _update_reflector_model(self._model)

        # Restore workspace/agent/ to the exact files for the candidate GEPA selected
        current_hash = get_current_candidate().get("files", "")
        if not candidate_store.restore_workspace(current_hash, _WORKSPACE_AGENT):
            print(f"[agentic-reflector] Warning: hash {current_hash[:12]}… not in store, using current workspace", flush=True)

        # Copy training artifacts into workspace/artifacts/ for the reflector to read
        _copy_artifacts(iter_dir)

        # Snapshot workspace/agent/ before reflection → iter_dir/agent/
        _materialize_dir(iter_dir / "parent", _snapshot_dir(_WORKSPACE_AGENT))

        message = (
            "Analyze the iteration artifacts and improve the CTF agent strategy.\n\n"
            "- Artifacts are in workspace/artifacts/ — read metadata.json and diagnosis.json for each benchmark\n"
            "- Current agent strategy is in workspace/agent/prompt.md — this is what you must improve\n"
            "- Edit workspace/agent/prompt.md in-place with a better strategy based on the failure diagnoses\n"
            "- You may create or update skill files in workspace/agent/skills/ for reusable attack patterns\n"
            "- Keep skills minimal: prefer updating an existing skill over creating a new one; avoid bloat\n"
            "- Do NOT modify workspace/artifacts/, workspace/agent/.opencode/, or .opencode/agents/reflector.md"
        )

        print(f"\n[agentic-reflector] Starting OpenCode for iteration {iteration}…", flush=True)

        proc = subprocess.run(
            [
                "opencode", "run",
                "--agent", "reflector",
                "--dir", str(_ROOT),
                message,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        raw = proc.stdout.strip()
        if not raw:
            print(f"[agentic-reflector] No stdout — stderr:\n{proc.stderr}", flush=True)

        # Snapshot workspace/agent/ after reflection → iter_dir/child/ + store
        new_files = _snapshot_dir(_WORKSPACE_AGENT)
        new_hash = candidate_hash(new_files)
        candidate_store.store(new_hash, new_files)
        _materialize_dir(iter_dir / "child", new_files)

        changes_path = _WORKSPACE / "reflector_changes.md"
        changes = changes_path.read_text(encoding="utf-8").strip() if changes_path.exists() else ""
        if changes:
            self._logger.log_reflector_changes(changes)

        input_tokens, output_tokens, cost, steps = self._trace_session()
        self._logger.log_reflector(input_tokens, output_tokens, message, raw, cost=cost, steps=steps)

        return f"```\n{new_hash}\n```"

    def _trace_session(self) -> tuple[int, int, float, list]:
        """Query OpenCode's SQLite DB for the most recent reflector session."""
        db_path = Path.home() / ".local/share/opencode/opencode.db"
        try:
            db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except Exception:
            return 0, 0, 0.0, []
        try:
            row = db.execute(
                "SELECT id, cost, tokens_input, tokens_output, tokens_reasoning, tokens_cache_read "
                "FROM session WHERE directory=? ORDER BY time_created DESC LIMIT 1",
                (str(_ROOT),),
            ).fetchone()
            if not row:
                return 0, 0, 0.0, []
            session_id, cost, input_tokens, output_tokens, *_ = row
            parts = db.execute(
                "SELECT data FROM part WHERE session_id=? ORDER BY time_created",
                (session_id,),
            ).fetchall()
            steps: list[dict] = []
            for (raw_part,) in parts:
                try:
                    part = json.loads(raw_part)
                except Exception:
                    continue
                ptype = part.get("type")
                if ptype == "reasoning" and part.get("text", "").strip():
                    steps.append({"type": "thinking", "text": part["text"].strip()})
                elif ptype == "text" and part.get("text", "").strip():
                    steps.append({"type": "text", "text": part["text"].strip()})
                elif ptype == "tool":
                    name = part.get("tool", "?")
                    state = part.get("state", {})
                    inp = state.get("input", {}) if isinstance(state, dict) else {}
                    step: dict = {"type": "tool", "name": name, "input": inp}
                    if name == "read":
                        step["file"] = inp.get("filePath", inp.get("path", "?"))
                    steps.append(step)
            return input_tokens, output_tokens, cost, steps
        finally:
            db.close()
