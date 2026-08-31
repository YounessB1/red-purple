"""Shared utility — query OpenCode's SQLite DB for session metrics."""

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_BIN = _REPO_ROOT / "node_modules" / ".bin" / "opencode"

# Prefer the pinned local binary (installed via `npm install`) over whatever
# is on PATH, so the project always runs against the declared opencode version.
OPENCODE_BIN: str = str(_LOCAL_BIN) if _LOCAL_BIN.exists() else (shutil.which("opencode") or "opencode")

# scorer/diagnoser run in scratch workdirs (tmp/scorer_*, tmp/diagnoser_*) that only
# ever get context_window.json/metadata.json/ground_truth.md written into them.
# Provider config (baseURL/apiKey/model registry) for on-prem models like polito/*
# lives only in source/seed/opencode.json, which otherwise only reaches the CTF
# agent's own materialized workdir (via runner.py). Without it here, any polito/*
# model configured as scorer/diagnoser fails every call with a generic server
# error and silently falls back to a fake score/empty diagnosis. Copying it in
# is harmless when the configured model doesn't need it (openrouter/* etc.).
_SEED_OPENCODE_JSON = _REPO_ROOT / "source" / "seed" / "opencode.json"


def trace_opencode_session(workdir: Path) -> tuple[int, int, float, list]:
    """Return (input_tokens, output_tokens, cost, steps) for the most recent session in workdir."""
    db_path = Path.home() / ".local/share/opencode/opencode.db"
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except Exception:
        return 0, 0, 0.0, []
    try:
        row = db.execute(
            "SELECT id, cost, tokens_input, tokens_output FROM session "
            "WHERE directory=? ORDER BY time_created DESC LIMIT 1",
            (str(workdir),),
        ).fetchone()
        if not row:
            return 0, 0, 0.0, []
        session_id, cost, input_tokens, output_tokens = row
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
                state = part.get("state", {}) if isinstance(part.get("state"), dict) else {}
                # Skip phantom abort entries (same callID re-emitted as unknown+interrupted
                # after a parallel tool completes — OpenCode parallel-execution artifact).
                if (
                    name == "unknown"
                    and state.get("status") == "error"
                    and (state.get("metadata") or {}).get("interrupted")
                ):
                    continue
                inp = state.get("input", {})
                steps.append({"type": "tool", "name": name, "input": inp})
        return input_tokens, output_tokens, cost or 0.0, steps
    finally:
        db.close()


def run_opencode_agent(
    agent: str,
    model: str,
    workdir: Path,
    prompt: str,
    timeout: int = 300,
    label: str = "",
) -> tuple[bool, int, int, float, list]:
    """Run one OpenCode agent call and return its trace.

    Returns (timed_out, input_tokens, output_tokens, cost, steps).
    timed_out=True means the subprocess hit the timeout; caller handles retry.
    """
    lbl = label or agent
    if _SEED_OPENCODE_JSON.exists():
        shutil.copy(_SEED_OPENCODE_JSON, workdir / "opencode.json")
    try:
        proc = subprocess.run(
            [OPENCODE_BIN, "run", "--agent", agent, "--model", model, "--dir", str(workdir), prompt],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0 and proc.stderr:
            print(f"[{lbl}] opencode stderr: {proc.stderr.strip()[:200]}")
    except subprocess.TimeoutExpired:
        return True, 0, 0, 0.0, []
    input_tokens, output_tokens, cost, steps = trace_opencode_session(workdir)
    return False, input_tokens, output_tokens, cost, steps


def last_agent_text(steps: list) -> str:
    """Return the last text-type part from a trace_opencode_session steps list."""
    for step in reversed(steps):
        if step.get("type") == "text" and step.get("text", "").strip():
            return step["text"].strip()
    return ""
