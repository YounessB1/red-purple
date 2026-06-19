"""Shared utility — query OpenCode's SQLite DB for session metrics."""

import json
import sqlite3
from pathlib import Path


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
                state = part.get("state", {})
                inp = state.get("input", {}) if isinstance(state, dict) else {}
                steps.append({"type": "tool", "name": name, "input": inp})
        return input_tokens, output_tokens, cost or 0.0, steps
    finally:
        db.close()


def last_agent_text(steps: list) -> str:
    """Return the last text-type part from a trace_opencode_session steps list."""
    for step in reversed(steps):
        if step.get("type") == "text" and step.get("text", "").strip():
            return step["text"].strip()
    return ""
