"""Agentic prompt reflector — spawns an OpenCode agent to browse iteration artifacts."""

import json
import re
import sqlite3
import subprocess
from pathlib import Path

from source.optimize_anything.logger import Logger
from source.optimize_anything.utils import extract_new_prompt
from source.optimize_anything.evaluator import _get_iteration

# ── Agentic reflector ──────────────────────────────────────────────────

class AgenticReflector:
    """Spawns an OpenCode agent that browses iteration artifacts and proposes an improved prompt."""

    _AGENT_MD = Path(__file__).parents[2] / ".opencode" / "agents" / "reflector.md"

    def __init__(self, model: str, logger: Logger, experiment_dir: Path) -> None:
        self._logger = logger
        self._experiment_dir = experiment_dir
        self._set_model(model)

    def _set_model(self, model: str) -> None:
        content = self._AGENT_MD.read_text(encoding="utf-8")
        updated = re.sub(r'(?m)^model:.*$', f'model: "{model}"', content)
        self._AGENT_MD.write_text(updated, encoding="utf-8")

    # for gepa signature prompt is passed to __call__, but we ignore it since the reflector reads the current prompt directly from the iteration artifacts
    def __call__(self, prompt: str | list[dict]) -> str:
        iteration = _get_iteration()
        iter_dir = self._experiment_dir / f"iteration_{iteration:03d}"

        current_prompt = self._extract_current_prompt(prompt)
        message = (
            "Analyze the iteration artifacts and return an improved strategy prompt as JSON.\n\n"
            f"## Current strategy prompt (what you must improve)\n\n```\n{current_prompt}\n```"
        )
        print(f"\n[agentic-reflector] Starting OpenCode for iteration {iteration}…", flush=True)

        proc = subprocess.run(
            [
                "opencode", "run",
                "--agent", "reflector",
                "--dir", str(iter_dir),
                message,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        raw = proc.stdout.strip()
        if not raw:
            print(f"[agentic-reflector] No output — stderr:\n{proc.stderr}", flush=True)

        input_tokens, output_tokens, cost, steps = self._trace_session(iter_dir)
        self._logger.log_reflector(input_tokens, output_tokens, message, raw, cost=cost, steps=steps)
        return extract_new_prompt(raw, self._logger)

    def _trace_session(self, iter_dir: Path) -> tuple[int, int, float, list]:
        """Query OpenCode's SQLite DB for the session trace; return (input_tokens, output_tokens, cost, steps)."""
        db_path = Path.home() / ".local/share/opencode/opencode.db"
        try:
            db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except Exception:
            return 0, 0, 0.0, []

        try:
            # iter_dir is unique per iteration so this always matches exactly one session.
            # ORDER BY time_created DESC LIMIT 1 handles the rare case of a re-run on the same dir.
            row = db.execute(
                "SELECT id, cost, tokens_input, tokens_output, tokens_reasoning, tokens_cache_read "
                "FROM session WHERE directory=? ORDER BY time_created DESC LIMIT 1",
                (str(iter_dir),),
            ).fetchone()
            if not row:
                return 0, 0, 0.0, []

            session_id, cost, input_tokens, output_tokens, reasoning_tokens, cache_read = row

            trace: dict = {
                "session_id": session_id,
                "cost_usd": round(cost, 6),
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "reasoning": reasoning_tokens,
                    "cache_read": cache_read,
                },
                "steps": [],
            }

            parts = db.execute(
                "SELECT data FROM part WHERE session_id=? ORDER BY time_created",
                (session_id,),
            ).fetchall()

            for (raw_part,) in parts:
                try:
                    part = json.loads(raw_part)
                except Exception:
                    continue

                ptype = part.get("type")

                if ptype == "reasoning" and part.get("text", "").strip():
                    trace["steps"].append({"type": "thinking", "text": part["text"].strip()})

                elif ptype == "text" and part.get("text", "").strip():
                    trace["steps"].append({"type": "text", "text": part["text"].strip()})

                elif ptype == "tool":
                    name = part.get("tool", "?")
                    state = part.get("state", {})
                    inp = state.get("input", {}) if isinstance(state, dict) else {}
                    step: dict = {"type": "tool", "name": name, "input": inp}
                    if name == "read":
                        step["file"] = inp.get("filePath", inp.get("path", "?"))
                    trace["steps"].append(step)

            # Skipped part types (fall through silently):
            #   step-start   — turn delimiter, no data
            #   step-finish  — per-turn cost/tokens, redundant with session.cost/tokens_*
            #   tool-result  — raw file content returned to the model, large and not useful for the trace

            return input_tokens, output_tokens, cost, trace["steps"]

        finally:
            db.close()

    def _extract_current_prompt(self, prompt: str | list[dict]) -> str:
        text = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        m = re.search(r"## Current strategy instructions.*?```\n(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else "(could not extract current prompt)"
