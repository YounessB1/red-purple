"""Simple local JSONL tracer — writes one event per line to runs/<run-id>/events.jsonl."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


class Tracer:
    def __init__(self, target: str) -> None:
        self.run_id = f"run-{uuid4().hex[:8]}"
        self.target = target

        runs_dir = Path("/app/runs")
        runs_dir.mkdir(parents=True, exist_ok=True)

        self._run_dir = runs_dir / self.run_id
        self._run_dir.mkdir()
        self._events_path = self._run_dir / "events.jsonl"

        self._emit("run.started", {"target": target, "run_id": self.run_id})

    # ------------------------------------------------------------------
    def log_message(self, role: str, content: str) -> None:
        self._emit("chat.message", {"role": role, "content": content})

    def log_tool_start(self, tool_name: str, args: dict) -> None:
        self._emit("tool.execution.started", {"tool": tool_name, "args": args})

    def log_tool_end(self, tool_name: str, result: str) -> None:
        self._emit("tool.execution.completed", {"tool": tool_name, "result": result})

    def finish(self) -> None:
        self._emit("run.completed", {"run_id": self.run_id})
        print(f"[tracer] run saved → {self._run_dir}")

    # ------------------------------------------------------------------
    def _emit(self, event_type: str, payload: dict) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "run_id": self.run_id,
            "payload": payload,
        }
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
