"""TracingCallback — persists GEPA engine events to disk for post-hoc inspection."""

import json
import threading
from pathlib import Path


class TracingCallback:
    """Writes one JSON file per reflection event to log_dir/."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self._lock = threading.Lock()

    def on_reflective_dataset_built(self, event) -> None:
        self._write(f"iter_{event['iteration']:03d}_reflective_dataset.json", {
            "candidate_idx": event.get("candidate_idx"),
            "components": event.get("components"),
            "dataset": event.get("dataset"),
        })

    def on_proposal_end(self, event) -> None:
        self._write(f"iter_{event['iteration']:03d}_proposal.json", {
            "new_instructions": event.get("new_instructions"),
        })

    def on_candidate_accepted(self, event) -> None:
        self._write(f"iter_{event['iteration']:03d}_accepted.json", {
            "new_candidate_idx": event.get("new_candidate_idx"),
            "new_score": event.get("new_score"),
            "parent_ids": list(event.get("parent_ids", [])),
        })

    def on_candidate_rejected(self, event) -> None:
        self._write(f"iter_{event['iteration']:03d}_rejected.json", {
            "old_score": event.get("old_score"),
            "new_score": event.get("new_score"),
            "reason": event.get("reason"),
        })

    def _write(self, filename: str, data: dict) -> None:
        with self._lock:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / filename).write_text(
                json.dumps(data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
