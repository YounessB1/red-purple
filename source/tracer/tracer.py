"""Red-Purple Tracer — records every run and writes two output files:
    experiments/<run_id>/metadata.json      — stats, tokens, cost, success, flag
    experiments/<run_id>/context_window.json — full conversation history
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_FLAG_RE = re.compile(r'[Ff][Ll][Aa][Gg]\{[^}]+\}')

_PRICES_PATH = Path(__file__).parent / "model_prices.json"


def _load_prices() -> dict:
    try:
        return json.loads(_PRICES_PATH.read_text(encoding="utf-8"))["models"]
    except Exception:
        return {}


def _compute_cost(model: str, input_tokens: int, output_tokens: int, prices: dict) -> float:
    entry = prices.get(model, {})
    return (
        input_tokens  * entry.get("input_per_token",  0.0)
        + output_tokens * entry.get("output_per_token", 0.0)
    )


def _detect_flag(history: list[dict]) -> str | None:
    """Scan assistant messages (newest first) for FLAG{...} pattern."""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            match = _FLAG_RE.search(msg.get("content", ""))
            if match:
                return match.group(0)
    return None


class Tracer:
    def __init__(
        self,
        run_id: str,
        target: str,
        task: str,
        model: str,
        runs_dir: str | Path,
        max_iterations: int = 100,
    ) -> None:
        self.run_id = run_id
        self.target = target
        self.task = task
        self.model = model
        self.runs_dir = Path(runs_dir)
        self.max_iterations = max_iterations

        self._prices = _load_prices()
        self._started_at = datetime.now(UTC)

        # accumulators
        self._total_input_tokens:  int   = 0
        self._total_output_tokens: int   = 0
        self._total_cost_usd:      float = 0.0
        self._llm_calls:  list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._stop_reason: str = "unknown"

    # ------------------------------------------------------------------
    # Write API — called during a run
    # ------------------------------------------------------------------

    def log_llm_call(self, input_tokens: int, output_tokens: int, latency_ms: float) -> None:
        cost = _compute_cost(self.model, input_tokens, output_tokens, self._prices)
        self._total_input_tokens  += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost_usd      += cost
        self._llm_calls.append({
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "latency_ms":    round(latency_ms, 1),
            "cost_usd":      round(cost, 8),
        })

    def log_tool_call(self, name: str, args: dict, latency_ms: float) -> None:
        self._tool_calls.append({
            "name":       name,
            "args":       args,
            "latency_ms": round(latency_ms, 1),
        })

    def set_stop_reason(self, reason: str) -> None:
        """Reason values: 'flag_found', 'agent_finished', 'max_iterations', 'error'"""
        self._stop_reason = reason

    # ------------------------------------------------------------------

    def finish(self, history: list[dict]) -> dict:
        """Write both output files and return the metadata dict."""
        finished_at = datetime.now(UTC)
        duration    = (finished_at - self._started_at).total_seconds()

        flag        = _detect_flag(history)
        success     = flag is not None

        if success and self._stop_reason == "unknown":
            self._stop_reason = "flag_found"

        metadata: dict[str, Any] = {
            # identity
            "run_id":        self.run_id,
            "target":        self.target,
            "model":         self.model,

            # outcome
            "success":       success,
            "flag":          flag,
            "stop_reason":   self._stop_reason,

            # timing
            "started_at":        self._started_at.isoformat(),
            "finished_at":       finished_at.isoformat(),
            "duration_seconds":  round(duration, 3),

            # iterations
            "iterations_used":  len(self._llm_calls),
            "max_iterations":   self.max_iterations,

            # LLM usage
            "llm_calls":            len(self._llm_calls),
            "tool_calls":           len(self._tool_calls),
            "total_input_tokens":   self._total_input_tokens,
            "total_output_tokens":  self._total_output_tokens,
            "total_tokens":         self._total_input_tokens + self._total_output_tokens,
            "total_cost_usd":       round(self._total_cost_usd, 6),
            "context_messages":     len(history),

        }

        out_dir = self.runs_dir / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "context_window.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return metadata, history
