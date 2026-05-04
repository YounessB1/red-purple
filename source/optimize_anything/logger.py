import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from source.optimize_anything.evaluator import _get_iteration
from source.tracer.tracer import _compute_cost, _load_prices


def _empty_bucket() -> dict:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


class Logger:
    """Experiment-level accumulator for reflector, scorer, and agent LLM usage."""

    def __init__(self, reflector_model: str, judge_model: str, agent_model: str, log_dir: Path) -> None:
        self._reflector_model = reflector_model
        self._judge_model = judge_model
        self._agent_model = agent_model
        self._log_dir = log_dir
        self._lock = threading.Lock()
        self._prices = _load_prices()
        self._started_at: datetime | None = None

        self._reflector, self._scorer, self._agents, self._previous_duration = self._load_existing()

    def _load_existing(self) -> tuple[dict, dict, dict, float]:
        summary_path = self._log_dir / "experiment_summary.json"
        if not summary_path.exists():
            return _empty_bucket(), _empty_bucket(), _empty_bucket(), 0.0
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            def _bucket_from(key: str) -> dict:
                src = data.get(key, {})
                return {
                    "calls":         src.get("calls", 0),
                    "input_tokens":  src.get("input_tokens", 0),
                    "output_tokens": src.get("output_tokens", 0),
                    "cost_usd":      src.get("cost_usd", 0.0),
                }
            previous_duration = data.get("duration_seconds") or 0.0
            return _bucket_from("reflector"), _bucket_from("scorer"), _bucket_from("agents"), previous_duration
        except Exception:
            return _empty_bucket(), _empty_bucket(), _empty_bucket(), 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start_logger(self) -> None:
        self._started_at = datetime.now(UTC)

    def stop_logger(self) -> None:
        self.write_summary()

    # ── Accumulator methods ────────────────────────────────────────────────
    def _accumulate(self, bucket: dict, tag: str, model: str,
                    input_tokens: int, output_tokens: int,
                    input_text, output_text: str) -> None:
        cost = _compute_cost(model, input_tokens, output_tokens, self._prices)
        with self._lock:
            bucket["calls"]         += 1
            bucket["input_tokens"]  += input_tokens
            bucket["output_tokens"] += output_tokens
            bucket["cost_usd"]      += cost
            n = bucket["calls"]
        iteration = _get_iteration()
        call_dir = self._log_dir / f"iteration_{iteration:03d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        (call_dir / f"{tag}.json").write_text(
            json.dumps({
                "iteration":     iteration,
                "input_tokens":  input_tokens,
                "output_tokens": output_tokens,
                "cost_usd":      round(cost, 8),
                "input":         input_text,
                "output":        output_text,
            }, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

    def log_reflector(self, input_tokens: int, output_tokens: int, input_text, output_text: str) -> None:
        self._accumulate(self._reflector, "reflector", self._reflector_model,
                         input_tokens, output_tokens, input_text, output_text)

    def log_scorer(self, input_tokens: int, output_tokens: int, input_text, output_text: str) -> None:
        self._accumulate(self._scorer, "scorer", self._judge_model,
                         input_tokens, output_tokens, input_text, output_text)


    def log_agents(self, metadata: dict) -> None:
        with self._lock:
            self._agents["calls"]         += metadata.get("llm_calls", 0)
            self._agents["input_tokens"]  += metadata.get("total_input_tokens", 0)
            self._agents["output_tokens"] += metadata.get("total_output_tokens", 0)
            self._agents["cost_usd"]      += metadata.get("total_cost_usd", 0.0)

    # ── Summary ────────────────────────────────────────────────────────────

    def write_summary(self) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        session_duration = (datetime.now(UTC) - self._started_at).total_seconds() if self._started_at else 0.0
        total_duration = round(self._previous_duration + session_duration, 3)
        total_input  = self._reflector["input_tokens"]  + self._scorer["input_tokens"]  + self._agents["input_tokens"]
        total_output = self._reflector["output_tokens"] + self._scorer["output_tokens"] + self._agents["output_tokens"]
        total_cost   = self._reflector["cost_usd"]      + self._scorer["cost_usd"]      + self._agents["cost_usd"]
        summary = {
            "duration_seconds":    total_duration,
            "total_input_tokens":  total_input,
            "total_output_tokens": total_output,
            "total_tokens":        total_input + total_output,
            "total_cost_usd":      round(total_cost, 6),
            "reflector": {**self._reflector, "model": self._reflector_model, "cost_usd": round(self._reflector["cost_usd"], 6)},
            "scorer":    {**self._scorer,    "model": self._judge_model,     "cost_usd": round(self._scorer["cost_usd"],    6)},
            "agents":    {**self._agents,    "model": self._agent_model, "cost_usd": round(self._agents["cost_usd"],    6)},
        }
        (self._log_dir / "experiment_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
