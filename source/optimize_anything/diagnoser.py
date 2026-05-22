"""Diagnoser — compresses one agent trace into a compact, reflector-ready diagnosis."""

import json
from pathlib import Path

from source.llm import LLM

_PRICES_PATH = Path(__file__).resolve().parents[1] / "tracer" / "model_prices.json"


def _load_context_limit(model: str) -> int | None:
    try:
        models = json.loads(_PRICES_PATH.read_text(encoding="utf-8"))["models"]
        return models.get(model, {}).get("context_length")
    except Exception:
        return None


def diagnose(
    context_window: list[dict],
    metadata: dict,
    model: str,
    logger,
    reflector_model: str = "",
    train_size: int = 1,
) -> str:
    try:
        context_limit = (_load_context_limit(reflector_model) if reflector_model else None) or 32000
        per_diagnosis_tokens = int(context_limit * 0.90 / max(train_size, 1))
        word_limit = int(per_diagnosis_tokens * 0.75)
        char_limit = per_diagnosis_tokens * 4

        trace = json.dumps(context_window, indent=2)
        prompt = DIAGNOSER_PROMPT.format(
            benchmark_id=metadata.get("benchmark_id", metadata.get("run_id", "unknown")),
            outcome="SUCCESS" if metadata.get("success") else "FAILURE",
            stop_reason=metadata.get("stop_reason", "unknown"),
            iterations_used=metadata.get("iterations_used", "unknown"),
            trace=trace,
            word_limit=word_limit,
        )
        messages = [{"role": "user", "content": prompt}]
        content, input_tokens, output_tokens = LLM(model).generate(messages)
        logger.log_diagnoser(input_tokens, output_tokens)
        if len(content) > char_limit:
            content = content[:char_limit]
        return content
    except Exception:
        return ""


DIAGNOSER_PROMPT = """\
You are reviewing the trace of a security CTF agent. The agent attempted to exploit \
a vulnerable web application to find a flag (format FLAG{{...}}).

Your job is to produce a compact failure analysis. Focus purely on what the agent did \
and where it went wrong 

## Run metadata
- Benchmark: {benchmark_id}
- Outcome: {outcome}
- Stop reason: {stop_reason}
- Iterations used: {iterations_used}

## Agent trace
{trace}

---

Produce a diagnosis using EXACTLY this format (no extra text):

**Vulnerability type**: [infer from the trace, e.g. SQLi, SSTI, IDOR, LFI, command injection]
**What the agent tried**: [2–3 sentences — tools used, hypotheses tested, commands run]
**Where it went wrong**: [the specific turn or decision that caused failure]
**What should have happened**: [the correct action or reasoning the agent should have taken]

Your response MUST be under {word_limit} words.
"""
