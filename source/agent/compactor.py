"""Context window compactor — summarises the middle of the agent history
when it approaches the model's context limit.

System prompt and the most recent messages are always preserved.
Everything in between is replaced with an LLM-generated summary.
"""

import json
from pathlib import Path

from source.llm import LLM

_PRICES_PATH = Path(__file__).resolve().parents[1] / "tracer" / "model_prices.json"
THRESHOLD = 0.70   # compact when estimated tokens reach this fraction of the limit
RECENT_MESSAGES = 6  # number of recent messages to keep verbatim after compaction

_SYSTEM = (
    "You are summarising the history of a CTF security agent to compress its context window.\n"
    "Preserve everything the agent will need to continue: discovered services, endpoints, "
    "vulnerabilities, credentials, tokens, cookies, flags, commands run and their key outputs, "
    "current hypothesis, and any dead ends already explored.\n"
    "Be concise but lose nothing security-relevant. Output only the summary, no preamble."
)


def _load_context_limit(model: str) -> int | None:
    try:
        models = json.loads(_PRICES_PATH.read_text(encoding="utf-8"))["models"]
        return models.get(model, {}).get("context_length")
    except Exception:
        return None


def _estimate_tokens(history: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in history) // 4


def should_compact(history: list[dict], model: str) -> bool:
    limit = _load_context_limit(model)
    if not limit:
        return False
    return _estimate_tokens(history) >= THRESHOLD * limit


def compact(history: list[dict], model: str, tracer=None) -> list[dict]:
    """Return a shortened history with the middle replaced by a summary."""
    if len(history) <= RECENT_MESSAGES + 1:
        return history

    system  = history[0]
    middle  = history[1 : -RECENT_MESSAGES]
    recent  = history[-RECENT_MESSAGES:]

    conversation = "\n\n".join(
        f"{m['role'].upper()}: {m.get('content', '')}" for m in middle
    )

    llm = LLM(model=model)
    summary, input_tokens, output_tokens = llm.generate([
        {"role": "system",  "content": _SYSTEM},
        {"role": "user",    "content": conversation},
    ])
    if tracer is not None:
        tracer.log_llm_call(input_tokens, output_tokens, tag="compactor")

    print(f"[compactor] context compacted — kept {RECENT_MESSAGES} recent messages", flush=True)
    summary_msg = {"role": "user", "content": f"<context_summary>\n{summary}\n</context_summary>"}
    return [system, summary_msg] + recent
