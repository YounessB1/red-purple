"""Context window compactor — summarises the middle of the agent history
when it approaches the model's context limit.

System prompt and the most recent messages are always preserved.
Everything in between is replaced with an LLM-generated summary.
"""

import json
from pathlib import Path

from source.llm import LLM

_PRICES_PATH = Path(__file__).resolve().parents[1] / "tracer" / "model_prices.json"
THRESHOLD = 0.70      # compact when estimated tokens reach this fraction of the limit
RECENT_BUDGET = 0.10  # fraction of context limit reserved for recent messages verbatim

_SYSTEM = (
    "You are summarising the history of a CTF security agent to compress its context window.\n"
    "Preserve everything the agent will need to continue: discovered services, endpoints, "
    "vulnerabilities, credentials, tokens, cookies, flags, commands run and their key outputs, "
    "current hypothesis, and any dead ends already explored.\n"
    "Be concise but lose nothing security-relevant. Output only the summary, no preamble.\n"
    "Your summary MUST be under {word_limit} words."
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


def _select_recent(messages: list[dict], token_budget: int) -> list[dict]:
    """Walk backwards and keep the most recent messages that fit within token_budget."""
    selected = []
    used = 0
    for msg in reversed(messages):
        cost = len(str(msg.get("content", ""))) // 4
        if used + cost > token_budget:
            break
        selected.append(msg)
        used += cost
    return list(reversed(selected))


def compact(history: list[dict], model: str, tracer=None) -> list[dict]:
    """Return a shortened history with the middle replaced by a summary."""
    limit = _load_context_limit(model) or 32000
    recent_budget = int(limit * RECENT_BUDGET)

    system = history[0]
    recent = _select_recent(history[1:], recent_budget)
    middle = history[1 : len(history) - len(recent)]

    if not middle:
        return history

    summary_budget_tokens = int(limit * 0.30)
    word_limit = int(summary_budget_tokens * 0.75)
    summary_budget_chars = summary_budget_tokens * 4

    system_content = _SYSTEM.format(word_limit=word_limit)
    llm = LLM(model=model)
    summary, input_tokens, output_tokens = llm.generate(
        [{"role": "system", "content": system_content}] + middle
    )
    if tracer is not None:
        tracer.log_llm_call(input_tokens, output_tokens, tag="compactor")

    if len(summary) > summary_budget_chars:
        summary = summary[:summary_budget_chars]

    print(f"[compactor] context compacted — kept {len(recent)} recent messages", flush=True)
    summary_msg = {"role": "user", "content": f"<context_summary>\n{summary}\n</context_summary>"}
    return [system, summary_msg] + recent
