import logging
import os
from typing import Any

import litellm


logger = logging.getLogger(__name__)

MAX_TOTAL_TOKENS = 100_000
MIN_RECENT_MESSAGES = 15

SUMMARY_PROMPT_TEMPLATE = """You are an agent performing context
condensation for a security agent. Your job is to compress scan data while preserving
ALL operationally critical information for continuing the security assessment.

CRITICAL ELEMENTS TO PRESERVE:
- Discovered vulnerabilities and potential attack vectors
- Scan results and tool outputs (compressed but maintaining key findings)
- Access credentials, tokens, or authentication details found
- System architecture insights and potential weak points
- Progress made in the assessment
- Failed attempts and dead ends (to avoid duplication)
- Any decisions made about the testing approach

COMPRESSION GUIDELINES:
- Preserve exact technical details (URLs, paths, parameters, payloads)
- Summarize verbose tool outputs while keeping critical findings
- Maintain version numbers, specific technologies identified
- Keep exact error messages that might indicate vulnerabilities
- Compress repetitive or similar findings into consolidated form

Remember: Another security agent will use this summary to continue the assessment.
They must be able to pick up exactly where you left off without losing any
operational advantage or context needed to find vulnerabilities.

CONVERSATION SEGMENT TO SUMMARIZE:
{conversation}

Provide a technically precise summary that preserves all operational security context while
keeping the summary concise and to the point."""


def _count_tokens(text: str, model: str) -> int:
    try:
        count = litellm.token_counter(model=model, text=text)
        return int(count)
    except Exception:
        return len(text) // 4


def _get_message_tokens(msg: dict[str, Any], model: str) -> int:
    content = msg.get("content", "")
    if isinstance(content, str):
        return _count_tokens(content, model)
    return 0


def _extract_message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _summarize_messages(messages: list[dict[str, Any]], model: str, timeout: int = 30) -> dict[str, Any]:
    if not messages:
        return {"role": "user", "content": "<context_summary message_count='0'>No messages to summarize</context_summary>"}

    formatted = [f"{msg.get('role', 'unknown')}: {_extract_message_text(msg)}" for msg in messages]
    conversation = "\n".join(formatted)
    prompt = SUMMARY_PROMPT_TEMPLATE.format(conversation=conversation)

    api_key = os.environ.get("LLM_API_KEY")
    api_base = os.environ.get("LLM_API_BASE")

    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": timeout,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base

        response = litellm.completion(**kwargs)
        summary = response.choices[0].message.content or ""
        if not summary.strip():
            return messages[0]
        return {
            "role": "user",
            "content": f"<context_summary message_count='{len(messages)}'>{summary}</context_summary>",
        }
    except Exception:
        logger.exception("Failed to summarize messages")
        return messages[0]


class MemoryCompressor:
    def __init__(self, model: str, timeout: int = 120):
        self.model = model
        self.timeout = timeout

    def compress_history(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compress conversation history to stay within token limits.

        Strategy:
        1. Keep all system messages
        2. Keep minimum recent messages intact
        3. Summarize older messages when total tokens exceed limit
        """
        if not messages:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        regular_msgs = [m for m in messages if m.get("role") != "system"]

        total_tokens = sum(_get_message_tokens(m, self.model) for m in system_msgs + regular_msgs)

        if total_tokens <= MAX_TOTAL_TOKENS * 0.9:
            return messages

        recent_msgs = regular_msgs[-MIN_RECENT_MESSAGES:]
        old_msgs = regular_msgs[:-MIN_RECENT_MESSAGES]

        compressed = []
        chunk_size = 10
        for i in range(0, len(old_msgs), chunk_size):
            chunk = old_msgs[i : i + chunk_size]
            summary = _summarize_messages(chunk, self.model, self.timeout)
            if summary:
                compressed.append(summary)

        return system_msgs + compressed + recent_msgs
