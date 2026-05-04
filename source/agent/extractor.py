"""Tool output extractor — condenses large tool results using an LLM.

Skips extraction when the output is short enough to fit comfortably in context.
For very large outputs, uses incremental chunk summarization: each chunk is
condensed together with the running summary from previous chunks.
Never drops security-relevant findings; removes decorative noise instead.
"""

import json
from pathlib import Path

from source.llm import LLM

_PRICES_PATH = Path(__file__).resolve().parents[1] / "tracer" / "model_prices.json"

THRESHOLD = 500  # chars — outputs below this are returned as-is

_SINGLE_FRACTION = 0.60
_CHUNK_FRACTION  = 0.30
_DEFAULT_MAX_SINGLE = 24_000
_DEFAULT_CHUNK_SIZE = 12_000

_SYSTEM = (
    "You are a security finding extractor for a CTF agent.\n"
    "Given raw tool output, or a running summary followed by a new chunk of output, "
    "extract ALL security-relevant information concisely.\n"
    "Never drop: flags, tokens, credentials, API keys, hints, hidden form values, "
    "HTML comments, cookies, error messages, file paths, version numbers, "
    "usernames, passwords, hashes, interesting source code.\n"
    "Remove: decorative HTML markup, boilerplate, repeated whitespace, irrelevant styling.\n"
)


def _load_context_limit(model: str) -> int | None:
    try:
        models = json.loads(_PRICES_PATH.read_text(encoding="utf-8"))["models"]
        return models.get(model, {}).get("context_length")
    except Exception:
        return None


def _single_extract(text: str, model: str, tracer=None) -> str:
    """Single LLM extraction call. Returns raw summary string. Raises on failure."""
    llm = LLM(model=model)
    summary, input_tokens, output_tokens = llm.generate([
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": text},
    ])
    if tracer is not None:
        tracer.log_llm_call(input_tokens, output_tokens, tag="scorer")
    return summary


def _incremental_extract(text: str, model: str, tracer=None, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> str:
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    running_summary = ""
    for chunk in chunks:
        input_text = f"Running summary:\n{running_summary}\n\nNew chunk:\n{chunk}" if running_summary else chunk
        try:
            running_summary = _single_extract(input_text, model, tracer)
        except Exception:
            running_summary = f"{running_summary}\n[raw chunk - extraction failed]:\n{chunk[:2_000]}"
    return running_summary


def extract(tool_output: str, model: str, tracer=None) -> str:
    """Return a condensed version of tool_output, or the original if short enough."""
    if len(tool_output) <= THRESHOLD:
        return tool_output

    limit = _load_context_limit(model)
    if limit:
        max_single = int(limit * _SINGLE_FRACTION * 4)
        chunk_size = int(limit * _CHUNK_FRACTION * 4)
    else:
        max_single = _DEFAULT_MAX_SINGLE
        chunk_size = _DEFAULT_CHUNK_SIZE

    try:
        if len(tool_output) <= max_single:
            result = _single_extract(tool_output, model, tracer)
        else:
            result = _incremental_extract(tool_output, model, tracer, chunk_size)
        return f"<extracted_output>\n{result}\n</extracted_output>"
    except Exception:
        return tool_output
