"""Tool output extractor — condenses large tool results using an LLM.

Skips extraction when the output is short enough to fit comfortably in context.
Never drops security-relevant findings; removes decorative noise instead.
"""

from source.llm import LLM

THRESHOLD = 500  # chars — outputs below this are returned as-is

_SYSTEM = (
    "You are a security finding extractor for a CTF agent.\n"
    "Given raw tool output, extract ALL security-relevant information concisely.\n"
    "Never drop: flags, tokens, credentials, API keys, hints, hidden form values, "
    "HTML comments, cookies, error messages, file paths, version numbers, "
    "usernames, passwords, hashes, interesting source code.\n"
    "Remove: decorative HTML markup, boilerplate, repeated whitespace, irrelevant styling.\n"
    "Output only the extracted findings, no preamble."
)


def extract(tool_output: str, model: str, tracer=None) -> str:
    """Return a condensed version of tool_output, or the original if short enough."""
    if len(tool_output) <= THRESHOLD:
        return tool_output

    try:
        llm = LLM(model=model, tracer=tracer)
        summary = llm.generate([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": tool_output},
        ])
        return f"<extracted_output>\n{summary}\n</extracted_output>"
    except Exception:
        return tool_output
