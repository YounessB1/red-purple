"""LLM-based prompt reflector for the GEPA optimization loop."""

from source.agent.base_prompt import BASE_PROMPT_SUMMARY
from source.llm import LLM
from source.optimize_anything.logger import Logger
from source.optimize_anything.utils import extract_new_prompt

# ── Reflection prompt template ─────────────────────────────────────────

_REFLECTOR_PROMPT = f"""
## Fixed Agent Infrastructure (read-only — already injected before the strategy instructions, do NOT reproduce or modify this)
```
{BASE_PROMPT_SUMMARY}
```

## Current strategy instructions (this is what you must improve)
```
<curr_param>
```

## Failure diagnoses from recent agent runs
Each entry shows what the agent tried, where it went wrong, and what it should have done instead:
```
<side_info>
```

## Your task
Write improved strategy instructions for the agent.

Focus only on attack strategy, methodology, and domain knowledge — do NOT redefine the role,
target, tools, or tool call format (those are fixed above).

Extract every generalizable insight from the failure diagnoses: vulnerability patterns,
correct exploitation techniques, step-by-step approaches that would have worked.
Include specific commands and techniques the agent should try.

Respond with a JSON object with exactly two fields:
- "changes": 3-5 bullet points explaining what you changed and why, referencing specific failure patterns
- "prompt": the full new strategy instructions as a string

Output only the JSON object, no preamble.
"""


def build_reflection_prompt_template(background_context: str) -> str:
    return f"## Domain Context\n\n{background_context}\n" + _REFLECTOR_PROMPT


# ── LLM-based reflector ────────────────────────────────────────────────

class ReflectorLLM:
    """Single-call LLM reflector — receives pre-formatted diagnoses and returns an improved prompt."""

    def __init__(self, model: str, logger: Logger) -> None:
        self._llm = LLM(model)
        self._logger = logger

    def __call__(self, prompt: str | list[dict]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        content, input_tokens, output_tokens = self._llm.generate(messages)
        self._logger.log_reflector(input_tokens, output_tokens, messages, content)
        return extract_new_prompt(content, self._logger)
