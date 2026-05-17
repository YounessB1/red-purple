"""Shared utilities for the optimize_anything reflector pipeline."""

import json
import re

from source.optimize_anything.logger import Logger


def extract_new_prompt(content: str, logger: Logger) -> str:
    """Parse {changes, prompt} JSON from reflector output and return the new prompt string."""
    raw = re.sub(r"^```[^\n]*\n|```$", "", content.strip(), flags=re.MULTILINE).strip()
    if not raw.startswith("{"):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)

    changes = ""
    new_prompt = ""

    try:
        data = json.loads(raw)
        changes    = data.get("changes", "")
        new_prompt = data.get("prompt", "")
    except json.JSONDecodeError:
        # The prompt string can contain unescaped quotes (e.g. XSS payload examples),
        # which corrupts the outer JSON. Extract the two fields individually.
        m = re.search(r'"changes"\s*:\s*(\[.*?\])\s*,\s*"prompt"', raw, re.DOTALL)
        if m:
            try:
                changes = json.loads(m.group(1))
            except Exception:
                pass

        m = re.search(r'"prompt"\s*:\s*"(.*)"[\s\n]*\}[\s\n]*$', raw, re.DOTALL)
        if m:
            new_prompt = re.sub(
                r'\\(.)',
                lambda x: {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"'}.get(x.group(1), x.group(0)),
                m.group(1),
            )

    if isinstance(changes, list):
        changes = "\n".join(f"- {c}" for c in changes)
    if changes:
        logger.log_reflector_changes(changes)
        print(f"\n[reflector] Changes:\n{changes}\n", flush=True)
    if not new_prompt:
        print("[reflector] Warning: could not extract prompt from reflector response", flush=True)
        return content
    return new_prompt
