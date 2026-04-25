import json
import os
from types import SimpleNamespace

from source.agent.runner import run
from source.agent.seed import TOOL_SCHEMAS, TOOLS


def _load_seed_override():
    """Load seed from SEED_OVERRIDE env var.

    Preferred shape: {"prompt": "..."}.
    Legacy shape: {"system_prompt": "...", "default_task": "..."}.
    """
    raw = os.environ.get("SEED_OVERRIDE")
    if not raw:
        return None
    data = json.loads(raw)
    prompt = data.get("prompt")
    if prompt is None:
        prompt = f"{data['system_prompt']}\n\n{data['default_task']}"
    return SimpleNamespace(
        PROMPT=prompt,
        TOOL_SCHEMAS=data.get("tool_schemas", TOOL_SCHEMAS),
        TOOLS=TOOLS,
    )


run(seed=_load_seed_override(), model=os.environ.get("MODEL") or None)
