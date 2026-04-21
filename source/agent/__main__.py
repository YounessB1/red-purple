import json
import os
from types import SimpleNamespace

from source.agent.runner import run
from source.agent.seed import TOOL_SCHEMAS, TOOLS


def _load_seed_override():
    """Load seed from SEED_OVERRIDE env var (JSON with system_prompt + default_task)."""
    raw = os.environ.get("SEED_OVERRIDE")
    if not raw:
        return None
    data = json.loads(raw)
    return SimpleNamespace(
        SYSTEM_PROMPT=data["system_prompt"],
        DEFAULT_TASK=data["default_task"],
        TOOL_SCHEMAS=data.get("tool_schemas", TOOL_SCHEMAS),
        TOOLS=TOOLS,
    )


run(seed=_load_seed_override())
