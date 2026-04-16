from typing import Any

from source.agent.tools.registry import register_tool


@register_tool
def finish(result: str) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print("AGENT FINISHED")
    print(f"{'=' * 60}")
    print(result.strip())
    print(f"{'=' * 60}\n")

    return {
        "should_finish": True,
        "message": "Agent finished.",
    }
