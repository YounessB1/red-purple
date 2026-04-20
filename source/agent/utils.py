"""Agent utilities — workflow plumbing that GEPA does not evolve."""

import html
import json
import re
import time
from typing import Any

from source.tracer import Tracer


def parse_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse <invoke name="...">...</invoke> blocks from LLM response."""
    calls = []
    for match in re.finditer(r'<invoke name="([^"]+)">(.*?)</invoke>', content, re.DOTALL):
        name = match.group(1)
        args_xml = match.group(2)
        args = {}
        for param in re.finditer(r"<(\w+)>(.*?)</\1>", args_xml, re.DOTALL):
            key, val = param.group(1), param.group(2).strip()
            try:
                args[key] = int(val)
            except ValueError:
                args[key] = html.unescape(val)
        calls.append({"name": name, "args": args})
    return calls


def run_tool(call: dict, tracer: Tracer, tools: dict[str, Any]) -> tuple[str, bool]:
    """Dispatch a single tool call and return (json_output, should_finish)."""
    func = tools.get(call["name"])
    if not func:
        err = json.dumps({"error": f"Unknown tool: {call['name']}"})
        tracer.log_tool_call(call["name"], call["args"], 0.0)
        return err, False
    try:
        t0 = time.perf_counter()
        result = func(**call["args"])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        should_finish = bool(result.get("should_finish", False)) if isinstance(result, dict) else False
        output = json.dumps(result, indent=2)
        tracer.log_tool_call(call["name"], call["args"], latency_ms)
        return output, should_finish
    except Exception as e:
        tracer.log_tool_call(call["name"], call["args"], 0.0)
        return json.dumps({"error": str(e)}), False
