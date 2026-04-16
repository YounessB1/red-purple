import html
import os
import re
import warnings
from typing import Any

import litellm

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from source.agent.llm.memory_compressor import MemoryCompressor

litellm.drop_params = True

if os.environ.get("LANGFUSE_PUBLIC_KEY"):
    litellm.success_callback = ["langfuse"]


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


class LLM:
    def __init__(self) -> None:
        self.model = os.environ.get("REDPURPLE_LLM")
        if not self.model:
            raise ValueError("REDPURPLE_LLM env var must be set (e.g. openrouter/openai/gpt-4o-mini)")
        self.api_key = os.environ.get("LLM_API_KEY")
        self.api_base = os.environ.get("LLM_API_BASE")
        self.compressor = MemoryCompressor(model=self.model)

    def generate(self, messages: list[dict[str, Any]], metadata: dict | None = None) -> tuple[str, list[dict[str, Any]]]:
        compressed = self.compressor.compress_history(messages)

        kwargs: dict[str, Any] = {"model": self.model, "messages": compressed}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if metadata:
            kwargs["metadata"] = metadata

        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content or ""
        tool_calls = parse_tool_calls(content)
        return content, tool_calls
