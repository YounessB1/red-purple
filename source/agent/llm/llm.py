import html
import os
import re
import time
import warnings
from typing import Any

import litellm

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from source.agent.llm.memory_compressor import MemoryCompressor

litellm.drop_params = True


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
    def __init__(self, tracer=None) -> None:
        self.model = os.environ.get("REDPURPLE_LLM")
        if not self.model:
            raise ValueError("REDPURPLE_LLM env var must be set (e.g. openrouter/openai/gpt-4o-mini)")
        self.api_key  = os.environ.get("LLM_API_KEY")
        self.api_base = os.environ.get("LLM_API_BASE")
        self.compressor = MemoryCompressor(model=self.model)
        self.tracer = tracer

    def generate(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        compressed = self.compressor.compress_history(messages)

        kwargs: dict[str, Any] = {"model": self.model, "messages": compressed}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        t0 = time.perf_counter()
        response = litellm.completion(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        content       = response.choices[0].message.content or ""
        input_tokens  = response.usage.prompt_tokens     if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        if self.tracer is not None:
            self.tracer.log_llm_call(input_tokens, output_tokens, latency_ms)

        return content, parse_tool_calls(content)
