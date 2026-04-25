import os
import time
import warnings
from typing import Any
import litellm

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

litellm.drop_params = True


class LLM:
    def __init__(self, model: str, tracer=None) -> None:
        self.model = model
        self.api_key  = os.environ.get("OPENROUTER_API_KEY")
        self.api_base = os.environ.get("LLM_API_BASE")
        self.tracer = tracer

    def generate(self, messages: list[dict[str, Any]], max_retries: int = 3) -> str:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        t0 = time.perf_counter()
        for attempt in range(max_retries):
            try:
                response = litellm.completion(**kwargs)
                break
            except (litellm.exceptions.APIError, litellm.exceptions.APIConnectionError):
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        content       = response.choices[0].message.content or ""
        input_tokens  = response.usage.prompt_tokens     if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        if self.tracer is not None:
            self.tracer.log_llm_call(input_tokens, output_tokens, latency_ms)

        return content
