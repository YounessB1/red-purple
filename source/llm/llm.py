import os
import time
import warnings
from typing import Any
import litellm

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

litellm.drop_params = True


class LLM:
    def __init__(self, model: str) -> None:
        self.model = model
        self.api_key = os.environ.get("OPENROUTER_API_KEY")

    def generate(self, messages: list[dict[str, Any]], max_retries: int = 3) -> tuple[str, int, int]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if self.api_key:
            kwargs["api_key"] = self.api_key

        for attempt in range(max_retries):
            try:
                response = litellm.completion(**kwargs, timeout=120)
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)

        content       = response.choices[0].message.content or ""
        input_tokens  = response.usage.prompt_tokens     if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return content, input_tokens, output_tokens

    def __call__(self, prompt: str | list[dict]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        content, _, _ = self.generate(messages)
        return content
