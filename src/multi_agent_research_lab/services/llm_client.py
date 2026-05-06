"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import time
from dataclasses import dataclass

from openai import APIError, OpenAI

from multi_agent_research_lab.core.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


# gpt-4o-mini pricing per 1K tokens (USD)
_COST_PER_1K = {"input": 0.000150, "output": 0.000600}

_MAX_RETRIES = 2
_RETRY_DELAY = 2.0  # seconds


class LLMClient:
    """Provider-agnostic LLM client với retry và timeout."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.timeout_seconds,
        )
        self._model = settings.openai_model

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion. Retry tối đa 2 lần nếu API lỗi."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = resp.choices[0].message.content or ""
                usage = resp.usage
                input_tokens = usage.prompt_tokens if usage else None
                output_tokens = usage.completion_tokens if usage else None
                cost = None
                if input_tokens is not None and output_tokens is not None:
                    cost = (
                        input_tokens / 1000 * _COST_PER_1K["input"]
                        + output_tokens / 1000 * _COST_PER_1K["output"]
                    )
                return LLMResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )
            except APIError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
        raise RuntimeError(f"LLM call failed after {_MAX_RETRIES + 1} attempts") from last_exc
