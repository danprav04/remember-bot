"""
Unified LLM provider — wraps the OpenAI SDK to talk to any OpenAI-compatible API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from src.config import ProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any provider."""
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMProvider:
    """Thin wrapper around AsyncOpenAI that targets a specific provider's base URL."""

    def __init__(self, config: ProviderConfig):
        self.name = config.name
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat completion request and return a standardized response."""
        start = time.monotonic()

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = await self.client.chat.completions.create(**kwargs)
        except Exception:
            logger.exception("LLM request failed on provider=%s model=%s", self.name, model)
            raise

        elapsed_ms = (time.monotonic() - start) * 1000
        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=model,
            provider=self.name,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=round(elapsed_ms, 1),
        )
