"""
Embedding service — generates text embeddings via AI Studio (Gemini Embedding 2).
Supports optional rate limiting for API quota management.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from src.config import AppConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate text embeddings using the configured embedding provider."""

    def __init__(self, config: AppConfig, task_name: str = "embeddings", rate_limiter=None):
        self._rate_limiter = rate_limiter

        task_config = config.llm.tasks.get(task_name)
        if task_config is None:
            logger.warning("No '%s' task configured — embedding service disabled", task_name)
            self._client = None
            self._model = ""
            return

        provider_config = config.llm.providers.get(task_config.provider)
        if provider_config is None or not provider_config.api_key:
            logger.warning("Embedding provider '%s' not available", task_config.provider)
            self._client = None
            self._model = ""
            return

        self._client = AsyncOpenAI(
            base_url=provider_config.base_url,
            api_key=provider_config.api_key,
        )
        self._model = task_config.model
        logger.info(
            "Embedding service ready: provider=%s model=%s task=%s rate_limited=%s",
            task_config.provider, self._model, task_name, rate_limiter is not None,
        )

    @property
    def available(self) -> bool:
        return self._client is not None

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text."""
        if not self._client:
            raise RuntimeError("Embedding service is not configured")

        # Rough token estimate for rate limiting (1 token ≈ 4 chars)
        estimated_tokens = max(1, len(text) // 4)

        if self._rate_limiter:
            await self._rate_limiter.acquire(estimated_tokens)

        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in one call."""
        if not self._client:
            raise RuntimeError("Embedding service is not configured")

        # Rate-limit the batch as a single request with total token count
        total_tokens = sum(max(1, len(t) // 4) for t in texts)

        if self._rate_limiter:
            await self._rate_limiter.acquire(total_tokens)

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]
