"""
LLM Router — selects the right provider + model for each task, with fallback chains.
"""

from __future__ import annotations

import logging

from src.config import AppConfig, FallbackEntry, TaskConfig
from src.llm.provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Routes LLM requests to the configured provider + model per task.
    Implements fallback chains: if the primary provider fails, tries each
    fallback in order.
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # Pre-build a provider instance for each configured provider
        self._providers: dict[str, LLMProvider] = {
            name: LLMProvider(pcfg)
            for name, pcfg in config.llm.providers.items()
            if pcfg.api_key  # skip providers without an API key
        }

        if not self._providers:
            raise RuntimeError(
                "No AI providers configured! Set at least one API key in .env"
            )

        logger.info(
            "LLM Router initialized with providers: %s",
            list(self._providers.keys()),
        )

    def _get_provider(self, name: str) -> LLMProvider:
        provider = self._providers.get(name)
        if provider is None:
            raise KeyError(
                f"Provider '{name}' not available. "
                f"Available: {list(self._providers.keys())}"
            )
        return provider

    async def chat(
        self,
        task: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion for the given task (e.g. 'chat', 'fact_extraction').
        Uses the configured provider + model, falling back if enabled.
        """
        task_config = self.config.llm.tasks.get(task)
        if task_config is None:
            raise KeyError(f"No LLM task config found for '{task}'")

        # Build the attempt chain: primary + fallbacks
        attempts: list[tuple[str, str]] = [(task_config.provider, task_config.model)]
        if self.config.llm.fallback_enabled:
            for fb in task_config.fallbacks:
                attempts.append((fb.provider, fb.model))

        last_error: Exception | None = None
        for provider_name, model in attempts:
            # Skip providers that aren't available (missing API key)
            if provider_name not in self._providers:
                logger.warning(
                    "Skipping fallback provider '%s' — not configured", provider_name
                )
                continue

            try:
                provider = self._get_provider(provider_name)
                result = await provider.chat(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if provider_name != task_config.provider:
                    logger.info(
                        "Task '%s' served by fallback: %s/%s", task, provider_name, model
                    )
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    "Provider '%s' model '%s' failed for task '%s': %s",
                    provider_name, model, task, e,
                )
                continue

        raise RuntimeError(
            f"All providers failed for task '{task}'. Last error: {last_error}"
        )

    async def get_task_info(self, task: str) -> dict:
        """Return the current provider + model config for a task (for metadata)."""
        task_config = self.config.llm.tasks.get(task)
        if task_config is None:
            return {"task": task, "error": "not configured"}
        return {
            "task": task,
            "provider": task_config.provider,
            "model": task_config.model,
            "fallbacks": [
                {"provider": fb.provider, "model": fb.model}
                for fb in task_config.fallbacks
            ],
        }
