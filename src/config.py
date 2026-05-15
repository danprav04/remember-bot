"""
Configuration module — loads settings from environment variables and config.yaml.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Locate config.yaml (next to the project root, or /app in Docker)
# ---------------------------------------------------------------------------

def _find_config_yaml() -> Path:
    """Walk upward from this file to find config.yaml."""
    candidates = [
        Path("/app/config.yaml"),                          # Docker mount
        Path(__file__).resolve().parent.parent / "config.yaml",  # Dev layout
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("config.yaml not found in any expected location")


def _load_yaml_config() -> dict[str, Any]:
    path = _find_config_yaml()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Pydantic settings — env vars (.env / docker env)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Core application settings sourced from environment variables."""

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://bot:botpass@localhost:5432/rememberbot",
        description="Async SQLAlchemy connection string",
    )

    # Telegram
    telegram_bot_token: str = Field(default="", description="Telegram Bot API token")

    # Webhook
    webhook_base_url: str = Field(
        default="http://localhost:8000",
        description="Public base URL for webhooks (ngrok or domain)",
    )

    # AI provider API keys
    aistudio_api_key: str = Field(default="", description="Google AI Studio API key")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    aihubmix_api_key: str = Field(default="", description="AIhubmix API key")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# ---------------------------------------------------------------------------
# Typed wrappers for config.yaml sections
# ---------------------------------------------------------------------------

class ProviderConfig:
    """Represents a single AI provider from config.yaml."""

    def __init__(self, name: str, data: dict[str, Any], settings: Settings):
        self.name = name
        self.base_url: str = data["base_url"]
        api_key_env: str = data["api_key_env"]
        self.api_key: str = getattr(settings, api_key_env.lower(), "") or os.getenv(api_key_env, "")


class FallbackEntry:
    """A single fallback step (provider + model)."""

    def __init__(self, data: dict[str, str]):
        self.provider: str = data["provider"]
        self.model: str = data["model"]


class TaskConfig:
    """Configuration for a single LLM task (chat, fact_extraction, etc.)."""

    def __init__(self, name: str, data: dict[str, Any]):
        self.name = name
        self.provider: str = data["provider"]
        self.model: str = data["model"]
        self.fallbacks: list[FallbackEntry] = [
            FallbackEntry(fb) for fb in data.get("fallback", [])
        ]


class LLMConfig:
    """Full LLM configuration parsed from config.yaml."""

    def __init__(self, data: dict[str, Any], settings: Settings):
        llm_data = data.get("llm", {})

        # Providers
        self.providers: dict[str, ProviderConfig] = {
            name: ProviderConfig(name, pdata, settings)
            for name, pdata in llm_data.get("providers", {}).items()
        }

        # Tasks
        self.tasks: dict[str, TaskConfig] = {
            name: TaskConfig(name, tdata)
            for name, tdata in llm_data.get("tasks", {}).items()
        }

        self.fallback_enabled: bool = llm_data.get("fallback_enabled", True)


class MemoryConfig:
    """Memory-related settings from config.yaml."""

    def __init__(self, data: dict[str, Any]):
        mem = data.get("memory", {})
        self.working_memory_size: int = mem.get("working_memory_size", 20)
        self.episodic_top_k: int = mem.get("episodic_top_k", 5)
        self.summary_threshold: int = mem.get("summary_threshold", 50)
        self.embedding_dimensions: int = mem.get("embedding_dimensions", 3072)
        self.max_context_tokens: int = mem.get("max_context_tokens", 8000)


class BotConfig:
    """Bot personality and behavior settings from config.yaml."""

    def __init__(self, data: dict[str, Any]):
        bot = data.get("bot", {})
        self.system_prompt: str = bot.get("system_prompt", "You are a helpful assistant.")


class AppConfig:
    """Top-level application configuration combining env vars and config.yaml."""

    def __init__(self):
        self.settings = Settings()
        yaml_data = _load_yaml_config()
        self.llm = LLMConfig(yaml_data, self.settings)
        self.memory = MemoryConfig(yaml_data)
        self.bot = BotConfig(yaml_data)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Singleton config accessor."""
    return AppConfig()
