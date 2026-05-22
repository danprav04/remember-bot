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

    # WhatsApp (Cloud API via PyWa)
    whatsapp_phone_id: str = Field(default="", description="WhatsApp phone number ID")
    whatsapp_token: str = Field(default="", description="WhatsApp Cloud API access token")
    whatsapp_verify_token: str = Field(default="", description="Webhook verification token (you choose this)")
    whatsapp_app_id: str = Field(default="", description="Meta App ID")
    whatsapp_app_secret: str = Field(default="", description="Meta App Secret")

    # Webhook
    webhook_base_url: str = Field(
        default="http://localhost:8000",
        description="Public base URL for webhooks (ngrok or domain)",
    )

    # AI provider API keys
    aistudio_api_key: str = Field(default="", description="Google AI Studio API key")
    aistudio_bg_api_key: str = Field(default="", description="Google AI Studio API key for background processing")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    aihubmix_api_key: str = Field(default="", description="AIhubmix API key")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

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


class DecayConfig:
    """Memory decay settings from config.yaml."""

    def __init__(self, data: dict[str, Any]):
        decay = data.get("decay", {})
        self.enabled: bool = decay.get("enabled", True)
        self.decay_factor: float = decay.get("decay_factor", 0.95)
        self.min_relevance: float = decay.get("min_relevance", 0.1)
        self.min_age_hours: int = decay.get("min_age_hours", 24)
        self.interval_messages: int = decay.get("interval_messages", 50)


class RateLimitConfig:
    """Rate limit settings per API from config.yaml."""

    def __init__(self, data: dict[str, Any]):
        rl = data.get("rate_limits", {})
        llm_rl = rl.get("llm", {})
        emb_rl = rl.get("embeddings", {})

        self.llm_rpm: int = llm_rl.get("rpm", 15)
        self.llm_tpm: int = llm_rl.get("tpm", 250_000)
        self.llm_rpd: int = llm_rl.get("rpd", 500)

        self.embedding_rpm: int = emb_rl.get("rpm", 100)
        self.embedding_tpm: int = emb_rl.get("tpm", 30_000)
        self.embedding_rpd: int = emb_rl.get("rpd", 1_000)


class DocumentConfig:
    """Document processing settings from config.yaml."""

    def __init__(self, data: dict[str, Any]):
        doc = data.get("documents", {})
        self.enabled: bool = doc.get("enabled", True)
        self.max_file_size_mb: int = doc.get("max_file_size_mb", 20)
        self.chunk_size_tokens: int = doc.get("chunk_size_tokens", 500)
        self.chunk_overlap_tokens: int = doc.get("chunk_overlap_tokens", 50)
        self.fact_extraction_batch_size: int = doc.get("fact_extraction_batch_size", 3)
        self.max_concurrent_documents: int = doc.get("max_concurrent_documents", 2)


class AppConfig:
    """Top-level application configuration combining env vars and config.yaml."""

    def __init__(self):
        self.settings = Settings()
        yaml_data = _load_yaml_config()
        self.llm = LLMConfig(yaml_data, self.settings)
        self.memory = MemoryConfig(yaml_data)
        self.bot = BotConfig(yaml_data)
        self.decay = DecayConfig(yaml_data)
        self.rate_limits = RateLimitConfig(yaml_data)
        self.documents = DocumentConfig(yaml_data)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Singleton config accessor."""
    return AppConfig()
