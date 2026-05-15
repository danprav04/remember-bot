"""
Remember Bot — FastAPI application entry point.

Wires up the gateway, orchestrator, LLM router, and database.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import get_config
from src.core.orchestrator import Orchestrator
from src.db.engine import get_engine
from src.db.models import Base
from src.gateway.telegram import TelegramGateway
from src.llm.router import LLMRouter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

telegram_gateway: TelegramGateway | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global telegram_gateway

    config = get_config()
    logger.info("Starting Remember Bot...")

    # Create database tables (dev convenience — use Alembic for prod)
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured")

    # Initialize LLM Router
    llm_router = LLMRouter(config)

    # Initialize Orchestrator
    orchestrator = Orchestrator(config, llm_router)

    # Initialize Telegram gateway
    if config.settings.telegram_bot_token:
        telegram_gateway = TelegramGateway(
            bot_token=config.settings.telegram_bot_token,
            webhook_base_url=config.settings.webhook_base_url,
        )
        telegram_gateway.set_orchestrator(orchestrator)
        await telegram_gateway.setup(app)
        await telegram_gateway.start()
        logger.info("Telegram gateway started")
    else:
        logger.warning("No TELEGRAM_BOT_TOKEN set — Telegram gateway disabled")

    logger.info("Remember Bot is ready!")

    yield  # App is running

    # Shutdown
    logger.info("Shutting down Remember Bot...")
    if telegram_gateway:
        await telegram_gateway.stop()

    await engine.dispose()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Remember Bot",
    description="A memory-first chatbot with infinite context retention.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "remember-bot"}
