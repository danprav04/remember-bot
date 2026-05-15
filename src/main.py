"""
Remember Bot — FastAPI application entry point.

Wires up the gateway, orchestrator, LLM router, memory subsystems, and database.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import get_config
from src.core.context_assembler import ContextAssembler
from src.core.commands import CommandHandler as BotCommandHandler
from src.core.fact_extractor import FactExtractor
from src.core.orchestrator import Orchestrator
from src.db.engine import get_engine
from src.db.models import Base
from src.gateway.telegram import TelegramGateway
from src.llm.embeddings import EmbeddingService
from src.llm.router import LLMRouter
from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.memory.summarizer import ConversationSummarizer

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

    # Initialize Embedding Service
    embedding_service = EmbeddingService(config)

    # Initialize Memory Subsystems
    episodic_memory = EpisodicMemory(embedding_service)
    semantic_memory = SemanticMemory(embedding_service=embedding_service)

    # Initialize Context Assembler
    context_assembler = ContextAssembler(
        config=config,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
    )

    # Initialize Fact Extractor
    fact_extractor = FactExtractor(
        llm_router=llm_router,
        embedding_service=embedding_service,
    )

    # Initialize Conversation Summarizer
    summarizer = ConversationSummarizer(
        config=config,
        llm_router=llm_router,
        embedding_service=embedding_service,
    )

    # Initialize Orchestrator
    orchestrator = Orchestrator(
        config=config,
        llm_router=llm_router,
        context_assembler=context_assembler,
        fact_extractor=fact_extractor,
        episodic_memory=episodic_memory,
        summarizer=summarizer,
    )

    # Initialize Telegram gateway
    if config.settings.telegram_bot_token:
        telegram_gateway = TelegramGateway(
            bot_token=config.settings.telegram_bot_token,
            webhook_base_url=config.settings.webhook_base_url,
        )
        telegram_gateway.set_orchestrator(orchestrator)

        # Initialize and inject command handler
        command_handler = BotCommandHandler(
            config=config,
            llm_router=llm_router,
            episodic_memory=episodic_memory,
        )
        telegram_gateway.set_command_handler(command_handler)

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
    version="0.5.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "remember-bot"}
