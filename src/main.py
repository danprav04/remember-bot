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
from src.gateway.whatsapp import WhatsAppGateway
from src.llm.embeddings import EmbeddingService
from src.llm.router import LLMRouter
from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.memory.summarizer import ConversationSummarizer
from fastapi.responses import HTMLResponse
from src.utils.static_pages import PRIVACY_POLICY_HTML

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
whatsapp_gateway: WhatsAppGateway | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global telegram_gateway, whatsapp_gateway

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

    # Initialize command handler (shared by all gateways)
    command_handler = BotCommandHandler(
        config=config,
        llm_router=llm_router,
        episodic_memory=episodic_memory,
    )

    # Initialize Telegram gateway
    if config.settings.telegram_bot_token:
        telegram_gateway = TelegramGateway(
            bot_token=config.settings.telegram_bot_token,
            webhook_base_url=config.settings.webhook_base_url,
        )
        telegram_gateway.set_orchestrator(orchestrator)
        telegram_gateway.set_command_handler(command_handler)

        await telegram_gateway.setup(app)
        await telegram_gateway.start()
        logger.info("Telegram gateway started")
    else:
        logger.warning("No TELEGRAM_BOT_TOKEN set — Telegram gateway disabled")

    # Initialize WhatsApp gateway (if credentials provided)
    if config.settings.whatsapp_phone_id and config.settings.whatsapp_token:
        whatsapp_gateway = WhatsAppGateway(
            phone_id=config.settings.whatsapp_phone_id,
            token=config.settings.whatsapp_token,
            verify_token=config.settings.whatsapp_verify_token,
            app_id=int(config.settings.whatsapp_app_id) if config.settings.whatsapp_app_id else 0,
            app_secret=config.settings.whatsapp_app_secret,
            webhook_base_url=config.settings.webhook_base_url,
        )
        whatsapp_gateway.set_orchestrator(orchestrator)

        # Reuse the same command handler instance (shares link codes)
        whatsapp_gateway.set_command_handler(command_handler)

        await whatsapp_gateway.setup(app)
        await whatsapp_gateway.start()
        logger.info("WhatsApp gateway started")
    else:
        logger.warning("No WHATSAPP_PHONE_ID/TOKEN set — WhatsApp gateway disabled")

    logger.info("Remember Bot is ready!")

    yield  # App is running

    # Shutdown
    logger.info("Shutting down Remember Bot...")
    if telegram_gateway:
        await telegram_gateway.stop()
    if whatsapp_gateway:
        await whatsapp_gateway.stop()

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


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy():
    """Serve the privacy policy page."""
    return PRIVACY_POLICY_HTML
