"""
Remember Bot — FastAPI application entry point.

Wires up the gateway, orchestrator, LLM router, memory subsystems,
document processor, rate limiters, and database.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI

from src.config import get_config
from src.core.context_assembler import ContextAssembler
from src.core.commands import CommandHandler as BotCommandHandler
from src.core.document_processor import DocumentProcessor
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
from src.utils.rate_limiter import RateLimiter
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
document_processor: DocumentProcessor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global telegram_gateway, whatsapp_gateway, document_processor

    config = get_config()
    logger.info("Starting Remember Bot...")

    # Create database tables (dev convenience — use Alembic for prod)
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured")

    # ------------------------------------------------------------------
    # Rate Limiters (one per API key)
    # ------------------------------------------------------------------

    # Chat rate limiters (main API key)
    chat_llm_limiter = RateLimiter(
        rpm=config.rate_limits.llm_rpm,
        tpm=config.rate_limits.llm_tpm,
        rpd=config.rate_limits.llm_rpd,
        name="chat_llm",
    )
    chat_embedding_limiter = RateLimiter(
        rpm=config.rate_limits.embedding_rpm,
        tpm=config.rate_limits.embedding_tpm,
        rpd=config.rate_limits.embedding_rpd,
        name="chat_embedding",
    )

    # Background rate limiters (dedicated background API key)
    bg_llm_limiter = RateLimiter(
        rpm=config.rate_limits.llm_rpm,
        tpm=config.rate_limits.llm_tpm,
        rpd=config.rate_limits.llm_rpd,
        name="bg_llm",
    )
    bg_embedding_limiter = RateLimiter(
        rpm=config.rate_limits.embedding_rpm,
        tpm=config.rate_limits.embedding_tpm,
        rpd=config.rate_limits.embedding_rpd,
        name="bg_embedding",
    )

    logger.info("Rate limiters initialized")

    # ------------------------------------------------------------------
    # Core services (Chat — main API key)
    # ------------------------------------------------------------------

    llm_router = LLMRouter(config, rate_limiter=chat_llm_limiter)
    embedding_service = EmbeddingService(
        config, task_name="embeddings", rate_limiter=chat_embedding_limiter
    )

    # Memory subsystems
    episodic_memory = EpisodicMemory(embedding_service)
    semantic_memory = SemanticMemory(embedding_service=embedding_service)

    context_assembler = ContextAssembler(
        config=config,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
    )

    fact_extractor = FactExtractor(
        llm_router=llm_router,
        embedding_service=embedding_service,
    )

    summarizer = ConversationSummarizer(
        config=config,
        llm_router=llm_router,
        embedding_service=embedding_service,
    )

    # ------------------------------------------------------------------
    # Document Processor (Background — dedicated API key)
    # ------------------------------------------------------------------

    redis_client: aioredis.Redis | None = None

    if config.documents.enabled and config.settings.aistudio_bg_api_key:
        try:
            redis_client = aioredis.from_url(
                config.settings.redis_url,
                decode_responses=False,
            )
            await redis_client.ping()
            logger.info("Redis connected: %s", config.settings.redis_url)

            # Background services — use the dedicated BG key
            bg_embedding_service = EmbeddingService(
                config, task_name="document_embeddings", rate_limiter=bg_embedding_limiter
            )
            bg_llm_router = LLMRouter(config, rate_limiter=bg_llm_limiter)

            # Notification callback — sends messages via the gateways
            async def notify_user(platform: str, chat_id: str, message: str) -> None:
                if platform == "telegram" and telegram_gateway:
                    await telegram_gateway.send_message(chat_id, message)
                elif platform == "whatsapp" and whatsapp_gateway:
                    await whatsapp_gateway.send_message(chat_id, message)

            document_processor = DocumentProcessor(
                config=config,
                bg_embedding_service=bg_embedding_service,
                bg_llm_router=bg_llm_router,
                redis_client=redis_client,
                notify_callback=notify_user,
            )

            # Recover any documents stuck from a previous crash
            await document_processor.recover_incomplete()

            # Start the background worker
            await document_processor.start_worker()
            logger.info("Document processor started with background API key")

        except Exception:
            logger.exception("Failed to initialize document processor — feature disabled")
            document_processor = None
            if redis_client:
                await redis_client.aclose()
                redis_client = None
    else:
        if not config.documents.enabled:
            logger.info("Document processing disabled in config")
        elif not config.settings.aistudio_bg_api_key:
            logger.warning("No AISTUDIO_BG_API_KEY set — document processing disabled")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    orchestrator = Orchestrator(
        config=config,
        llm_router=llm_router,
        context_assembler=context_assembler,
        fact_extractor=fact_extractor,
        episodic_memory=episodic_memory,
        summarizer=summarizer,
        document_processor=document_processor,
    )

    # Command handler (shared by all gateways)
    command_handler = BotCommandHandler(
        config=config,
        llm_router=llm_router,
        episodic_memory=episodic_memory,
    )

    # ------------------------------------------------------------------
    # Gateways
    # ------------------------------------------------------------------

    # Telegram gateway
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

    # WhatsApp gateway (if credentials provided)
    if config.settings.whatsapp_phone_id and config.settings.whatsapp_token:
        whatsapp_gateway = WhatsAppGateway(
            phone_id=config.settings.whatsapp_phone_id,
            token=config.settings.whatsapp_token,
            verify_token=config.settings.whatsapp_verify_token,
            app_id=int(config.settings.whatsapp_app_id) if config.settings.whatsapp_app_id else 0,
            app_secret=config.settings.whatsapp_app_secret,
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
    if document_processor:
        await document_processor.stop_worker()
    if telegram_gateway:
        await telegram_gateway.stop()
    if whatsapp_gateway:
        await whatsapp_gateway.stop()
    if redis_client:
        await redis_client.aclose()

    await engine.dispose()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Remember Bot",
    description="A memory-first chatbot with infinite context retention.",
    version="0.6.0",
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
