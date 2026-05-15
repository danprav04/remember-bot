"""
Telegram gateway — handles incoming messages via webhook and sends responses.
Uses python-telegram-bot v22 with FastAPI webhook integration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import Request, Response
from telegram import Bot, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.gateway.base import BaseGateway, IncomingMessage

if TYPE_CHECKING:
    from src.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class TelegramGateway(BaseGateway):
    """Telegram Bot integration via webhooks."""

    def __init__(self, bot_token: str, webhook_base_url: str):
        self.bot_token = bot_token
        self.webhook_base_url = webhook_base_url
        self.webhook_path = "/webhook/telegram"
        self.webhook_url = f"{webhook_base_url}{self.webhook_path}"

        self._orchestrator: Orchestrator | None = None
        self._application: Application | None = None

    def set_orchestrator(self, orchestrator: Orchestrator) -> None:
        """Inject the orchestrator dependency (avoids circular imports)."""
        self._orchestrator = orchestrator

    async def setup(self, app) -> None:
        """Register the webhook endpoint with FastAPI."""

        # Build the telegram Application (but don't start polling — we use webhooks)
        self._application = (
            Application.builder()
            .token(self.bot_token)
            .updater(None)  # No built-in updater; we feed updates from FastAPI
            .build()
        )

        # Register handlers
        self._application.add_handler(CommandHandler("start", self._handle_start))
        self._application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        # Initialize the application (sets up the bot object, etc.)
        await self._application.initialize()

        # Register FastAPI route for webhook
        @app.post(self.webhook_path)
        async def telegram_webhook(request: Request) -> Response:
            """Receive Telegram webhook updates."""
            data = await request.json()
            update = Update.de_json(data, self._application.bot)
            await self._application.process_update(update)
            return Response(status_code=200)

        logger.info("Telegram webhook route registered at %s", self.webhook_path)

    async def start(self) -> None:
        """Set the webhook URL with Telegram."""
        if self._application is None:
            raise RuntimeError("Gateway not set up — call setup() first")

        await self._application.start()

        # Set webhook
        await self._application.bot.set_webhook(
            url=self.webhook_url,
            allowed_updates=["message"],
        )
        logger.info("Telegram webhook set: %s", self.webhook_url)

    async def stop(self) -> None:
        """Remove webhook and shut down."""
        if self._application:
            await self._application.bot.delete_webhook()
            await self._application.stop()
            await self._application.shutdown()
            logger.info("Telegram gateway stopped")

    async def send_message(self, chat_id: str, text: str) -> None:
        """Send a text message to a Telegram chat."""
        if self._application is None:
            raise RuntimeError("Gateway not set up")

        # Telegram has a 4096 char limit per message — split if needed
        max_len = 4096
        for i in range(0, len(text), max_len):
            chunk = text[i : i + max_len]
            await self._application.bot.send_message(
                chat_id=int(chat_id), text=chunk
            )

    # ------------------------------------------------------------------
    # Telegram handlers
    # ------------------------------------------------------------------

    async def _handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /start command."""
        await update.message.reply_text(
            "👋 Hi! I'm your memory bot. Tell me anything and I'll remember it.\n\n"
            "Just chat naturally — I'll remember everything you tell me."
        )

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming text messages."""
        if not update.message or not update.message.text:
            return

        user = update.message.from_user
        incoming = IncomingMessage(
            platform="telegram",
            platform_user_id=str(user.id),
            platform_chat_id=str(update.message.chat_id),
            display_name=user.full_name,
            text=update.message.text,
        )

        logger.info(
            "Incoming message from %s (%s): %s",
            incoming.display_name,
            incoming.platform_user_id,
            incoming.text[:100],
        )

        if self._orchestrator is None:
            await update.message.reply_text("⚠️ Bot is still starting up, please wait...")
            return

        try:
            response_text = await self._orchestrator.handle_message(incoming)
            await update.message.reply_text(response_text)
        except Exception:
            logger.exception("Error processing message from %s", incoming.platform_user_id)
            await update.message.reply_text(
                "❌ Sorry, something went wrong. Please try again."
            )
