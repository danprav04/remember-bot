"""
Telegram gateway — handles incoming messages via webhook and sends responses.
Supports text, voice, and photo messages.
Uses python-telegram-bot v22 with FastAPI webhook integration.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
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
    from src.core.commands import CommandHandler as BotCommandHandler
    from src.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class TelegramGateway(BaseGateway):
    """Telegram Bot integration via webhooks."""

    def __init__(self, bot_token: str, webhook_base_url: str):
        self.bot_token = bot_token
        self.webhook_base_url = webhook_base_url
        self.webhook_path = "/webhook/telegram"
        self.webhook_url = f"{webhook_base_url}{self.webhook_path}"
        if not self.webhook_url.startswith("http"):
            self.webhook_url = f"https://{self.webhook_url}"

        self._orchestrator: Orchestrator | None = None
        self._command_handler: BotCommandHandler | None = None
        self._application: Application | None = None

    def set_orchestrator(self, orchestrator: Orchestrator) -> None:
        """Inject the orchestrator dependency (avoids circular imports)."""
        self._orchestrator = orchestrator

    def set_command_handler(self, handler: BotCommandHandler) -> None:
        """Inject the command handler dependency."""
        self._command_handler = handler

    async def setup(self, app) -> None:
        """Register the webhook endpoint with FastAPI."""

        # Build the telegram Application (but don't start polling — we use webhooks)
        self._application = (
            Application.builder()
            .token(self.bot_token)
            .updater(None)  # No built-in updater; we feed updates from FastAPI
            .build()
        )

        # Register command handlers
        self._application.add_handler(CommandHandler("start", self._handle_start))
        self._application.add_handler(CommandHandler("help", self._handle_help))
        self._application.add_handler(CommandHandler("facts", self._handle_facts))
        self._application.add_handler(CommandHandler("search", self._handle_search))
        self._application.add_handler(CommandHandler("forget", self._handle_forget))
        self._application.add_handler(CommandHandler("model", self._handle_model))
        self._application.add_handler(CommandHandler("stats", self._handle_stats))
        self._application.add_handler(CommandHandler("export", self._handle_export))
        self._application.add_handler(CommandHandler("connect", self._handle_connect))

        # Media message handlers
        self._application.add_handler(
            MessageHandler(filters.VOICE | filters.AUDIO, self._handle_voice)
        )
        self._application.add_handler(
            MessageHandler(filters.PHOTO, self._handle_photo)
        )

        # Text message handler (must be last — catches everything else)
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
                chat_id=int(chat_id),
                text=chunk,
                parse_mode="Markdown",
            )

    # ------------------------------------------------------------------
    # Helper — download Telegram file to base64
    # ------------------------------------------------------------------

    async def _download_file_base64(self, file_id: str) -> bytes:
        """Download a Telegram file and return raw bytes."""
        tg_file = await self._application.bot.get_file(file_id)
        buf = BytesIO()
        await tg_file.download_to_memory(buf)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Helper — Send Formatted Message
    # ------------------------------------------------------------------

    async def _safe_reply(self, update: Update, text: str) -> None:
        """Reply with Telegram-safe HTML, falling back to plain text if parsing fails."""
        import re

        # Escape HTML special chars first
        formatted = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Code blocks: ```lang\ncode\n``` or ```code```
        formatted = re.sub(
            r'```(?:\w+)?\n(.*?)\n```', r'<pre><code>\1</code></pre>', formatted, flags=re.DOTALL
        )
        formatted = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', formatted, flags=re.DOTALL)

        # Inline code: `code`
        formatted = re.sub(r'(?<!`)`([^`]+)`(?!`)', r'<code>\1</code>', formatted)

        # Bold: **bold**
        formatted = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted)

        # Italic: *italic* or _italic_
        formatted = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', formatted)
        formatted = re.sub(r'\b_(.*?)_\b', r'<i>\1</i>', formatted)

        # Links: [text](url)
        formatted = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', formatted)

        # Headers: # Header
        formatted = re.sub(r'^#+\s+(.*?)$', r'<b>\1</b>', formatted, flags=re.MULTILINE)

        if not update.message:
            return

        try:
            await update.message.reply_text(formatted, parse_mode="HTML")
        except Exception as e:
            logger.warning("Failed to send HTML formatted message. Falling back to plain text. Error: %s", e)
            await update.message.reply_text(text)

    # ------------------------------------------------------------------
    # Telegram handlers — Commands
    # ------------------------------------------------------------------

    async def _handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /start command."""
        await update.message.reply_text(
            "👋 Hi! I'm your memory bot. Tell me anything and I'll remember it.\n\n"
            "Just chat naturally — I'll remember everything you tell me.\n"
            "You can also send me voice messages and photos!\n\n"
            "Type /help to see all available commands.",
        )

    async def _handle_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /help command."""
        if self._command_handler is None:
            await update.message.reply_text("⚠️ Bot is still starting up...")
            return
        text = await self._command_handler.handle_help()
        await update.message.reply_text(text, parse_mode="HTML")

    async def _handle_facts(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /facts command."""
        if self._command_handler is None:
            await update.message.reply_text("⚠️ Bot is still starting up...")
            return
        user = update.message.from_user
        text = await self._command_handler.handle_facts(
            platform="telegram",
            platform_user_id=str(user.id),
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _handle_search(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /search <query> command."""
        if self._command_handler is None:
            await update.message.reply_text("⚠️ Bot is still starting up...")
            return
        user = update.message.from_user
        query = " ".join(context.args) if context.args else ""
        text = await self._command_handler.handle_search(
            platform="telegram",
            platform_user_id=str(user.id),
            query=query,
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _handle_forget(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /forget <id|all> command."""
        if self._command_handler is None:
            await update.message.reply_text("⚠️ Bot is still starting up...")
            return
        user = update.message.from_user
        arg = " ".join(context.args) if context.args else ""
        text = await self._command_handler.handle_forget(
            platform="telegram",
            platform_user_id=str(user.id),
            arg=arg,
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _handle_model(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /model command."""
        if self._command_handler is None:
            await update.message.reply_text("⚠️ Bot is still starting up...")
            return
        text = await self._command_handler.handle_model()
        await update.message.reply_text(text, parse_mode="HTML")

    async def _handle_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /stats command."""
        if self._command_handler is None:
            await update.message.reply_text("⚠️ Bot is still starting up...")
            return
        user = update.message.from_user
        text = await self._command_handler.handle_stats(
            platform="telegram",
            platform_user_id=str(user.id),
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _handle_export(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /export command to download a ZIP of user data."""
        if self._command_handler is None:
            await update.message.reply_text("⚠️ Bot is still starting up...")
            return

        user = update.message.from_user

        # Send typing/uploading document action
        await update.message.chat.send_action("upload_document")

        data = await self._command_handler.handle_export(
            platform="telegram",
            platform_user_id=str(user.id),
        )

        if isinstance(data, str):
            await update.message.reply_text(data, parse_mode="HTML")
            return

        # Generate files and zip in memory
        import json
        import zipfile
        from datetime import datetime

        # 1. JSON Export
        json_str = json.dumps(data, indent=2, default=str)

        # 2. Markdown Export
        md_lines = [
            f"# Memory Bot Export for {data['user']['display_name'] or 'User'}",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 🧠 Your Facts (Semantic Memory)",
            "Facts are pieces of information the bot extracted and actively uses.",
            ""
        ]

        for f in data.get("facts", []):
            status = "🟢 Active" if f["is_active"] else "🔴 Forgotten"
            md_lines.append(f"- **[{f['id']}]** {f['content']} ({status})")

        md_lines.extend(["", "## 💬 Chat History", ""])

        for c in data.get("conversations", []):
            md_lines.append(f"### Conversation on {c['created_at']}")
            for m in c.get("messages", []):
                role = "👤 You" if m["role"] == "user" else "🤖 Bot"
                # Strip newlines for single-line display or indent for block display
                content_indented = m['content'].replace('\n', '\n  ')
                md_lines.append(f"**{role}**: {content_indented}")
            md_lines.append("")

        md_lines.extend([
            "## 🔗 Episodic Memory Embeddings",
            "To give the bot long-term memory, your messages are converted into mathematical vectors (embeddings) and stored in a database. When you talk to the bot, it searches this vector database for similar past messages to give it 'episodic recall'.",
            "To save space, the raw mathematical vectors have been omitted from this export. Below are the actual text chunks that were embedded and saved:",
            ""
        ])

        for e in data.get("embeddings_history", []):
            md_lines.append(f"- *[{e['created_at']}]*: {e['chunk_text']}")

        md_str = "\n".join(md_lines)

        # 3. Zip file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.json", json_str)
            zf.writestr("readable_export.md", md_str)

        zip_buffer.seek(0)
        zip_buffer.name = f"remember_bot_export_{datetime.now().strftime('%Y%m%d')}.zip"

        await update.message.reply_document(
            document=zip_buffer,
            caption="📦 <b>Export complete!</b> Here is your data in both JSON and readable text formats.",
            parse_mode="HTML"
        )

    async def _handle_connect(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /connect <code> command to link a WhatsApp account."""
        if self._command_handler is None:
            await update.message.reply_text("⚠️ Bot is still starting up...")
            return
        user = update.message.from_user
        code = " ".join(context.args) if context.args else ""
        text = await self._command_handler.handle_connect(
            platform="telegram",
            platform_user_id=str(user.id),
            code=code,
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ------------------------------------------------------------------
    # Telegram handlers — Media Messages
    # ------------------------------------------------------------------

    async def _handle_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming voice/audio messages."""
        if not update.message:
            return

        user = update.message.from_user

        logger.info(
            "Incoming voice message from %s (%s)",
            user.full_name, user.id,
        )

        if self._orchestrator is None:
            await update.message.reply_text("⚠️ Bot is still starting up, please wait...")
            return

        try:
            # Get file info
            voice = update.message.voice or update.message.audio
            if voice is None:
                return

            # Download voice file
            file_bytes = await self._download_file_base64(voice.file_id)

            # Convert OGG to WAV for Gemini API using pydub
            from pydub import AudioSegment
            audio_segment = AudioSegment.from_file(BytesIO(file_bytes))
            wav_io = BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_bytes = wav_io.getvalue()

            media_b64 = base64.b64encode(wav_bytes).decode("utf-8")

            incoming = IncomingMessage(
                platform="telegram",
                platform_user_id=str(user.id),
                platform_chat_id=str(update.message.chat_id),
                display_name=user.full_name,
                text="[Voice message]",
                media_type="voice",
                media_base64=media_b64,
                media_mime="audio/wav",
            )

            response_text = await self._orchestrator.handle_message(incoming)
            await self._safe_reply(update, response_text)

        except Exception:
            logger.exception("Error processing voice from %s", user.id)
            await update.message.reply_text(
                "❌ Sorry, I couldn't process your voice message. Please try again."
            )

    async def _handle_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming photo messages."""
        if not update.message:
            return

        user = update.message.from_user

        logger.info(
            "Incoming photo from %s (%s)",
            user.full_name, user.id,
        )

        if self._orchestrator is None:
            await update.message.reply_text("⚠️ Bot is still starting up, please wait...")
            return

        try:
            # Get the highest resolution photo
            photo = update.message.photo[-1]  # Last element is highest res

            # Download photo
            file_bytes = await self._download_file_base64(photo.file_id)
            media_b64 = base64.b64encode(file_bytes).decode("utf-8")

            # Caption (if any)
            caption = update.message.caption or ""

            incoming = IncomingMessage(
                platform="telegram",
                platform_user_id=str(user.id),
                platform_chat_id=str(update.message.chat_id),
                display_name=user.full_name,
                text=caption if caption else "[Photo sent by user]",
                media_type="photo",
                media_base64=media_b64,
                media_mime="image/jpeg",
            )

            response_text = await self._orchestrator.handle_message(incoming)
            await self._safe_reply(update, response_text)

        except Exception:
            logger.exception("Error processing photo from %s", user.id)
            await update.message.reply_text(
                "❌ Sorry, I couldn't process your photo. Please try again."
            )

    # ------------------------------------------------------------------
    # Telegram handlers — Text Messages
    # ------------------------------------------------------------------

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
            await self._safe_reply(update, response_text)
        except Exception:
            logger.exception("Error processing message from %s", incoming.platform_user_id)
            await update.message.reply_text(
                "❌ Sorry, something went wrong. Please try again."
            )
