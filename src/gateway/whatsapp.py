"""
WhatsApp gateway — handles incoming messages via the WhatsApp Cloud API
using PyWa (async) with FastAPI webhook integration.

Supports text, voice/audio, and image messages. Converts incoming messages
to the shared IncomingMessage format and delegates to the orchestrator.
"""

from __future__ import annotations

import base64
import logging
import re
from io import BytesIO
from typing import TYPE_CHECKING

from pywa_async import WhatsApp
from pywa.types import Message as WAMessage

from src.gateway.base import BaseGateway, IncomingMessage

if TYPE_CHECKING:
    from src.core.commands import CommandHandler as BotCommandHandler
    from src.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class WhatsAppGateway(BaseGateway):
    """WhatsApp Cloud API integration via PyWa webhooks."""

    def __init__(
        self,
        phone_id: str,
        token: str,
        verify_token: str,
        app_id: int,
        app_secret: str,
        webhook_base_url: str,
    ):
        self.phone_id = phone_id
        self.token = token
        self.verify_token = verify_token
        self.app_id = app_id
        self.app_secret = app_secret
        self.webhook_base_url = webhook_base_url
        self.callback_url = f"{webhook_base_url}/webhook/whatsapp"
        if not self.callback_url.startswith("http"):
            self.callback_url = f"https://{self.callback_url}"

        self._orchestrator: Orchestrator | None = None
        self._command_handler: BotCommandHandler | None = None
        self._wa: WhatsApp | None = None

    def set_orchestrator(self, orchestrator: Orchestrator) -> None:
        """Inject the orchestrator dependency."""
        self._orchestrator = orchestrator

    def set_command_handler(self, handler: BotCommandHandler) -> None:
        """Inject the command handler dependency."""
        self._command_handler = handler

    async def setup(self, app) -> None:
        """Create the PyWa WhatsApp client and register handlers."""

        self._wa = WhatsApp(
            phone_id=self.phone_id,
            token=self.token,
            server=app,
            callback_url=self.callback_url,
            verify_token=self.verify_token,
            app_id=self.app_id,
            app_secret=self.app_secret,
            webhook_challenge_delay=15,  # Give FastAPI time to start before Meta verifies
        )

        # Register the message handler on the PyWa instance
        @self._wa.on_message()
        async def on_message(client: WhatsApp, msg: WAMessage):
            await self._route_message(client, msg)

        logger.info(
            "WhatsApp webhook route registered, callback URL: %s",
            self.callback_url,
        )

    async def start(self) -> None:
        """WhatsApp webhooks are configured in the Meta dashboard, not via API.
        Nothing to do here — the webhook route is already live from setup()."""
        logger.info("WhatsApp gateway started (webhook at /webhook/whatsapp)")

    async def stop(self) -> None:
        """Graceful shutdown — nothing to tear down for WhatsApp."""
        logger.info("WhatsApp gateway stopped")

    async def send_message(self, chat_id: str, text: str) -> None:
        """Send a text message to a WhatsApp user by phone number."""
        if self._wa is None:
            raise RuntimeError("Gateway not set up")
        await self._wa.send_message(to=chat_id, text=text)

    # ------------------------------------------------------------------
    # Internal — Route incoming messages
    # ------------------------------------------------------------------

    async def _route_message(
        self, client: WhatsApp, msg: WAMessage
    ) -> None:
        """Main entry point for all incoming WhatsApp messages."""
        try:
            sender = msg.from_user.wa_id  # phone number like "972501234567"
            sender_name = msg.from_user.name

            # Check for command messages (starting with /)
            if msg.text and msg.text.startswith("/"):
                await self._handle_command(client, msg, sender, sender_name)
                return

            # Route by message type
            if msg.has_media:
                if msg.type.value == "image":
                    await self._handle_image(client, msg, sender, sender_name)
                elif msg.type.value in ("audio", "voice"):
                    await self._handle_audio(client, msg, sender, sender_name)
                else:
                    # Unsupported media — treat caption/text as a text message
                    await self._handle_text(client, msg, sender, sender_name)
            else:
                await self._handle_text(client, msg, sender, sender_name)

        except Exception:
            logger.exception("Error processing WhatsApp message from %s", msg.from_user.wa_id)
            try:
                await client.send_message(
                    to=msg.from_user.wa_id,
                    text="❌ Sorry, something went wrong. Please try again.",
                )
            except Exception:
                logger.exception("Failed to send error message")

    # ------------------------------------------------------------------
    # Text messages
    # ------------------------------------------------------------------

    async def _handle_text(
        self,
        client: WhatsApp,
        msg: WAMessage,
        sender: str,
        sender_name: str,
    ) -> None:
        """Handle a plain text message."""
        text = msg.text or ""
        if not text.strip():
            return

        logger.info(
            "Incoming WhatsApp text from %s (%s): %s",
            sender_name, sender, text[:100],
        )

        if self._orchestrator is None:
            await client.send_message(to=sender, text="⚠️ Bot is still starting up, please wait...")
            return

        incoming = IncomingMessage(
            platform="whatsapp",
            platform_user_id=sender,
            platform_chat_id=sender,  # In WhatsApp, chat ID = phone number for 1:1
            display_name=sender_name,
            text=text,
        )

        response_text = await self._orchestrator.handle_message(incoming)
        await client.send_message(to=sender, text=response_text)

    # ------------------------------------------------------------------
    # Image messages
    # ------------------------------------------------------------------

    async def _handle_image(
        self,
        client: WhatsApp,
        msg: WAMessage,
        sender: str,
        sender_name: str,
    ) -> None:
        """Handle an image message."""
        logger.info("Incoming WhatsApp image from %s (%s)", sender_name, sender)

        if self._orchestrator is None:
            await client.send_message(to=sender, text="⚠️ Bot is still starting up, please wait...")
            return

        try:
            image_bytes = await msg.image.download(in_memory=True)
            media_b64 = base64.b64encode(image_bytes).decode("utf-8")
            caption = msg.caption or ""

            incoming = IncomingMessage(
                platform="whatsapp",
                platform_user_id=sender,
                platform_chat_id=sender,
                display_name=sender_name,
                text=caption if caption else "[Photo sent by user]",
                media_type="photo",
                media_base64=media_b64,
                media_mime=msg.image.mime_type or "image/jpeg",
            )

            response_text = await self._orchestrator.handle_message(incoming)
            await client.send_message(to=sender, text=response_text)

        except Exception:
            logger.exception("Error processing WhatsApp image from %s", sender)
            await client.send_message(
                to=sender,
                text="❌ Sorry, I couldn't process your image. Please try again.",
            )

    # ------------------------------------------------------------------
    # Audio / Voice messages
    # ------------------------------------------------------------------

    async def _handle_audio(
        self,
        client: WhatsApp,
        msg: WAMessage,
        sender: str,
        sender_name: str,
    ) -> None:
        """Handle an audio or voice message."""
        logger.info("Incoming WhatsApp voice/audio from %s (%s)", sender_name, sender)

        if self._orchestrator is None:
            await client.send_message(to=sender, text="⚠️ Bot is still starting up, please wait...")
            return

        try:
            media_obj = msg.audio or msg.voice
            if media_obj is None:
                return

            audio_bytes = await media_obj.download(in_memory=True)

            # Convert to WAV for the transcription pipeline (same as Telegram)
            from pydub import AudioSegment

            audio_segment = AudioSegment.from_file(BytesIO(audio_bytes))
            wav_io = BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_bytes = wav_io.getvalue()

            media_b64 = base64.b64encode(wav_bytes).decode("utf-8")

            incoming = IncomingMessage(
                platform="whatsapp",
                platform_user_id=sender,
                platform_chat_id=sender,
                display_name=sender_name,
                text="[Voice message]",
                media_type="voice",
                media_base64=media_b64,
                media_mime="audio/wav",
            )

            response_text = await self._orchestrator.handle_message(incoming)
            await client.send_message(to=sender, text=response_text)

        except Exception:
            logger.exception("Error processing WhatsApp audio from %s", sender)
            await client.send_message(
                to=sender,
                text="❌ Sorry, I couldn't process your voice message. Please try again.",
            )

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------

    async def _handle_command(
        self,
        client: WhatsApp,
        msg: WAMessage,
        sender: str,
        sender_name: str,
    ) -> None:
        """Parse slash commands and route to the command handler."""
        parts = msg.text.strip().split(maxsplit=1)
        command = parts[0][1:].lower()  # Strip the leading /
        args = parts[1] if len(parts) > 1 else ""

        if self._command_handler is None:
            await client.send_message(to=sender, text="⚠️ Bot is still starting up...")
            return

        response: str | None = None

        if command == "start":
            response = (
                "👋 Hi! I'm your memory bot. Tell me anything and I'll remember it.\n\n"
                "Just chat naturally — I'll remember everything you tell me.\n"
                "You can also send me voice messages and photos!\n\n"
                "Type /help to see all available commands."
            )
        elif command == "help":
            response = await self._command_handler.handle_help()
        elif command == "facts":
            response = await self._command_handler.handle_facts(
                platform="whatsapp", platform_user_id=sender,
            )
        elif command == "search":
            response = await self._command_handler.handle_search(
                platform="whatsapp", platform_user_id=sender, query=args,
            )
        elif command == "forget":
            response = await self._command_handler.handle_forget(
                platform="whatsapp", platform_user_id=sender, arg=args,
            )
        elif command == "model":
            response = await self._command_handler.handle_model()
        elif command == "stats":
            response = await self._command_handler.handle_stats(
                platform="whatsapp", platform_user_id=sender,
            )
        elif command == "link":
            response = await self._command_handler.handle_link(
                platform="whatsapp", platform_user_id=sender,
            )
        else:
            # Unknown command — treat as regular message
            await self._handle_text(client, msg, sender, sender_name)
            return

        if response:
            # Convert HTML formatting to WhatsApp-compatible formatting
            response = self._html_to_whatsapp(response)
            await client.send_message(to=sender, text=response)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _html_to_whatsapp(text: str) -> str:
        """Convert simple HTML tags (used by the command handler) to
        WhatsApp-compatible formatting."""
        # Bold: <b>text</b> → *text*
        text = re.sub(r"<b>(.*?)</b>", r"*\1*", text, flags=re.DOTALL)
        # Italic: <i>text</i> → _text_
        text = re.sub(r"<i>(.*?)</i>", r"_\1_", text, flags=re.DOTALL)
        # Code: <code>text</code> → `text`
        text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)
        # HTML entities
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&amp;", "&")
        return text
