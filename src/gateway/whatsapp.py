"""
WhatsApp gateway — handles incoming messages via the WhatsApp Cloud API.

Uses manual FastAPI route registration + httpx for API calls instead of
PyWa's server integration (which has route registration timing issues
with FastAPI's lifespan).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
from io import BytesIO
from typing import TYPE_CHECKING

import httpx
from fastapi import Request
from fastapi.responses import PlainTextResponse, Response

from src.gateway.base import BaseGateway, IncomingMessage

if TYPE_CHECKING:
    from src.core.commands import CommandHandler as BotCommandHandler
    from src.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v21.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


class WhatsAppGateway(BaseGateway):
    """WhatsApp Cloud API integration via manual webhook routes."""

    def __init__(
        self,
        phone_id: str,
        token: str,
        verify_token: str,
        app_id: int,
        app_secret: str,
    ):
        self.phone_id = phone_id
        self.token = token
        self.verify_token = verify_token
        self.app_id = app_id
        self.app_secret = app_secret

        self._orchestrator: Orchestrator | None = None
        self._command_handler: BotCommandHandler | None = None
        self._http: httpx.AsyncClient | None = None

    def set_orchestrator(self, orchestrator: Orchestrator) -> None:
        self._orchestrator = orchestrator

    def set_command_handler(self, handler: BotCommandHandler) -> None:
        self._command_handler = handler

    async def setup(self, app) -> None:
        """Register webhook routes and create the HTTP client."""
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30.0,
        )

        # Register webhook routes manually on the FastAPI app
        app.add_api_route(
            "/webhook/whatsapp",
            self._verify_webhook,
            methods=["GET"],
        )
        app.add_api_route(
            "/webhook/whatsapp",
            self._handle_webhook,
            methods=["POST"],
        )

        logger.info("WhatsApp webhook routes registered at /webhook/whatsapp")

    async def start(self) -> None:
        logger.info("WhatsApp gateway started (webhook at /webhook/whatsapp)")

    async def stop(self) -> None:
        if self._http:
            await self._http.aclose()
        logger.info("WhatsApp gateway stopped")

    async def send_message(self, chat_id: str, text: str) -> None:
        await self._send_text(chat_id, text)

    # ------------------------------------------------------------------
    # Webhook endpoints
    # ------------------------------------------------------------------

    async def _verify_webhook(self, request: Request) -> Response:
        """Handle Meta's webhook verification challenge (GET)."""
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")

        if mode == "subscribe" and token == self.verify_token:
            logger.info("WhatsApp webhook verified successfully")
            return PlainTextResponse(content=challenge)

        logger.warning("WhatsApp webhook verification failed (bad token)")
        return PlainTextResponse(content="Forbidden", status_code=403)

    async def _handle_webhook(self, request: Request) -> Response:
        """Handle incoming WhatsApp messages (POST)."""
        body = await request.body()

        # Validate signature
        signature = request.headers.get("X-Hub-Signature-256", "")
        if self.app_secret and not self._validate_signature(body, signature):
            logger.warning("Invalid webhook signature — rejecting")
            return Response(status_code=403)

        data = await request.json()

        # WhatsApp sends a 'statuses' field for delivery receipts — ignore
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                contacts = value.get("contacts", [])

                for msg in messages:
                    sender = msg.get("from", "")
                    sender_name = sender
                    if contacts:
                        sender_name = contacts[0].get("profile", {}).get("name", sender)

                    try:
                        await self._route_message(msg, sender, sender_name)
                    except Exception:
                        logger.exception("Error processing WhatsApp message from %s", sender)
                        try:
                            await self._send_text(
                                sender,
                                "❌ Sorry, something went wrong. Please try again.",
                            )
                        except Exception:
                            logger.exception("Failed to send error message")

        return Response(status_code=200)

    # ------------------------------------------------------------------
    # Signature validation
    # ------------------------------------------------------------------

    def _validate_signature(self, payload: bytes, signature: str) -> bool:
        if not signature:
            return False
        expected = hmac.new(
            self.app_secret.encode() if isinstance(self.app_secret, str) else self.app_secret,
            payload,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", signature)

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    async def _route_message(
        self, msg: dict, sender: str, sender_name: str
    ) -> None:
        msg_type = msg.get("type", "")

        # Check for commands
        if msg_type == "text":
            text = msg.get("text", {}).get("body", "")
            if text.startswith("/"):
                await self._handle_command(text, sender, sender_name)
                return
            await self._handle_text(text, sender, sender_name)
        elif msg_type == "image":
            await self._handle_image(msg, sender, sender_name)
        elif msg_type in ("audio", "voice"):
            await self._handle_audio(msg, sender, sender_name)
        elif msg_type == "document":
            await self._handle_document(msg, sender, sender_name)
        else:
            # Unsupported type — try to get any text
            text = msg.get("text", {}).get("body", "")
            if text:
                await self._handle_text(text, sender, sender_name)

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    async def _handle_text(self, text: str, sender: str, sender_name: str) -> None:
        if not text.strip():
            return

        logger.info("Incoming WhatsApp text from %s (%s): %s", sender_name, sender, text[:100])

        if self._orchestrator is None:
            await self._send_text(sender, "⚠️ Bot is still starting up, please wait...")
            return

        incoming = IncomingMessage(
            platform="whatsapp",
            platform_user_id=sender,
            platform_chat_id=sender,
            display_name=sender_name,
            text=text,
        )

        import asyncio
        asyncio.create_task(self._send_typing_indicator(sender))
        response = await self._orchestrator.handle_message(incoming)
        response = self._markdown_to_whatsapp(response)
        await self._send_text(sender, response)

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------

    async def _handle_image(self, msg: dict, sender: str, sender_name: str) -> None:
        logger.info("Incoming WhatsApp image from %s (%s)", sender_name, sender)

        if self._orchestrator is None:
            await self._send_text(sender, "⚠️ Bot is still starting up, please wait...")
            return

        try:
            image_data = msg.get("image", {})
            media_id = image_data.get("id", "")
            mime_type = image_data.get("mime_type", "image/jpeg")
            caption = image_data.get("caption", "")

            image_bytes = await self._download_media(media_id)
            media_b64 = base64.b64encode(image_bytes).decode("utf-8")

            incoming = IncomingMessage(
                platform="whatsapp",
                platform_user_id=sender,
                platform_chat_id=sender,
                display_name=sender_name,
                text=caption if caption else "[Photo sent by user]",
                media_type="photo",
                media_base64=media_b64,
                media_mime=mime_type,
            )

            import asyncio
            asyncio.create_task(self._send_typing_indicator(sender))
            response = await self._orchestrator.handle_message(incoming)
            response = self._markdown_to_whatsapp(response)
            await self._send_text(sender, response)
        except Exception:
            logger.exception("Error processing WhatsApp image from %s", sender)
            await self._send_text(sender, "❌ Sorry, I couldn't process your image.")

    # ------------------------------------------------------------------
    # Audio / Voice
    # ------------------------------------------------------------------

    async def _handle_audio(self, msg: dict, sender: str, sender_name: str) -> None:
        logger.info("Incoming WhatsApp voice/audio from %s (%s)", sender_name, sender)

        if self._orchestrator is None:
            await self._send_text(sender, "⚠️ Bot is still starting up, please wait...")
            return

        try:
            audio_data = msg.get("audio") or msg.get("voice") or {}
            media_id = audio_data.get("id", "")

            audio_bytes = await self._download_media(media_id)

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

            import asyncio
            asyncio.create_task(self._send_typing_indicator(sender))
            response = await self._orchestrator.handle_message(incoming)
            response = self._markdown_to_whatsapp(response)
            await self._send_text(sender, response)
        except Exception:
            logger.exception("Error processing WhatsApp audio from %s", sender)
            await self._send_text(sender, "❌ Sorry, I couldn't process your voice message.")

    # ------------------------------------------------------------------
    # Document
    # ------------------------------------------------------------------

    async def _handle_document(self, msg: dict, sender: str, sender_name: str) -> None:
        logger.info("Incoming WhatsApp document from %s (%s)", sender_name, sender)

        if self._orchestrator is None:
            await self._send_text(sender, "⚠️ Bot is still starting up, please wait...")
            return

        try:
            doc_data = msg.get("document", {})
            media_id = doc_data.get("id", "")
            filename = doc_data.get("filename", "unknown_file")
            caption = doc_data.get("caption", "")

            # Check supported extensions
            import os
            ext = os.path.splitext(filename)[1].lower()
            supported = {".pdf", ".docx", ".doc", ".md", ".txt", ".text"}
            if ext not in supported:
                await self._send_text(
                    sender,
                    f"❌ Unsupported file type: {ext}. "
                    f"I can process: PDF, DOCX, Markdown (.md), and text (.txt) files."
                )
                return

            # Download file bytes
            file_bytes = await self._download_media(media_id)

            incoming = IncomingMessage(
                platform="whatsapp",
                platform_user_id=sender,
                platform_chat_id=sender,
                display_name=sender_name,
                text=caption if caption else f"[Document: {filename}]",
                media_type="document",
                document_bytes=file_bytes,
                document_filename=filename,
            )

            import asyncio
            asyncio.create_task(self._send_typing_indicator(sender))
            response = await self._orchestrator.handle_message(incoming)
            response = self._markdown_to_whatsapp(response)
            await self._send_text(sender, response)
        except Exception:
            logger.exception("Error processing WhatsApp document from %s", sender)
            await self._send_text(sender, "❌ Sorry, I couldn't process your document.")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    async def _handle_command(self, text: str, sender: str, sender_name: str) -> None:
        parts = text.strip().split(maxsplit=1)
        command = parts[0][1:].lower()
        args = parts[1] if len(parts) > 1 else ""

        if self._command_handler is None:
            await self._send_text(sender, "⚠️ Bot is still starting up...")
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
        elif command == "documents":
            response = await self._command_handler.handle_documents(
                platform="whatsapp", platform_user_id=sender,
            )
        else:
            await self._handle_text(text, sender, sender_name)
            return

        if response:
            response = self._html_to_whatsapp(response)
            await self._send_text(sender, response)

    # ------------------------------------------------------------------
    # WhatsApp Cloud API calls
    # ------------------------------------------------------------------

    async def _send_text(self, to: str, text: str) -> None:
        """Send a text message via WhatsApp Cloud API."""
        if self._http is None:
            raise RuntimeError("Gateway not set up")

        url = f"{GRAPH_API_BASE}/{self.phone_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": text},
        }
        resp = await self._http.post(url, json=payload)
        if resp.status_code != 200:
            logger.error("Failed to send WhatsApp message: %s %s", resp.status_code, resp.text)

    async def _send_typing_indicator(self, to: str) -> None:
        """Send a typing indicator (action: typing_on)."""
        if self._http is None:
            return

        url = f"{GRAPH_API_BASE}/{self.phone_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "typing_indicator",
            "typing_indicator": {
                "action": "typing_on"
            }
        }
        resp = await self._http.post(url, json=payload)
        if resp.status_code != 200:
            logger.debug("Typing indicator not supported or failed: %s", resp.text)

    async def _download_media(self, media_id: str) -> bytes:
        """Download media from WhatsApp (two-step: get URL, then download)."""
        if self._http is None:
            raise RuntimeError("Gateway not set up")

        # Step 1: Get the media URL
        resp = await self._http.get(f"{GRAPH_API_BASE}/{media_id}")
        resp.raise_for_status()
        media_url = resp.json()["url"]

        # Step 2: Download the actual bytes
        resp = await self._http.get(media_url)
        resp.raise_for_status()
        return resp.content

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _html_to_whatsapp(text: str) -> str:
        """Convert HTML tags to WhatsApp formatting."""
        text = re.sub(r"<b>(.*?)</b>", r"*\1*", text, flags=re.DOTALL)
        text = re.sub(r"<i>(.*?)</i>", r"_\1_", text, flags=re.DOTALL)
        text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&amp;", "&")
        return text

    @staticmethod
    def _markdown_to_whatsapp(text: str) -> str:
        """Convert standard Markdown to WhatsApp formatting (best effort fallback)."""
        # Convert **bold** to *bold*
        text = re.sub(r"\*\*(.*?)\*\*", r"*\1*", text)
        # Convert ### headers to *bold*
        text = re.sub(r"^#+\s+(.*?)$", r"*\1*", text, flags=re.MULTILINE)
        return text
