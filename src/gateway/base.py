"""
Abstract gateway interface — defines the contract for messaging platform integrations.
Telegram implements this now; WhatsApp will implement it in Phase 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class IncomingMessage:
    """Normalized incoming message from any platform."""
    platform: str                # 'telegram' | 'whatsapp'
    platform_user_id: str        # Telegram user ID or WhatsApp phone
    platform_chat_id: str        # Chat/conversation ID on the platform
    display_name: str | None     # User's display name (if available)
    text: str                    # Message text content


class BaseGateway(ABC):
    """Abstract base for messaging platform gateways."""

    @abstractmethod
    async def setup(self, app) -> None:
        """Register routes/webhooks with the FastAPI app."""
        ...

    @abstractmethod
    async def send_message(self, chat_id: str, text: str) -> None:
        """Send a text message to a chat."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the gateway (e.g., set webhook URL)."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the gateway."""
        ...
