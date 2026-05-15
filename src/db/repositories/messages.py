"""
Message & Conversation repository — stores and retrieves chat history.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Conversation, Message


class MessageRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def get_or_create_conversation(
        self, user_id: int, platform: str, platform_chat_id: str
    ) -> Conversation:
        stmt = select(Conversation).where(
            Conversation.platform == platform,
            Conversation.platform_chat_id == platform_chat_id,
        )
        result = await self.session.execute(stmt)
        conv = result.scalar_one_or_none()

        if conv is None:
            conv = Conversation(
                user_id=user_id,
                platform=platform,
                platform_chat_id=platform_chat_id,
            )
            self.session.add(conv)
            await self.session.flush()

        return conv

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def save_message(
        self,
        conversation_id: int,
        user_id: int,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=content,
            metadata_=metadata or {},
        )
        self.session.add(msg)
        await self.session.flush()
        return msg

    async def get_recent_messages(
        self, conversation_id: int, limit: int = 20
    ) -> list[Message]:
        """Return the most recent `limit` messages, ordered oldest-first."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        messages = list(result.scalars().all())
        messages.reverse()  # oldest first
        return messages

    async def count_messages(self, conversation_id: int) -> int:
        from sqlalchemy import func
        stmt = select(func.count(Message.id)).where(
            Message.conversation_id == conversation_id
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def get_messages_range(
        self,
        conversation_id: int,
        after_id: int | None = None,
        limit: int = 50,
    ) -> list[Message]:
        """
        Get messages in a conversation, optionally starting after a given message ID.
        Returns oldest-first, used by the summarizer to grab un-summarized chunks.
        """
        stmt = select(Message).where(
            Message.conversation_id == conversation_id
        )
        if after_id is not None:
            stmt = stmt.where(Message.id > after_id)
        stmt = stmt.order_by(Message.created_at.asc()).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_oldest_unsummarized_messages(
        self,
        conversation_id: int,
        last_summarized_message_id: int | None,
        limit: int = 50,
    ) -> list[Message]:
        """
        Get messages that haven't been summarized yet, oldest first.
        Used to determine if summarization threshold has been reached.
        """
        stmt = select(Message).where(
            Message.conversation_id == conversation_id
        )
        if last_summarized_message_id is not None:
            stmt = stmt.where(Message.id > last_summarized_message_id)
        stmt = stmt.order_by(Message.created_at.asc()).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

