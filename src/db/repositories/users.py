"""
User repository — create/find users by platform identity.
Supports cross-platform user linking via the `linked_to` column.
"""

from __future__ import annotations

import logging

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    Conversation,
    ConversationSummary,
    Fact,
    Message,
    MessageEmbedding,
    User,
)

logger = logging.getLogger(__name__)


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create(
        self, platform: str, platform_user_id: str, display_name: str | None = None
    ) -> User:
        """Find an existing user or create a new one.

        If the user has a ``linked_to`` reference (cross-platform link),
        the **primary** user is returned instead so that all downstream
        operations (memory, facts, etc.) are scoped under one identity.
        """
        stmt = select(User).where(
            User.platform == platform,
            User.platform_user_id == platform_user_id,
        )
        result = await self.session.execute(stmt)
        user = result.scalar_one_or_none()

        if user is None:
            user = User(
                platform=platform,
                platform_user_id=platform_user_id,
                display_name=display_name,
            )
            self.session.add(user)
            await self.session.flush()

        elif display_name and user.display_name != display_name:
            user.display_name = display_name
            await self.session.flush()

        # Follow cross-platform link
        if user.linked_to is not None:
            primary = await self.get_by_id(user.linked_to)
            if primary is not None:
                return primary

        return user

    async def get_by_id(self, user_id: int) -> User | None:
        stmt = select(User).where(User.id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_platform(
        self, platform: str, platform_user_id: str
    ) -> User | None:
        """Find a user by platform identity without creating one."""
        stmt = select(User).where(
            User.platform == platform,
            User.platform_user_id == platform_user_id,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def merge_users(self, primary_id: int, secondary_id: int) -> None:
        """Migrate ALL data from secondary user to primary user.

        After this, the secondary user's ``linked_to`` points to the primary
        and all historical data is unified under one user ID.
        """
        logger.info(
            "Merging user %d into primary user %d", secondary_id, primary_id
        )

        # Migrate messages
        await self.session.execute(
            update(Message)
            .where(Message.user_id == secondary_id)
            .values(user_id=primary_id)
        )

        # Migrate facts
        await self.session.execute(
            update(Fact)
            .where(Fact.user_id == secondary_id)
            .values(user_id=primary_id)
        )

        # Migrate embeddings
        await self.session.execute(
            update(MessageEmbedding)
            .where(MessageEmbedding.user_id == secondary_id)
            .values(user_id=primary_id)
        )

        # Migrate conversation summaries
        await self.session.execute(
            update(ConversationSummary)
            .where(ConversationSummary.user_id == secondary_id)
            .values(user_id=primary_id)
        )

        # Migrate conversations
        await self.session.execute(
            update(Conversation)
            .where(Conversation.user_id == secondary_id)
            .values(user_id=primary_id)
        )

        # Set the link
        await self.session.execute(
            update(User)
            .where(User.id == secondary_id)
            .values(linked_to=primary_id)
        )

        await self.session.flush()
        logger.info("User merge complete: %d → %d", secondary_id, primary_id)
