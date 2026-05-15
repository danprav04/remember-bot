"""
User repository — create/find users by platform identity.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import User


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create(
        self, platform: str, platform_user_id: str, display_name: str | None = None
    ) -> User:
        """Find an existing user or create a new one. Returns the User row."""
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

        return user

    async def get_by_id(self, user_id: int) -> User | None:
        stmt = select(User).where(User.id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
