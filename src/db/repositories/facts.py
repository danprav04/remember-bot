"""
Facts repository — CRUD for the dynamic semantic memory (facts table).
Supports superseding: when a fact is updated, the old one is marked inactive
and linked to the new version.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Fact


class FactRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_fact(
        self,
        user_id: int,
        content: str,
        tags: list[str] | None = None,
        relevance_score: float = 1.0,
        source_message_id: int | None = None,
        embedding: list[float] | None = None,
    ) -> Fact:
        """Store a new fact for a user."""
        fact = Fact(
            user_id=user_id,
            content=content,
            tags=tags or [],
            embedding=embedding,
            relevance_score=relevance_score,
            source_message_id=source_message_id,
            is_active=True,
        )
        self.session.add(fact)
        await self.session.flush()
        return fact

    async def supersede_fact(
        self,
        old_fact_id: int,
        user_id: int,
        new_content: str,
        tags: list[str] | None = None,
        relevance_score: float = 1.0,
        source_message_id: int | None = None,
        embedding: list[float] | None = None,
    ) -> Fact:
        """
        Mark an existing fact as superseded and create the replacement.
        The old fact is kept for history but marked inactive.
        """
        # Deactivate the old fact
        old_fact = await self.get_fact_by_id(old_fact_id, user_id)
        if old_fact:
            old_fact.is_active = False

        # Create the new version
        new_fact = await self.create_fact(
            user_id=user_id,
            content=new_content,
            tags=tags,
            relevance_score=relevance_score,
            source_message_id=source_message_id,
            embedding=embedding,
        )

        # Link old → new
        if old_fact:
            old_fact.superseded_by = new_fact.id
            old_fact.updated_at = datetime.now(timezone.utc)

        await self.session.flush()
        return new_fact

    async def get_active_facts(
        self,
        user_id: int,
        limit: int = 50,
    ) -> list[Fact]:
        """Get all active (non-superseded) facts for a user, most relevant first."""
        stmt = (
            select(Fact)
            .where(Fact.user_id == user_id, Fact.is_active == True)
            .order_by(Fact.relevance_score.desc(), Fact.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def search_facts_by_similarity(
        self,
        user_id: int,
        query_embedding: list[float],
        limit: int = 20,
    ) -> list[Fact]:
        """
        Find active facts that are semantically similar to the query.
        Uses cosine distance (<=> operator in pgvector).
        """
        stmt = (
            select(Fact)
            .where(Fact.user_id == user_id, Fact.is_active == True, Fact.embedding.is_not(None))
            .order_by(Fact.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def search_facts_by_tags(
        self,
        user_id: int,
        tags: list[str],
        limit: int = 20,
    ) -> list[Fact]:
        """Find active facts that overlap with any of the given tags."""
        stmt = (
            select(Fact)
            .where(
                Fact.user_id == user_id,
                Fact.is_active == True,
                Fact.tags.overlap(tags),
            )
            .order_by(Fact.relevance_score.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_fact_by_id(self, fact_id: int, user_id: int) -> Fact | None:
        """Get a specific fact (ensures user scoping)."""
        stmt = select(Fact).where(Fact.id == fact_id, Fact.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def count_active_facts(self, user_id: int) -> int:
        """Count total active facts for a user."""
        stmt = select(func.count(Fact.id)).where(
            Fact.user_id == user_id, Fact.is_active == True
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def deactivate_fact(self, fact_id: int, user_id: int) -> bool:
        """Deactivate (soft-delete) a fact. Returns True if found and deactivated."""
        fact = await self.get_fact_by_id(fact_id, user_id)
        if fact and fact.is_active:
            fact.is_active = False
            fact.updated_at = datetime.now(timezone.utc)
            await self.session.flush()
            return True
        return False

    async def deactivate_all_facts(self, user_id: int) -> int:
        """Deactivate all active facts for a user. Returns count deactivated."""
        facts = await self.get_active_facts(user_id=user_id, limit=1000)
        count = 0
        for fact in facts:
            fact.is_active = False
            fact.updated_at = datetime.now(timezone.utc)
            count += 1
        if count:
            await self.session.flush()
        return count

    async def search_facts_by_text(
        self,
        user_id: int,
        query: str,
        limit: int = 10,
    ) -> list[Fact]:
        """Search active facts by content text (case-insensitive ILIKE)."""
        stmt = (
            select(Fact)
            .where(
                Fact.user_id == user_id,
                Fact.is_active == True,
                Fact.content.ilike(f"%{query}%"),
            )
            .order_by(Fact.relevance_score.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

