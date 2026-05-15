"""
Memory Decay — periodically reduces relevance scores of old facts.

Implements a configurable half-life decay model:
  - Every `decay_interval_hours`, facts older than `min_age_hours` have their
    relevance score reduced by `decay_factor`.
  - Facts accessed (referenced in context) get their score boosted.
  - Facts below `min_relevance` threshold are deactivated (soft-deleted).

This keeps the fact store lean and ensures recently relevant information
is prioritized in context assembly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Fact

logger = logging.getLogger(__name__)


class MemoryDecay:
    """Manages relevance score decay for stored facts."""

    def __init__(
        self,
        decay_factor: float = 0.95,
        min_relevance: float = 0.1,
        min_age_hours: int = 24,
    ):
        """
        Args:
            decay_factor: Multiply relevance by this each cycle (0.95 = 5% decay)
            min_relevance: Deactivate facts below this threshold
            min_age_hours: Only decay facts older than this many hours
        """
        self.decay_factor = decay_factor
        self.min_relevance = min_relevance
        self.min_age_hours = min_age_hours

    async def run_decay_cycle(self, session: AsyncSession) -> dict:
        """
        Run one decay cycle across all users' facts.
        Returns stats: {decayed: int, deactivated: int}
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.min_age_hours)

        # 1. Decay relevance scores for old active facts
        decay_stmt = (
            update(Fact)
            .where(
                Fact.is_active == True,
                Fact.created_at < cutoff,
            )
            .values(
                relevance_score=Fact.relevance_score * self.decay_factor,
                updated_at=datetime.now(timezone.utc),
            )
            .execution_options(synchronize_session="fetch")
        )
        decay_result = await session.execute(decay_stmt)
        decayed_count = decay_result.rowcount

        # 2. Deactivate facts that have decayed below the minimum threshold
        deactivate_stmt = (
            update(Fact)
            .where(
                Fact.is_active == True,
                Fact.relevance_score < self.min_relevance,
            )
            .values(
                is_active=False,
                updated_at=datetime.now(timezone.utc),
            )
            .execution_options(synchronize_session="fetch")
        )
        deactivate_result = await session.execute(deactivate_stmt)
        deactivated_count = deactivate_result.rowcount

        await session.flush()

        stats = {
            "decayed": decayed_count,
            "deactivated": deactivated_count,
        }

        if decayed_count > 0 or deactivated_count > 0:
            logger.info(
                "Memory decay cycle: decayed=%d, deactivated=%d",
                decayed_count, deactivated_count,
            )

        return stats

    async def boost_fact(
        self,
        session: AsyncSession,
        fact_id: int,
        boost_amount: float = 0.1,
    ) -> None:
        """
        Boost a fact's relevance score (e.g. when it's used in context).
        Caps at 1.0.
        """
        stmt = select(Fact).where(Fact.id == fact_id)
        result = await session.execute(stmt)
        fact = result.scalar_one_or_none()

        if fact and fact.is_active:
            new_score = min(1.0, fact.relevance_score + boost_amount)
            fact.relevance_score = new_score
            fact.updated_at = datetime.now(timezone.utc)
            await session.flush()
