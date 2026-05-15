"""
Semantic memory — retrieves stored facts about a user for context assembly.

Facts are created by the FactExtractor (async background task) and stored
in the `facts` table. This module provides the read-side interface used
by the ContextAssembler.
"""

from __future__ import annotations

import logging
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.repositories.facts import FactRepository

logger = logging.getLogger(__name__)


class SemanticMemory:
    """Retrieves the most relevant stored facts for a user."""

    async def recall(
        self,
        session: AsyncSession,
        user_id: int,
        limit: int = 20,
    ) -> list[str]:
        """
        Get the most relevant active facts for a user as plain-text strings.
        Returns facts ordered by relevance score (highest first).
        """
        try:
            repo = FactRepository(session)
            facts = await repo.get_active_facts(user_id=user_id, limit=limit)
            return [f.content for f in facts]
        except Exception:
            logger.exception("Semantic recall failed for user %d", user_id)
            return []
