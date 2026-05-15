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
from src.llm.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class SemanticMemory:
    """Retrieves the most relevant stored facts for a user."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    async def recall(
        self,
        session: AsyncSession,
        user_id: int,
        query_text: str | None = None,
        limit: int = 20,
    ) -> list[str]:
        """
        Get the most relevant active facts for a user as plain-text strings.
        If query_text is provided, uses vector search to find exact semantic matches.
        Otherwise, falls back to most relevant recent facts.
        """
        try:
            repo = FactRepository(session)
            
            if query_text:
                try:
                    query_embedding = await self.embedding_service.embed(query_text)
                    if query_embedding:
                        facts = await repo.search_facts_by_similarity(
                            user_id=user_id,
                            query_embedding=query_embedding,
                            limit=limit,
                        )
                        if facts:
                            return [f.content for f in facts]
                except Exception as e:
                    logger.warning("Fact vector search failed for query '%s': %s", query_text, e)
            
            # Fallback if no query, or if embedding/search failed, or if no facts had embeddings
            facts = await repo.get_active_facts(user_id=user_id, limit=limit)
            return [f.content for f in facts]
        except Exception:
            logger.exception("Semantic recall failed for user %d", user_id)
            return []
