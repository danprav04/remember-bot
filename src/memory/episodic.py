"""
Episodic memory — embeds messages and retrieves relevant past context
using vector similarity search (pgvector).

Every user message is embedded asynchronously after the response is sent.
At retrieval time, the user's current message is embedded and compared
against their stored embeddings to find the most relevant past exchanges.
"""

from __future__ import annotations

import logging
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.repositories.embeddings import EmbeddingRepository
from src.llm.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """Handles embedding storage and retrieval for episodic recall."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    async def embed_message(
        self,
        session: AsyncSession,
        message_id: int,
        user_id: int,
        text: str,
    ) -> None:
        """
        Embed a single message and store it.
        Skips if the embedding service is unavailable or if already embedded.
        """
        if not self.embedding_service.available:
            logger.debug("Embedding service unavailable — skipping embed for msg %d", message_id)
            return

        repo = EmbeddingRepository(session)

        # Don't re-embed
        if await repo.has_embedding(message_id):
            return

        try:
            vector = await self.embedding_service.embed(text)
            await repo.save_embedding(
                message_id=message_id,
                user_id=user_id,
                chunk_text=text,
                embedding=vector,
            )
            logger.debug("Embedded message %d for user %d", message_id, user_id)
        except Exception:
            logger.exception("Failed to embed message %d", message_id)

    async def recall(
        self,
        session: AsyncSession,
        user_id: int,
        query_text: str,
        top_k: int = 5,
    ) -> list[str]:
        """
        Retrieve the most relevant past message chunks for a user
        based on semantic similarity to the query text.

        Returns a list of chunk_text strings, most relevant first.
        """
        if not self.embedding_service.available:
            return []

        try:
            query_vector = await self.embedding_service.embed(query_text)
            repo = EmbeddingRepository(session)
            results = await repo.search_similar(
                user_id=user_id,
                query_embedding=query_vector,
                top_k=top_k,
            )
            return [r["chunk_text"] for r in results]
        except Exception:
            logger.exception("Episodic recall failed for user %d", user_id)
            return []
