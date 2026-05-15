"""
Embeddings repository — stores and searches message embeddings via pgvector.
All queries are scoped by user_id for data isolation.
"""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import MessageEmbedding


class EmbeddingRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_embedding(
        self,
        message_id: int,
        user_id: int,
        chunk_text: str,
        embedding: list[float],
    ) -> MessageEmbedding:
        """Store an embedding vector for a message chunk."""
        record = MessageEmbedding(
            message_id=message_id,
            user_id=user_id,
            chunk_text=chunk_text,
            embedding=embedding,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def search_similar(
        self,
        user_id: int,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find the top-K most similar message embeddings for a user
        using cosine distance (pgvector <=> operator).

        Returns list of dicts with keys: id, message_id, chunk_text, distance
        """
        # Use raw SQL for the pgvector distance operator
        stmt = text("""
            SELECT id, message_id, chunk_text,
                   embedding <=> :query_vec AS distance
            FROM message_embeddings
            WHERE user_id = :user_id
            ORDER BY embedding <=> :query_vec
            LIMIT :top_k
        """)

        result = await self.session.execute(
            stmt,
            {
                "query_vec": str(query_embedding),
                "user_id": user_id,
                "top_k": top_k,
            },
        )

        rows = result.fetchall()
        return [
            {
                "id": row.id,
                "message_id": row.message_id,
                "chunk_text": row.chunk_text,
                "distance": row.distance,
            }
            for row in rows
        ]

    async def has_embedding(self, message_id: int) -> bool:
        """Check if a message already has an embedding stored."""
        stmt = select(MessageEmbedding.id).where(
            MessageEmbedding.message_id == message_id
        ).limit(1)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None
