"""
Documents repository — stores document records and embedded chunks,
provides vector search over document chunks via pgvector.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Document, DocumentChunk


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ------------------------------------------------------------------
    # Document lifecycle
    # ------------------------------------------------------------------

    async def create_document(
        self,
        user_id: int,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        platform: str | None = None,
        platform_chat_id: str | None = None,
    ) -> Document:
        """Create a new document record with status='pending'."""
        doc = Document(
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            status="pending",
            platform=platform,
            platform_chat_id=platform_chat_id,
        )
        self.session.add(doc)
        await self.session.flush()
        return doc

    async def update_status(
        self,
        document_id: int,
        status: str,
        error_message: str | None = None,
        total_chunks: int | None = None,
        extracted_text_preview: str | None = None,
    ) -> None:
        """Update a document's processing status."""
        values: dict = {"status": status}
        if error_message is not None:
            values["error_message"] = error_message
        if total_chunks is not None:
            values["total_chunks"] = total_chunks
        if extracted_text_preview is not None:
            values["extracted_text_preview"] = extracted_text_preview
        if status == "completed":
            values["completed_at"] = datetime.now(timezone.utc)

        stmt = update(Document).where(Document.id == document_id).values(**values)
        await self.session.execute(stmt)
        await self.session.flush()

    async def increment_processed(self, document_id: int) -> None:
        """Increment the processed_chunks counter by 1."""
        stmt = (
            update(Document)
            .where(Document.id == document_id)
            .values(processed_chunks=Document.processed_chunks + 1)
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def get_document(self, document_id: int) -> Document | None:
        """Get a document by ID."""
        stmt = select(Document).where(Document.id == document_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Chunk storage
    # ------------------------------------------------------------------

    async def save_chunk(
        self,
        document_id: int,
        user_id: int,
        chunk_index: int,
        chunk_text: str,
        embedding: list[float] | None = None,
    ) -> DocumentChunk:
        """Store an embedded chunk from a document."""
        chunk = DocumentChunk(
            document_id=document_id,
            user_id=user_id,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            embedding=embedding,
        )
        self.session.add(chunk)
        await self.session.flush()
        return chunk

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    async def search_chunks_by_similarity(
        self,
        user_id: int,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find the top-K most similar document chunks for a user
        using cosine distance (pgvector <=> operator).

        Returns list of dicts with keys: id, document_id, chunk_text, distance, filename
        """
        stmt = text("""
            SELECT dc.id, dc.document_id, dc.chunk_text,
                   dc.embedding <=> :query_vec AS distance,
                   d.filename
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.user_id = :user_id
              AND dc.embedding IS NOT NULL
              AND d.status = 'completed'
            ORDER BY dc.embedding <=> :query_vec
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
                "document_id": row.document_id,
                "chunk_text": row.chunk_text,
                "distance": row.distance,
                "filename": row.filename,
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # User queries
    # ------------------------------------------------------------------

    async def get_user_documents(
        self, user_id: int, limit: int = 20
    ) -> list[Document]:
        """List a user's documents, newest first."""
        stmt = (
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_incomplete_documents(self) -> list[Document]:
        """Get all documents stuck in 'processing' state (for recovery after restart)."""
        stmt = select(Document).where(Document.status == "processing")
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_pending_documents(self) -> list[Document]:
        """Get all documents in 'pending' state."""
        stmt = select(Document).where(Document.status == "pending")
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Filename-based search
    # ------------------------------------------------------------------

    async def search_chunks_by_filename(
        self,
        user_id: int,
        filename_pattern: str,
        max_chunks: int = 10,
    ) -> list[dict]:
        """
        Find document chunks for documents whose filename matches the
        given pattern (case-insensitive substring match).

        Returns list of dicts with keys: id, document_id, chunk_index,
        chunk_text, filename
        """
        stmt = text("""
            SELECT dc.id, dc.document_id, dc.chunk_index, dc.chunk_text,
                   d.filename
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.user_id = :user_id
              AND d.status = 'completed'
              AND LOWER(d.filename) LIKE LOWER(:pattern)
            ORDER BY dc.chunk_index
            LIMIT :max_chunks
        """)

        result = await self.session.execute(
            stmt,
            {
                "user_id": user_id,
                "pattern": f"%{filename_pattern}%",
                "max_chunks": max_chunks,
            },
        )

        rows = result.fetchall()
        return [
            {
                "id": row.id,
                "document_id": row.document_id,
                "chunk_index": row.chunk_index,
                "chunk_text": row.chunk_text,
                "filename": row.filename,
                "distance": 0.0,  # exact filename match = best relevance
            }
            for row in rows
        ]

