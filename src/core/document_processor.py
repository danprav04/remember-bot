"""
Document Processor — background pipeline for parsing, chunking, embedding,
and extracting facts from uploaded documents.

Uses a dedicated background API key with its own rate limits,
and Redis for persistent job queuing across restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Callable, Awaitable

import redis.asyncio as aioredis

from src.config import AppConfig
from src.core.chunker import TextChunker
from src.core.file_parser import parse_file
from src.db.engine import get_session_factory
from src.db.repositories.documents import DocumentRepository
from src.db.repositories.facts import FactRepository
from src.llm.embeddings import EmbeddingService
from src.llm.router import LLMRouter

logger = logging.getLogger(__name__)

REDIS_QUEUE_KEY = "remember_bot:document_queue"

# Prompt for extracting facts from document chunks
DOCUMENT_FACT_PROMPT = """\
You are a memory extraction system. Analyze the following document excerpt(s) and extract any information that would be valuable to remember about the user's document for future conversations.

The document is titled: "{filename}"

---

{chunks_text}

---

For each fact worth remembering, respond in JSON format:
{{
  "facts": [
    {{
      "content": "concise fact statement (include the source document name)",
      "tags": ["tag1", "tag2"],
      "relevance_score": 0.8
    }}
  ]
}}

If nothing in these excerpts is worth remembering as standalone facts, respond with:
{{"facts": []}}

IMPORTANT: Respond ONLY with valid JSON, no markdown or extra text.\
"""


class DocumentProcessor:
    """
    Background document processing pipeline.

    Lifecycle: Upload → Parse → Chunk → [Embed chunks + Extract facts] → Complete

    Uses a dedicated EmbeddingService and LLMRouter configured with the
    background API key so that document processing never competes with
    chat traffic for rate limits.
    """

    def __init__(
        self,
        config: AppConfig,
        bg_embedding_service: EmbeddingService,
        bg_llm_router: LLMRouter,
        redis_client: aioredis.Redis,
        notify_callback: Callable[[str, str, str], Awaitable[None]] | None = None,
    ):
        """
        Args:
            config: Application configuration.
            bg_embedding_service: EmbeddingService using the background API key.
            bg_llm_router: LLMRouter using the background API key.
            redis_client: Async Redis client for job queuing.
            notify_callback: async callback(platform, chat_id, message) to send user notifications.
        """
        self.config = config
        self.bg_embedding = bg_embedding_service
        self.bg_llm_router = bg_llm_router
        self.redis = redis_client
        self.notify_callback = notify_callback
        self._session_factory = get_session_factory()
        self._chunker = TextChunker(
            chunk_size=config.documents.chunk_size_tokens,
            overlap=config.documents.chunk_overlap_tokens,
        )
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._semaphore = asyncio.Semaphore(config.documents.max_concurrent_documents)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enqueue(self, document_id: int, file_bytes: bytes) -> None:
        """
        Enqueue a document for background processing.
        Stores file bytes in Redis temporarily (until processed).
        """
        # Store file bytes in Redis with a TTL of 1 hour
        data_key = f"remember_bot:doc_data:{document_id}"
        await self.redis.set(data_key, file_bytes, ex=3600)

        # Push document_id to the processing queue
        await self.redis.lpush(REDIS_QUEUE_KEY, str(document_id))
        logger.info("Document %d enqueued for processing", document_id)

    async def start_worker(self) -> None:
        """Start the background worker that consumes the Redis queue."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Document processor worker started")

    async def stop_worker(self) -> None:
        """Stop the background worker gracefully."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Document processor worker stopped")

    async def recover_incomplete(self) -> None:
        """
        Re-queue documents stuck in 'processing' state (e.g., after a restart).
        Also picks up any 'pending' documents that didn't make it to Redis.
        """
        async with self._session_factory() as session:
            doc_repo = DocumentRepository(session)

            # Reset processing → pending
            incomplete = await doc_repo.get_incomplete_documents()
            for doc in incomplete:
                logger.warning("Recovering stuck document %d (%s)", doc.id, doc.filename)
                await doc_repo.update_status(doc.id, "pending",
                                             error_message="Recovered after restart")
                await session.commit()

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Continuously poll Redis for document processing jobs."""
        while self._running:
            try:
                # BRPOP blocks for up to 5 seconds, then loops
                result = await self.redis.brpop(REDIS_QUEUE_KEY, timeout=5)
                if result is None:
                    continue

                _, doc_id_bytes = result
                document_id = int(doc_id_bytes)

                # Retrieve file bytes from Redis
                data_key = f"remember_bot:doc_data:{document_id}"
                file_bytes = await self.redis.get(data_key)

                if file_bytes is None:
                    logger.error("File bytes not found in Redis for document %d", document_id)
                    async with self._session_factory() as session:
                        doc_repo = DocumentRepository(session)
                        await doc_repo.update_status(
                            document_id, "failed",
                            error_message="File data expired from Redis"
                        )
                        await session.commit()
                    continue

                # Process with concurrency limit
                await self._semaphore.acquire()
                asyncio.create_task(
                    self._process_with_semaphore(document_id, file_bytes, data_key)
                )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in document processor worker loop")
                await asyncio.sleep(5)

    async def _process_with_semaphore(
        self, document_id: int, file_bytes: bytes, data_key: str
    ) -> None:
        """Process a document, then release the semaphore."""
        try:
            await self._process_document(document_id, file_bytes)
        finally:
            # Clean up Redis data
            await self.redis.delete(data_key)
            self._semaphore.release()

    # ------------------------------------------------------------------
    # Core processing pipeline
    # ------------------------------------------------------------------

    async def _process_document(
        self, document_id: int, file_bytes: bytes
    ) -> None:
        """Full processing pipeline for a single document."""
        async with self._session_factory() as session:
            doc_repo = DocumentRepository(session)
            doc = await doc_repo.get_document(document_id)
            if doc is None:
                logger.error("Document %d not found in DB", document_id)
                return

            filename = doc.filename
            user_id = doc.user_id
            platform = doc.platform
            chat_id = doc.platform_chat_id

            logger.info("Processing document %d: %s (%d bytes)",
                        document_id, filename, doc.file_size_bytes)

            # Update status
            await doc_repo.update_status(document_id, "processing")
            await session.commit()

        try:
            # 1. Parse the file
            parsed = await asyncio.to_thread(parse_file, file_bytes, filename)
            preview = parsed.text[:500] if parsed.text else ""

            # 2. Chunk the text
            logger.info("Document %d: parsed OK, chunking %d chars...",
                        document_id, len(parsed.text))
            chunks = self._chunker.chunk(parsed.text)
            if not chunks:
                async with self._session_factory() as session:
                    doc_repo = DocumentRepository(session)
                    await doc_repo.update_status(
                        document_id, "completed",
                        total_chunks=0,
                        extracted_text_preview=preview,
                    )
                    await session.commit()
                await self._notify(platform, chat_id,
                    f"📄 Finished processing *{filename}* — no text content found.")
                return

            # Update total chunks
            logger.info("Document %d: updating DB with %d chunks...",
                        document_id, len(chunks))
            async with self._session_factory() as session:
                doc_repo = DocumentRepository(session)
                await doc_repo.update_status(
                    document_id, "processing",
                    total_chunks=len(chunks),
                    extracted_text_preview=preview,
                )
                await session.commit()
            logger.info("Document %d: DB updated OK", document_id)

            logger.info("Document %d: %d pages, %d chunks",
                        document_id, parsed.page_count, len(chunks))

            # 3. Embed each chunk (rate-limited via bg_embedding_service)
            logger.info("Document %d: starting embedding of %d chunks",
                        document_id, len(chunks))
            for chunk in chunks:
                try:
                    logger.info("Document %d: embedding chunk %d/%d (%d tokens)...",
                                 document_id, chunk.index + 1, len(chunks), chunk.token_count)
                    embedding = await asyncio.wait_for(
                        self.bg_embedding.embed(chunk.text),
                        timeout=120.0,
                    )
                    logger.info("Document %d: chunk %d embedded OK (dim=%d)",
                                 document_id, chunk.index + 1, len(embedding))

                    async with self._session_factory() as session:
                        doc_repo = DocumentRepository(session)
                        await doc_repo.save_chunk(
                            document_id=document_id,
                            user_id=user_id,
                            chunk_index=chunk.index,
                            chunk_text=chunk.text,
                            embedding=embedding,
                        )
                        await doc_repo.increment_processed(document_id)
                        await session.commit()

                except asyncio.TimeoutError:
                    logger.error(
                        "Document %d: embedding chunk %d timed out after 120s",
                        document_id, chunk.index + 1,
                    )
                except Exception:
                    logger.exception(
                        "Failed to embed chunk %d of document %d",
                        chunk.index, document_id,
                    )
            logger.info("Document %d: all chunks embedded", document_id)

            # 4. Extract facts from document (in batches)
            await self._extract_document_facts(
                document_id=document_id,
                user_id=user_id,
                filename=filename,
                chunks=chunks,
            )

            # 5. Mark as completed
            async with self._session_factory() as session:
                doc_repo = DocumentRepository(session)
                await doc_repo.update_status(document_id, "completed")
                await session.commit()

            logger.info("Document %d processing completed: %s", document_id, filename)
            await self._notify(platform, chat_id,
                f"✅ Finished processing *{filename}*! "
                f"({len(chunks)} chunks embedded). "
                f"You can now ask me questions about this document.")

        except ValueError as e:
            # File parsing errors (unsupported format, too large, etc.)
            logger.warning("Document %d failed to parse: %s", document_id, e)
            async with self._session_factory() as session:
                doc_repo = DocumentRepository(session)
                await doc_repo.update_status(
                    document_id, "failed", error_message=str(e)
                )
                await session.commit()
            await self._notify(platform, chat_id,
                f"❌ Failed to process *{filename}*: {e}")

        except Exception:
            logger.exception("Document %d processing failed", document_id)
            async with self._session_factory() as session:
                doc_repo = DocumentRepository(session)
                await doc_repo.update_status(
                    document_id, "failed",
                    error_message="Internal processing error"
                )
                await session.commit()
            await self._notify(platform, chat_id,
                f"❌ Sorry, processing *{filename}* failed due to an internal error.")

    # ------------------------------------------------------------------
    # Fact extraction from document chunks
    # ------------------------------------------------------------------

    async def _extract_document_facts(
        self,
        document_id: int,
        user_id: int,
        filename: str,
        chunks: list,
    ) -> None:
        """Extract facts from document chunks in batches."""
        batch_size = self.config.documents.fact_extraction_batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            chunks_text = "\n\n---\n\n".join(
                f"[Chunk {c.index + 1}]\n{c.text}" for c in batch
            )

            prompt = DOCUMENT_FACT_PROMPT.format(
                filename=filename,
                chunks_text=chunks_text,
            )

            try:
                logger.info("Document %d: extracting facts from chunks %d-%d",
                            document_id, i + 1, i + len(batch))
                llm_response = await asyncio.wait_for(
                    self.bg_llm_router.chat(
                        task="document_fact_extraction",
                        messages=[
                            {"role": "system", "content": "You are a precise fact extraction system. Respond only in valid JSON."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                    ),
                    timeout=120.0,
                )

                facts_data = self._parse_facts_json(llm_response.content)

                if facts_data:
                    async with self._session_factory() as session:
                        fact_repo = FactRepository(session)
                        for fact in facts_data:
                            try:
                                # Embed the fact
                                embedding = None
                                try:
                                    embedding = await self.bg_embedding.embed(fact["content"])
                                except Exception as e:
                                    logger.warning("Failed to embed document fact: %s", e)

                                await fact_repo.create_fact(
                                    user_id=user_id,
                                    content=fact["content"],
                                    tags=fact.get("tags", []),
                                    relevance_score=min(1.0, max(0.0, fact.get("relevance_score", 0.8))),
                                    source_message_id=None,
                                    embedding=embedding,
                                )
                            except Exception:
                                logger.exception("Failed to store document fact: %s", fact.get("content", ""))
                        await session.commit()

                    logger.info(
                        "Extracted %d facts from document %d chunks %d-%d",
                        len(facts_data), document_id, i, i + len(batch) - 1,
                    )

            except Exception:
                logger.exception(
                    "Fact extraction failed for document %d chunks %d-%d",
                    document_id, i, i + len(batch) - 1,
                )

    def _parse_facts_json(self, text: str) -> list[dict]:
        """Parse LLM JSON response into fact dicts."""
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)

            data = json.loads(cleaned)
            facts = data.get("facts", [])
            return [f for f in facts if f.get("content")]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse document fact response: %s", e)
            return []

    # ------------------------------------------------------------------
    # Notification helper
    # ------------------------------------------------------------------

    async def _notify(
        self, platform: str | None, chat_id: str | None, message: str
    ) -> None:
        """Send a notification to the user if callback is configured."""
        if self.notify_callback and platform and chat_id:
            try:
                await self.notify_callback(platform, chat_id, message)
            except Exception:
                logger.exception("Failed to send document notification")
