"""
Conversation Summarizer — compresses old messages into summaries.

When a conversation's unsummarized message count exceeds the threshold,
the summarizer generates a concise summary using the LLM and stores it
in the conversation_summaries table with an embedding for vector recall.

This keeps the episodic memory layer lean while preserving key context.
"""

from __future__ import annotations

import logging

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import AppConfig
from src.db.models import ConversationSummary, Message
from src.db.repositories.messages import MessageRepository
from src.llm.embeddings import EmbeddingService
from src.llm.router import LLMRouter

logger = logging.getLogger(__name__)


SUMMARIZATION_PROMPT = """\
Summarize the following conversation chunk concisely. Capture:
- Key topics discussed
- Important facts mentioned by the user (names, preferences, decisions, etc.)
- Any requests or commitments made
- The overall tone/intent

Be concise but preserve all information that could be useful in future conversations.

Conversation:
{conversation_text}

Summary:\
"""


class ConversationSummarizer:
    """Generates and stores conversation summaries when threshold is exceeded."""

    def __init__(
        self,
        config: AppConfig,
        llm_router: LLMRouter,
        embedding_service: EmbeddingService,
    ):
        self.config = config
        self.llm_router = llm_router
        self.embedding_service = embedding_service
        self.threshold = config.memory.summary_threshold

    async def maybe_summarize(
        self,
        session: AsyncSession,
        conversation_id: int,
        user_id: int,
    ) -> bool:
        """
        Check if summarization is needed and run it if so.
        Returns True if a summary was generated, False otherwise.
        """
        # Find the last summarized message ID for this conversation
        last_summary_end = await self._get_last_summarized_message_id(
            session, conversation_id
        )

        # Count unsummarized messages
        msg_repo = MessageRepository(session)
        unsummarized = await msg_repo.get_oldest_unsummarized_messages(
            conversation_id=conversation_id,
            last_summarized_message_id=last_summary_end,
            limit=self.threshold + 1,  # just need to know if we exceed threshold
        )

        if len(unsummarized) < self.threshold:
            return False

        logger.info(
            "Summarization triggered for conversation %d (%d unsummarized msgs)",
            conversation_id, len(unsummarized),
        )

        # Take the chunk to summarize (keep the most recent messages unsummarized
        # so working memory still has them)
        keep_recent = self.config.memory.working_memory_size
        to_summarize = unsummarized[:-keep_recent] if len(unsummarized) > keep_recent else unsummarized

        if not to_summarize:
            return False

        # Build the conversation text
        conversation_text = "\n".join(
            f"{msg.role.capitalize()}: {msg.content}" for msg in to_summarize
        )

        # Generate summary via LLM
        try:
            prompt = SUMMARIZATION_PROMPT.format(conversation_text=conversation_text)
            llm_response = await self.llm_router.chat(
                task="summarization",
                messages=[
                    {"role": "system", "content": "You are a precise conversation summarizer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            summary_text = llm_response.content.strip()
        except Exception:
            logger.exception("Summarization LLM call failed for conversation %d", conversation_id)
            return False

        # Generate embedding for the summary
        summary_embedding = None
        if self.embedding_service.available:
            try:
                summary_embedding = await self.embedding_service.embed(summary_text)
            except Exception:
                logger.exception("Failed to embed summary for conversation %d", conversation_id)

        # Store the summary
        summary = ConversationSummary(
            conversation_id=conversation_id,
            user_id=user_id,
            summary_text=summary_text,
            message_range_start=to_summarize[0].id,
            message_range_end=to_summarize[-1].id,
            embedding=summary_embedding,
        )
        session.add(summary)
        await session.flush()

        logger.info(
            "Summary created for conversation %d: messages %d–%d (%d chars)",
            conversation_id,
            to_summarize[0].id,
            to_summarize[-1].id,
            len(summary_text),
        )
        return True

    async def _get_last_summarized_message_id(
        self, session: AsyncSession, conversation_id: int
    ) -> int | None:
        """Get the message ID where the last summary ended."""
        stmt = (
            select(ConversationSummary.message_range_end)
            .where(ConversationSummary.conversation_id == conversation_id)
            .order_by(ConversationSummary.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
