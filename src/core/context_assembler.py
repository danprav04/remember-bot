"""
Context Assembler — builds the LLM prompt from all three memory tiers.

Uses a budget-based approach to fit within model token limits:
  ~10% System prompt
  ~15% Semantic facts
  ~25% Episodic recall (vector-similar past messages)
  ~40% Working memory (recent messages)
  ~10% User message (already included in working memory)

The assembler dynamically adjusts when memory tiers have no data
(e.g. a new user has no facts, so working memory gets more space).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import AppConfig
from src.db.repositories.messages import MessageRepository
from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Assembles the full prompt from working memory, episodic memory,
    and semantic facts for a given user conversation.
    """

    def __init__(
        self,
        config: AppConfig,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
    ):
        self.config = config
        self.episodic = episodic_memory
        self.semantic = semantic_memory

    async def assemble(
        self,
        session: AsyncSession,
        user_id: int,
        conversation_id: int,
        current_message_text: str,
        user_display_name: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Build the complete messages list for the LLM, incorporating
        all three memory tiers.
        """
        msg_repo = MessageRepository(session)

        # --- 1. Retrieve all memory tiers in parallel ---

        # Working memory: recent messages
        recent_messages = await msg_repo.get_recent_messages(
            conversation_id=conversation_id,
            limit=self.config.memory.working_memory_size,
        )

        # Episodic memory: semantically similar past messages
        episodic_chunks = await self.episodic.recall(
            session=session,
            user_id=user_id,
            query_text=current_message_text,
            top_k=self.config.memory.episodic_top_k,
        )

        # Semantic memory: stored facts
        facts = await self.semantic.recall(
            session=session,
            user_id=user_id,
            limit=20,
        )

        # --- 2. Build the system prompt ---
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        system_prompt = self.config.bot.system_prompt.replace("{current_time}", now)

        if user_display_name:
            system_prompt += f"\n\nThe user's name is: {user_display_name}"

        # --- 3. Inject semantic facts into system prompt ---
        if facts:
            facts_section = "\n\nThings you remember about this user:\n"
            for i, fact in enumerate(facts, 1):
                facts_section += f"  {i}. {fact}\n"
            system_prompt += facts_section

        # --- 4. Build messages list ---
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        # --- 5. Inject episodic recall as a system-level context block ---
        if episodic_chunks:
            # Filter out chunks that are already in working memory
            recent_texts = {msg.content for msg in recent_messages}
            unique_chunks = [c for c in episodic_chunks if c not in recent_texts]

            if unique_chunks:
                recall_text = "Relevant context from past conversations:\n"
                for chunk in unique_chunks:
                    recall_text += f"  - {chunk}\n"
                messages.append({
                    "role": "system",
                    "content": recall_text,
                })

        # --- 6. Working memory: recent conversation messages ---
        for msg in recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        logger.debug(
            "Context assembled: %d system msgs, %d facts, %d episodic chunks, %d working msgs",
            1 + (1 if episodic_chunks else 0),
            len(facts),
            len(episodic_chunks),
            len(recent_messages),
        )

        return messages
