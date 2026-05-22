"""
Context Assembler — builds the LLM prompt from all memory tiers.

Uses a budget-based approach to fit within model token limits:
  ~10% System prompt (including semantic facts)
  ~20% Document chunks (vector-similar content from uploaded docs)
  ~20% Episodic recall (vector-similar past messages + summaries)
  ~50% Working memory (recent messages)

The assembler dynamically redistributes unused budget from empty tiers
to tiers that have content, and truncates when over budget.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import AppConfig
from src.db.repositories.documents import DocumentRepository
from src.db.repositories.messages import MessageRepository
from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Assembles the full prompt from working memory, episodic memory,
    document chunks, and semantic facts for a given user conversation.
    Enforces a total token budget to prevent exceeding model limits.
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
        platform: str = "telegram",
    ) -> list[dict[str, str]]:
        """
        Build the complete messages list for the LLM, incorporating
        all memory tiers within token budget.
        """
        max_tokens = self.config.memory.max_context_tokens
        msg_repo = MessageRepository(session)

        # --- 1. Retrieve all memory tiers ---

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

        # Semantic memory: stored facts (dynamically searched via vector if query provided)
        facts = await self.semantic.recall(
            session=session,
            user_id=user_id,
            query_text=current_message_text,
            limit=20,
        )

        # Document memory: similar chunks from uploaded documents
        document_chunks = await self._retrieve_document_context(
            session=session,
            user_id=user_id,
            query_text=current_message_text,
        )

        # --- 2. Build the system prompt (base + facts) ---
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        system_prompt = self.config.bot.system_prompt.replace("{current_time}", now)

        if user_display_name:
            system_prompt += f"\n\nThe user's name is: {user_display_name}"

        # Inject platform-specific formatting instructions
        if platform == "whatsapp":
            system_prompt += "\n\nCRITICAL: You are chatting on WhatsApp. You MUST format your text using ONLY WhatsApp formatting rules: *bold*, _italic_, ~strikethrough~. Do NOT use **bold** or markdown headers (#) or HTML."
        else:
            system_prompt += "\n\nCRITICAL: You are chatting on Telegram. You MUST format your text using Telegram Markdown formatting (e.g. **bold**, *italic*)."

        # Inject semantic facts into system prompt
        if facts:
            facts_section = "\n\nThings you remember about this user:\n"
            for i, fact in enumerate(facts, 1):
                facts_section += f"  {i}. {fact}\n"
            system_prompt += facts_section

        system_tokens = count_tokens(system_prompt) + 4  # +4 for message overhead

        # --- 3. Budget allocation ---
        remaining_budget = max_tokens - system_tokens

        # Determine which tiers have content
        has_docs = bool(document_chunks)
        has_episodic = bool(episodic_chunks)
        has_working = bool(recent_messages)

        # Base allocations: 25% docs, 25% episodic, 50% working
        # Redistribute empty tier budgets proportionally
        tiers = {
            "docs": (has_docs, 0.25),
            "episodic": (has_episodic, 0.25),
            "working": (has_working, 0.50),
        }

        active_tiers = {k: v[1] for k, v in tiers.items() if v[0]}
        if not active_tiers:
            active_tiers = {"working": 1.0}

        # Redistribute: scale active tier fractions to sum to 1.0
        total_fraction = sum(active_tiers.values())
        for k in active_tiers:
            active_tiers[k] = active_tiers[k] / total_fraction

        doc_budget = int(remaining_budget * active_tiers.get("docs", 0))
        episodic_budget = int(remaining_budget * active_tiers.get("episodic", 0))
        working_budget = remaining_budget - doc_budget - episodic_budget

        # --- 4. Build messages list ---
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        # --- 5. Document chunks (trimmed to budget) ---
        if document_chunks:
            trimmed_docs = self._trim_doc_chunks(document_chunks, doc_budget)
            if trimmed_docs:
                doc_text = "Relevant content from the user's uploaded documents:\n"
                for dc in trimmed_docs:
                    doc_text += f"  [From: {dc['filename']}] {dc['chunk_text']}\n"
                messages.append({
                    "role": "system",
                    "content": doc_text,
                })
                doc_used = count_tokens(doc_text) + 4
                # Reclaim unused budget
                working_budget += (doc_budget - doc_used)
            else:
                working_budget += doc_budget
        else:
            pass  # budget already redistributed above

        # --- 6. Episodic recall (trimmed to budget) ---
        if episodic_chunks:
            # Filter out chunks already in working memory
            recent_texts = {msg.content for msg in recent_messages}
            unique_chunks = [c for c in episodic_chunks if c not in recent_texts]

            if unique_chunks:
                # Trim chunks to fit budget
                trimmed_chunks = self._trim_to_budget(unique_chunks, episodic_budget)
                if trimmed_chunks:
                    recall_text = "Relevant context from past conversations:\n"
                    for chunk in trimmed_chunks:
                        recall_text += f"  - {chunk}\n"
                    messages.append({
                        "role": "system",
                        "content": recall_text,
                    })
                    # Reclaim unused episodic budget for working memory
                    episodic_used = count_tokens(recall_text) + 4
                    working_budget += (episodic_budget - episodic_used)
                else:
                    # All episodic budget goes to working memory
                    working_budget += episodic_budget
            else:
                working_budget += episodic_budget

        # --- 7. Working memory (trimmed to budget, keep most recent) ---
        working_messages = []
        tokens_used = 0
        # Iterate from newest to oldest so we keep the most recent messages
        for msg in reversed(recent_messages):
            msg_tokens = count_tokens(msg.content) + 4
            if tokens_used + msg_tokens > working_budget:
                break
            working_messages.append({
                "role": msg.role,
                "content": msg.content,
            })
            tokens_used += msg_tokens
        working_messages.reverse()  # Back to chronological order
        messages.extend(working_messages)

        # --- 8. Log final context stats ---
        total_tokens = system_tokens + count_tokens(
            " ".join(m.get("content", "") for m in messages[1:])
        )
        logger.info(
            "Context assembled: %d tokens (budget=%d), %d facts, %d episodic, %d doc chunks, %d working msgs",
            total_tokens,
            max_tokens,
            len(facts),
            len(episodic_chunks),
            len(document_chunks),
            len(working_messages),
        )
        
        # Verbose logging of the exact data pulled
        if facts:
            logger.info("Semantic facts used:\n%s", "\n".join(f"  - {f}" for f in facts))
        if episodic_chunks:
            logger.info("Episodic chunks recalled:\n%s", "\n".join(f"  - {c}" for c in episodic_chunks))
        if document_chunks:
            logger.info("Document chunks recalled:\n%s", "\n".join(
                f"  - [{dc['filename']}] {dc['chunk_text'][:80]}..." for dc in document_chunks
            ))
        if working_messages:
            logger.info("Working memory included %d messages (oldest first).", len(working_messages))

        return messages

    # ------------------------------------------------------------------
    # Document retrieval
    # ------------------------------------------------------------------

    async def _retrieve_document_context(
        self,
        session: AsyncSession,
        user_id: int,
        query_text: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Retrieve relevant document chunks via vector similarity search."""
        if not self.episodic._embedding_service.available:
            return []

        try:
            query_embedding = await self.episodic._embedding_service.embed(query_text)
            doc_repo = DocumentRepository(session)
            results = await doc_repo.search_chunks_by_similarity(
                user_id=user_id,
                query_embedding=query_embedding,
                top_k=top_k,
            )
            # Filter out low-relevance results (distance > 0.8 means not very similar)
            return [r for r in results if r["distance"] < 0.8]
        except Exception:
            logger.exception("Document chunk retrieval failed")
            return []

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    def _trim_to_budget(self, chunks: list[str], budget_tokens: int) -> list[str]:
        """Keep as many chunks as fit within the token budget."""
        result = []
        tokens_used = 0
        overhead = count_tokens("Relevant context from past conversations:\n") + 4
        tokens_used += overhead

        for chunk in chunks:
            chunk_tokens = count_tokens(f"  - {chunk}\n")
            if tokens_used + chunk_tokens > budget_tokens:
                break
            result.append(chunk)
            tokens_used += chunk_tokens

        return result

    def _trim_doc_chunks(self, chunks: list[dict], budget_tokens: int) -> list[dict]:
        """Keep as many document chunks as fit within the token budget."""
        result = []
        tokens_used = 0
        overhead = count_tokens("Relevant content from the user's uploaded documents:\n") + 4
        tokens_used += overhead

        for chunk in chunks:
            text = f"  [From: {chunk['filename']}] {chunk['chunk_text']}\n"
            chunk_tokens = count_tokens(text)
            if tokens_used + chunk_tokens > budget_tokens:
                break
            result.append(chunk)
            tokens_used += chunk_tokens

        return result
