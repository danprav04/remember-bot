"""
Context Assembler — builds the LLM prompt from all memory tiers.

Uses a budget-based approach to fit within model token limits:
  ~10% System prompt (including semantic facts)
  ~40% Document chunks (grouped by source, positioned near the question)
  ~15% Episodic recall (vector-similar past messages + summaries)
  ~35% Working memory (recent messages)

Document chunks are placed AFTER working memory (right before the user's
current message) so the LLM pays maximum attention to them when answering
document-related questions.

When a specific document is identified by filename, ALL of its chunks are
retrieved for full document recall.
"""

from __future__ import annotations

import logging
import re
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

        Prompt order (optimized for LLM attention):
          1. System prompt (with facts)
          2. Episodic recall
          3. Working memory (conversation history, EXCLUDING the current message)
          4. Document context (right before the question — maximum attention)
          5. Current user message (last)
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
        # This now includes focused full-document retrieval when a
        # specific document is referenced by name.
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

        # Budget allocations: when docs are present, give them 40%
        # to allow full document recall. Otherwise redistribute.
        tiers = {
            "docs": (has_docs, 0.40),
            "episodic": (has_episodic, 0.15),
            "working": (has_working, 0.45),
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

        # --- 5. Episodic recall (placed first, before working memory) ---
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

        # --- 6. Working memory (conversation history) ---
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

        # --- 7. Document chunks (RIGHT BEFORE the user's question) ---
        # This positioning ensures maximum LLM attention on document content
        # when answering document-related queries.
        if document_chunks:
            trimmed_docs = self._trim_doc_chunks(document_chunks, doc_budget)
            if trimmed_docs:
                doc_text = self._format_document_context(trimmed_docs)
                messages.append({
                    "role": "system",
                    "content": doc_text,
                })
                doc_used = count_tokens(doc_text) + 4
                # Note: unused doc budget is not redistributed since
                # working memory was already built above.
            # If no docs fit, budget is simply unused
        # (budget already redistributed above if no docs at all)

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
    # Document context formatting
    # ------------------------------------------------------------------

    def _format_document_context(self, chunks: list[dict]) -> str:
        """Format document chunks grouped by source document with clear
        headers. This makes it obvious to the LLM which document each
        chunk belongs to."""
        # Group chunks by document
        docs: dict[str, list[dict]] = {}
        for chunk in chunks:
            fname = chunk.get("filename", "unknown")
            if fname not in docs:
                docs[fname] = []
            docs[fname].append(chunk)

        lines = [
            "RETRIEVED DOCUMENT CONTENT — use this as your PRIMARY source "
            "when answering questions about these documents. If the user asks "
            "about a document and its content is below, answer FROM this content. "
            "If the user's question could refer to multiple documents listed here, "
            "briefly list the matching documents and ask which one they mean.\n"
        ]

        for fname, file_chunks in docs.items():
            # Sort by chunk_index if available
            file_chunks.sort(key=lambda c: c.get("chunk_index", 0))
            lines.append(f"📄 DOCUMENT: {fname}")
            lines.append("-" * 40)
            for fc in file_chunks:
                lines.append(fc["chunk_text"])
            lines.append("")  # blank line between documents

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Document retrieval
    # ------------------------------------------------------------------

    async def _retrieve_document_context(
        self,
        session: AsyncSession,
        user_id: int,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve relevant document chunks via vector similarity search,
        with focused full-document retrieval when a specific document is
        referenced by name.

        Strategy:
        1. Try filename-based matching first — if the user references a
           specific document, retrieve ALL chunks from that document.
        2. Also run vector similarity search for content-based matches.
        3. Merge and deduplicate results, prioritizing filename matches.
        """
        doc_repo = DocumentRepository(session)
        results: list[dict] = []
        filename_doc_ids: set[int] = set()

        # 1. Filename-based focused retrieval (highest priority)
        # If the user references a specific document by name, retrieve
        # ALL chunks from that document for full recall.
        filename_results = await self._search_by_filename_hints(
            doc_repo, user_id, query_text
        )
        if filename_results:
            # Identify which documents were matched by filename
            filename_doc_ids = {r["document_id"] for r in filename_results}

            # For each matched document, retrieve ALL chunks (full recall)
            for doc_id in filename_doc_ids:
                all_chunks = await doc_repo.get_all_chunks_for_document(
                    document_id=doc_id,
                    user_id=user_id,
                )
                # Add all chunks, deduplicating by ID
                existing_ids = {r["id"] for r in results}
                for chunk in all_chunks:
                    if chunk["id"] not in existing_ids:
                        results.append(chunk)
                        existing_ids.add(chunk["id"])

            logger.info(
                "Filename match: retrieved ALL chunks from %d document(s): %s",
                len(filename_doc_ids),
                ", ".join(r["filename"] for r in filename_results[:3]),
            )

        # 2. Vector similarity search (supplements filename results)
        if self.episodic.embedding_service.available:
            try:
                query_embedding = await self.episodic.embedding_service.embed(query_text)
                vector_results = await doc_repo.search_chunks_by_similarity(
                    user_id=user_id,
                    query_embedding=query_embedding,
                    top_k=top_k,
                )
                # Relaxed threshold: cosine distance range is 0–2, allow up to 1.2
                filtered = [r for r in vector_results if r["distance"] < 1.2]

                # Add vector results that aren't already from filename matches
                existing_ids = {r["id"] for r in results}
                for r in filtered:
                    if r["id"] not in existing_ids:
                        results.append(r)
                        existing_ids.add(r["id"])
            except Exception:
                logger.exception("Document chunk vector retrieval failed")

        return results

    async def _search_by_filename_hints(
        self,
        doc_repo: DocumentRepository,
        user_id: int,
        query_text: str,
    ) -> list[dict]:
        """Extract potential document name references from the query and
        search by filename.

        Strategy:
        1. First, fetch all the user's document filenames and check if
           any filename (without extension) appears in the query. This
           works for any language (Hebrew, English, etc.).
        2. Then, also look for filename-like patterns (dates, underscores,
           extensions, quoted strings) as a fallback.
        """
        hints: list[str] = []

        # --- Strategy 1: Match against actual document filenames ---
        # This catches "מה יש בסילבוס" matching "סילבוס.pdf"
        try:
            user_docs = await doc_repo.get_user_documents(user_id, limit=50)
            query_lower = query_text.lower()
            for doc in user_docs:
                if doc.status != "completed":
                    continue
                fname = doc.filename
                # Check filename with and without extension
                name_no_ext = fname.rsplit(".", 1)[0] if "." in fname else fname
                # Check if the filename (or name without extension) appears
                # in the query text (case-insensitive)
                if name_no_ext.lower() in query_lower or fname.lower() in query_lower:
                    hints.append(name_no_ext)
                    logger.info(
                        "Filename match: query contains '%s' (from document '%s')",
                        name_no_ext, fname,
                    )
        except Exception:
            logger.exception("Failed to check user document filenames")

        # --- Strategy 2: Pattern-based hints (fallback) ---
        # Match date-like patterns: 22_05, 22.05, 2026, etc.
        date_patterns = re.findall(r'\d{1,4}[_.\-/]\d{1,4}(?:[_.\-/]\d{2,4})?', query_text)
        hints.extend(date_patterns)

        # Match tokens with underscores (likely filenames)
        underscore_tokens = re.findall(r'\b\w+(?:_\w+)+\b', query_text)
        hints.extend(underscore_tokens)

        # Match quoted strings
        quoted = re.findall(r'["\']([^"\']+)["\']', query_text)
        hints.extend(quoted)

        # Match tokens with file extensions
        ext_tokens = re.findall(r'\b\w+\.(?:pdf|docx|doc|txt|md)\b', query_text, re.IGNORECASE)
        hints.extend(ext_tokens)

        if not hints:
            return []

        # Search for each hint and collect unique results
        all_results: list[dict] = []
        seen_ids: set[int] = set()

        for hint in hints:
            try:
                matches = await doc_repo.search_chunks_by_filename(
                    user_id=user_id,
                    filename_pattern=hint,
                    max_chunks=5,
                )
                for m in matches:
                    if m["id"] not in seen_ids:
                        all_results.append(m)
                        seen_ids.add(m["id"])
            except Exception:
                logger.exception("Filename search failed for hint '%s'", hint)

        return all_results

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
        """Keep as many document chunks as fit within the token budget.
        Prioritizes chunks from filename-matched documents (distance=0.0)."""
        # Sort: filename matches first (distance=0), then by distance
        sorted_chunks = sorted(chunks, key=lambda c: c.get("distance", 999))

        result = []
        tokens_used = 0
        # Account for the header text
        overhead = count_tokens(
            "RETRIEVED DOCUMENT CONTENT — use this as your PRIMARY source "
            "when answering questions about these documents.\n"
        ) + 4
        tokens_used += overhead

        for chunk in sorted_chunks:
            text = f"  [{chunk.get('filename', 'unknown')}] {chunk['chunk_text']}\n"
            chunk_tokens = count_tokens(text)
            if tokens_used + chunk_tokens > budget_tokens:
                break
            result.append(chunk)
            tokens_used += chunk_tokens

        return result
