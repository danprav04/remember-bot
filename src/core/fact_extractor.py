"""
Fact Extractor — LLM-driven dynamic fact extraction.

After each user message is responded to, this module runs asynchronously
in the background. It sends the conversation turn to the LLM with an
open-ended extraction prompt. The LLM decides autonomously:
  - Whether anything is worth remembering
  - What the fact is (natural language)
  - How to tag it (free-form)
  - Whether it updates/supersedes a previously stored fact
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.repositories.facts import FactRepository
from src.llm.router import LLMRouter
from src.llm.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """\
You are a memory extraction system. Analyze the following conversation turn and extract any information that would be valuable to remember about the user for future conversations.

This could be anything: preferences, locations of objects, names of people, important dates, instructions for how to behave, technical details they've shared, opinions, goals, routines, health info, or anything else you judge to be worth persisting.

Here are the facts currently stored about this user (if any):
{existing_facts}

---

Conversation turn:
User: {user_message}
Assistant: {assistant_response}

---

For each fact worth remembering, respond in JSON format:
{{
  "facts": [
    {{
      "content": "fact statement using the user's original wording",
      "tags": ["tag1", "tag2"],
      "relevance_score": 0.0-1.0,
      "supersedes_fact_id": null or integer ID of an existing fact this replaces
    }}
  ]
}}

If nothing in this turn is worth remembering, respond with:
{{"facts": []}}

CRITICAL RULES — you MUST follow these:
1. PRESERVE the user's original wording exactly. Do NOT rephrase, summarize, or reword their statements.
2. If the user provides a list, store EACH item as a SEPARATE fact. Do NOT merge or combine items.
3. NEVER fabricate, infer, or add information the user did not explicitly state.
4. NEVER mix information from different lists, topics, or contexts.
5. Keep the user's original structure, ordering, and terminology intact.
6. If the user gives specific names, numbers, or details, reproduce them exactly.

IMPORTANT: Respond ONLY with valid JSON, no markdown or extra text.\
"""


@dataclass
class ExtractedFact:
    content: str
    tags: list[str]
    relevance_score: float
    supersedes_fact_id: int | None


class FactExtractor:
    """Extracts facts from conversation turns using the LLM."""

    def __init__(self, llm_router: LLMRouter, embedding_service: EmbeddingService):
        self.llm_router = llm_router
        self.embedding_service = embedding_service

    async def extract_and_store(
        self,
        session: AsyncSession,
        user_id: int,
        user_message: str,
        assistant_response: str,
        source_message_id: int | None = None,
    ) -> list[ExtractedFact]:
        """
        Extract facts from a conversation turn and store them in the DB.
        Returns the list of extracted facts (may be empty).
        """
        try:
            # Get existing facts for context
            fact_repo = FactRepository(session)
            existing = await fact_repo.get_active_facts(user_id=user_id, limit=50)

            existing_facts_text = "None yet." if not existing else "\n".join(
                f"  [ID={f.id}] {f.content} (tags: {', '.join(f.tags)})"
                for f in existing
            )

            # Build the extraction prompt
            prompt = EXTRACTION_PROMPT.format(
                existing_facts=existing_facts_text,
                user_message=user_message,
                assistant_response=assistant_response,
            )

            # Call the LLM
            llm_response = await self.llm_router.chat(
                task="fact_extraction",
                messages=[
                    {"role": "system", "content": "You are a precise fact extraction system. Respond only in valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistency
            )

            # Parse the response
            extracted = self._parse_response(llm_response.content)

            if not extracted:
                logger.debug("No facts extracted from message for user %d", user_id)
                return []

            # Store each extracted fact
            stored = []
            for fact in extracted:
                try:
                    # Generate embedding for the fact content
                    try:
                        embedding = await self.embedding_service.embed(fact.content)
                        fact_embedding = embedding if embedding else None
                    except Exception as e:
                        logger.warning("Failed to embed fact '%s': %s", fact.content, e)
                        fact_embedding = None

                    if fact.supersedes_fact_id is not None:
                        # Verify the old fact actually belongs to this user
                        old = await fact_repo.get_fact_by_id(fact.supersedes_fact_id, user_id)
                        if old:
                            await fact_repo.supersede_fact(
                                old_fact_id=fact.supersedes_fact_id,
                                user_id=user_id,
                                new_content=fact.content,
                                tags=fact.tags,
                                relevance_score=fact.relevance_score,
                                source_message_id=source_message_id,
                                embedding=fact_embedding,
                            )
                        else:
                            # Old fact not found — just create new
                            await fact_repo.create_fact(
                                user_id=user_id,
                                content=fact.content,
                                tags=fact.tags,
                                relevance_score=fact.relevance_score,
                                source_message_id=source_message_id,
                                embedding=fact_embedding,
                            )
                    else:
                        await fact_repo.create_fact(
                            user_id=user_id,
                            content=fact.content,
                            tags=fact.tags,
                            relevance_score=fact.relevance_score,
                            source_message_id=source_message_id,
                            embedding=fact_embedding,
                        )
                    stored.append(fact)
                except Exception:
                    logger.exception("Failed to store extracted fact: %s", fact.content)

            await session.commit()
            logger.info(
                "Extracted and stored %d fact(s) for user %d",
                len(stored), user_id,
            )
            return stored

        except Exception:
            logger.exception("Fact extraction failed for user %d", user_id)
            return []

    def _parse_response(self, text: str) -> list[ExtractedFact]:
        """Parse the LLM's JSON response into ExtractedFact objects."""
        try:
            # Strip markdown code fences if present
            cleaned = text.strip()
            if cleaned.startswith("```"):
                # Remove ```json and trailing ```
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)

            data = json.loads(cleaned)
            facts_data = data.get("facts", [])

            results = []
            for f in facts_data:
                results.append(ExtractedFact(
                    content=f.get("content", ""),
                    tags=f.get("tags", []),
                    relevance_score=min(1.0, max(0.0, float(f.get("relevance_score", 0.8)))),
                    supersedes_fact_id=f.get("supersedes_fact_id"),
                ))

            return [r for r in results if r.content]  # Filter out empty

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse fact extraction response: %s — raw: %s", e, text[:200])
            return []
