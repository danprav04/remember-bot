"""
Orchestrator — the main message processing pipeline.

Phase 1: Uses working memory (last N messages) to build context.
Phase 2 will add episodic (vector) and semantic (facts) memory retrieval.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.config import AppConfig
from src.db.engine import get_session_factory
from src.db.repositories.messages import MessageRepository
from src.db.repositories.users import UserRepository
from src.gateway.base import IncomingMessage
from src.llm.router import LLMRouter

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Receives normalized messages from any gateway, assembles context,
    queries the LLM, stores everything, and returns the response.
    """

    def __init__(self, config: AppConfig, llm_router: LLMRouter):
        self.config = config
        self.llm_router = llm_router
        self._session_factory = get_session_factory()

    async def handle_message(self, incoming: IncomingMessage) -> str:
        """
        Full message processing pipeline:
        1. Resolve user & conversation
        2. Store incoming message
        3. Assemble context (working memory for Phase 1)
        4. Call LLM
        5. Store response
        6. Return response text
        """
        async with self._session_factory() as session:
            try:
                # 1. Resolve user and conversation
                user_repo = UserRepository(session)
                msg_repo = MessageRepository(session)

                user = await user_repo.get_or_create(
                    platform=incoming.platform,
                    platform_user_id=incoming.platform_user_id,
                    display_name=incoming.display_name,
                )

                conversation = await msg_repo.get_or_create_conversation(
                    user_id=user.id,
                    platform=incoming.platform,
                    platform_chat_id=incoming.platform_chat_id,
                )

                # 2. Store the incoming message
                await msg_repo.save_message(
                    conversation_id=conversation.id,
                    user_id=user.id,
                    role="user",
                    content=incoming.text,
                )
                await session.commit()

                # 3. Assemble context — Phase 1: working memory only
                context_messages = await self._assemble_context(
                    msg_repo=msg_repo,
                    conversation_id=conversation.id,
                    user_display_name=incoming.display_name,
                )

                # 4. Call LLM
                llm_response = await self.llm_router.chat(
                    task="chat",
                    messages=context_messages,
                )

                logger.info(
                    "LLM response: provider=%s model=%s tokens=%d+%d latency=%.0fms",
                    llm_response.provider,
                    llm_response.model,
                    llm_response.prompt_tokens,
                    llm_response.completion_tokens,
                    llm_response.latency_ms,
                )

                # 5. Store the bot response
                await msg_repo.save_message(
                    conversation_id=conversation.id,
                    user_id=user.id,
                    role="assistant",
                    content=llm_response.content,
                    metadata={
                        "provider": llm_response.provider,
                        "model": llm_response.model,
                        "prompt_tokens": llm_response.prompt_tokens,
                        "completion_tokens": llm_response.completion_tokens,
                        "latency_ms": llm_response.latency_ms,
                    },
                )
                await session.commit()

                return llm_response.content

            except Exception:
                await session.rollback()
                logger.exception("Error in orchestrator pipeline")
                raise

    async def _assemble_context(
        self,
        msg_repo: MessageRepository,
        conversation_id: int,
        user_display_name: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Build the messages list for the LLM.

        Phase 1: System prompt + last N messages (working memory).
        Phase 2 will add episodic and semantic memory retrieval here.
        """
        # System prompt
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        system_prompt = self.config.bot.system_prompt.replace("{current_time}", now)

        if user_display_name:
            system_prompt += f"\n\nThe user's name is: {user_display_name}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        # Working memory: recent messages
        recent = await msg_repo.get_recent_messages(
            conversation_id=conversation_id,
            limit=self.config.memory.working_memory_size,
        )

        for msg in recent:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        return messages
