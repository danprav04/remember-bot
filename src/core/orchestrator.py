"""
Orchestrator — the main message processing pipeline.

Uses all three memory tiers (working, episodic, semantic) to build context.
Runs fact extraction, embedding, and summarization as async background
tasks after the response is sent.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from src.config import AppConfig
from src.core.context_assembler import ContextAssembler
from src.core.fact_extractor import FactExtractor
from src.db.engine import get_session_factory
from src.db.repositories.messages import MessageRepository
from src.db.repositories.users import UserRepository
from src.gateway.base import IncomingMessage
from src.llm.router import LLMRouter
from src.memory.episodic import EpisodicMemory
from src.memory.decay import MemoryDecay
from src.memory.summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Receives normalized messages from any gateway, assembles context,
    queries the LLM, stores everything, and returns the response.
    """

    def __init__(
        self,
        config: AppConfig,
        llm_router: LLMRouter,
        context_assembler: ContextAssembler,
        fact_extractor: FactExtractor,
        episodic_memory: EpisodicMemory,
        summarizer: ConversationSummarizer,
    ):
        self.config = config
        self.llm_router = llm_router
        self.context_assembler = context_assembler
        self.fact_extractor = fact_extractor
        self.episodic_memory = episodic_memory
        self.summarizer = summarizer
        self._session_factory = get_session_factory()

        # Memory decay — runs periodically based on message count
        self._memory_decay = MemoryDecay(
            decay_factor=config.decay.decay_factor,
            min_relevance=config.decay.min_relevance,
            min_age_hours=config.decay.min_age_hours,
        )
        self._decay_enabled = config.decay.enabled
        self._decay_interval = config.decay.interval_messages
        self._message_counter = 0

    async def handle_message(self, incoming: IncomingMessage) -> str:
        """
        Full message processing pipeline:
        1. Resolve user & conversation
        2. Store incoming message
        3. Assemble context (all memory tiers)
        4. Call LLM
        5. Store response
        6. Kick off background tasks (embedding + fact extraction + summarization)
        7. Return response text
        """
        # --- Pre-processing: Transcribe Voice Messages ---
        if incoming.media_type == "voice" and incoming.media_base64:
            try:
                logger.info("Transcribing voice message for user %s...", incoming.platform_user_id)
                transcript_response = await self.llm_router.chat_with_media(
                    task="vision",
                    text="Please transcribe this voice message exactly word-for-word. Output ONLY the transcription and nothing else.",
                    media_base64=incoming.media_base64,
                    media_mime=incoming.media_mime,
                    system_prompt="You are a highly accurate audio transcription engine. Do not answer questions, just transcribe the audio. Do not include markdown or quotation marks.",
                    temperature=0.0,
                )
                transcription = transcript_response.content.strip()
                if transcription:
                    incoming.text = transcription
                    logger.info("Transcription result: %s", incoming.text)
            except Exception as e:
                logger.warning("Voice transcription failed, falling back to default text. Error: %s", e)

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

                # 2. Store the incoming message (now contains transcription if voice)
                user_msg = await msg_repo.save_message(
                    conversation_id=conversation.id,
                    user_id=user.id,
                    role="user",
                    content=incoming.text,
                )
                await session.commit()

                # 3. Assemble context — all memory tiers
                context_messages = await self.context_assembler.assemble(
                    session=session,
                    user_id=user.id,
                    conversation_id=conversation.id,
                    current_message_text=incoming.text,
                    user_display_name=incoming.display_name,
                )

                # 4. Call LLM (with media support)
                if incoming.media_type and incoming.media_base64:
                    llm_response = await self._handle_media_message(
                        incoming=incoming,
                        context_messages=context_messages,
                    )
                else:
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
                bot_msg = await msg_repo.save_message(
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

                # 6. Background tasks — fire and forget
                asyncio.create_task(
                    self._background_memory_tasks(
                        user_id=user.id,
                        conversation_id=conversation.id,
                        user_message_id=user_msg.id,
                        bot_message_id=bot_msg.id,
                        user_text=incoming.text,
                        bot_text=llm_response.content,
                    )
                )

                return llm_response.content

            except Exception:
                await session.rollback()
                logger.exception("Error in orchestrator pipeline")
                raise

    async def _handle_media_message(
        self,
        incoming: IncomingMessage,
        context_messages: list[dict],
    ) -> LLMResponse:
        """
        Handle voice or photo messages via multimodal LLM.
        Builds context from the assembled messages and sends the media inline.
        """
        from src.llm.provider import LLMResponse

        # Build a system prompt from context (system messages + facts)
        system_parts = []
        for msg in context_messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
        system_prompt = "\n\n".join(system_parts)

        if incoming.media_type == "voice":
            text_prompt = (
                "The user sent a voice message. Listen to the audio and respond "
                "to what they said. If they asked you to remember something, confirm it. "
                "If they asked a question, answer it using your memory of this user."
            )
            if incoming.text:
                text_prompt = incoming.text
        elif incoming.media_type == "photo":
            text_prompt = incoming.text or (
                "The user sent a photo. Describe what you see and respond helpfully."
            )
        else:
            text_prompt = incoming.text

        # Use 'vision' task for both images and audio
        task = "vision"

        return await self.llm_router.chat_with_media(
            task=task,
            text=text_prompt,
            media_base64=incoming.media_base64,
            media_mime=incoming.media_mime,
            system_prompt=system_prompt,
            temperature=0.7,
        )

    async def _background_memory_tasks(
        self,
        user_id: int,
        conversation_id: int,
        user_message_id: int,
        bot_message_id: int,
        user_text: str,
        bot_text: str,
    ) -> None:
        """
        Run embedding, fact extraction, and summarization as background tasks.
        Uses separate DB sessions so the main response path isn't affected.
        """
        try:
            # 1. Embed messages
            async with self._session_factory() as session:
                await self.episodic_memory.embed_message(
                    session=session,
                    message_id=user_message_id,
                    user_id=user_id,
                    text=user_text,
                )
                await self.episodic_memory.embed_message(
                    session=session,
                    message_id=bot_message_id,
                    user_id=user_id,
                    text=bot_text,
                )
                await session.commit()

            # 2. Fact extraction
            async with self._session_factory() as session:
                await self.fact_extractor.extract_and_store(
                    session=session,
                    user_id=user_id,
                    user_message=user_text,
                    assistant_response=bot_text,
                    source_message_id=user_message_id,
                )

            # 3. Conversation summarization (if threshold exceeded)
            async with self._session_factory() as session:
                summarized = await self.summarizer.maybe_summarize(
                    session=session,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                if summarized:
                    await session.commit()

            # 4. Memory decay (periodically)
            self._message_counter += 1
            if (
                self._decay_enabled
                and self._message_counter % self._decay_interval == 0
            ):
                async with self._session_factory() as session:
                    stats = await self._memory_decay.run_decay_cycle(session)
                    await session.commit()

        except Exception:
            logger.exception(
                "Background memory tasks failed for user %d", user_id
            )
