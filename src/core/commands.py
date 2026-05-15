"""
Command Handler — processes bot commands like /facts, /search, /forget, /model, /help.

Handles all command logic and DB interactions, called by the gateway layer.
Each method returns a formatted string response to send back to the user.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import AppConfig
from src.db.engine import get_session_factory
from src.db.repositories.facts import FactRepository
from src.db.repositories.users import UserRepository
from src.db.repositories.embeddings import EmbeddingRepository
from src.db.repositories.messages import MessageRepository
from src.llm.router import LLMRouter
from src.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)


class CommandHandler:
    """Handles all bot slash commands."""

    def __init__(
        self,
        config: AppConfig,
        llm_router: LLMRouter,
        episodic_memory: EpisodicMemory,
    ):
        self.config = config
        self.llm_router = llm_router
        self.episodic_memory = episodic_memory
        self._session_factory = get_session_factory()

    async def handle_help(self) -> str:
        """Return a help message listing all available commands."""
        return (
            "🧠 *Remember Bot — Commands*\n"
            "\n"
            "💬 *Memory*\n"
            "/facts — Show all stored facts about you\n"
            "/search <query> — Search your facts by keyword\n"
            "/forget <id|all> — Forget a specific fact or all facts\n"
            "\n"
            "⚙️ *Info*\n"
            "/model — Show current AI model configuration\n"
            "/stats — Show your memory statistics\n"
            "/help — Show this help message\n"
            "\n"
            "💡 Chat naturally, send voice messages 🎤, or photos 📷\n"
            "I'll remember important details automatically!"
        )

    async def handle_facts(
        self, platform: str, platform_user_id: str
    ) -> str:
        """List all active facts stored about the user."""
        async with self._session_factory() as session:
            user = await self._resolve_user(session, platform, platform_user_id)
            if user is None:
                return "No data found. Start chatting and I'll learn about you!"

            fact_repo = FactRepository(session)
            facts = await fact_repo.get_active_facts(user_id=user.id, limit=50)

            if not facts:
                return "📭 No facts stored yet. Chat with me and I'll start remembering!"

            lines = [f"🧠 *Your stored facts* ({len(facts)} total):\n"]
            for fact in facts:
                tags_str = f"  [{', '.join(fact.tags)}]" if fact.tags else ""
                lines.append(f"  `[{fact.id}]` {fact.content}{tags_str}")

            return "\n".join(lines)

    async def handle_search(
        self, platform: str, platform_user_id: str, query: str
    ) -> str:
        """Search stored facts by keyword, plus optionally episodic memory."""
        if not query.strip():
            return "Usage: /search <keyword>\nExample: /search watch"

        async with self._session_factory() as session:
            user = await self._resolve_user(session, platform, platform_user_id)
            if user is None:
                return "No data found yet."

            fact_repo = FactRepository(session)

            # Search by text
            text_results = await fact_repo.search_facts_by_text(
                user_id=user.id, query=query.strip(), limit=10
            )

            # Also try tag search
            tag_results = await fact_repo.search_facts_by_tags(
                user_id=user.id, tags=[query.strip().lower()], limit=5
            )

            # Merge, deduplicate
            seen_ids = set()
            results = []
            for fact in text_results + tag_results:
                if fact.id not in seen_ids:
                    seen_ids.add(fact.id)
                    results.append(fact)

            if not results:
                return f"🔍 No facts found matching \"{query.strip()}\"."

            lines = [f"🔍 *Facts matching \"{query.strip()}\"* ({len(results)}):\n"]
            for fact in results:
                tags_str = f"  [{', '.join(fact.tags)}]" if fact.tags else ""
                lines.append(f"  `[{fact.id}]` {fact.content}{tags_str}")

            return "\n".join(lines)

    async def handle_forget(
        self, platform: str, platform_user_id: str, arg: str
    ) -> str:
        """Forget a specific fact by ID or all facts."""
        if not arg.strip():
            return (
                "Usage:\n"
                "  /forget <id> — Forget a specific fact\n"
                "  /forget all — Forget everything\n"
                "\nUse /facts to see fact IDs."
            )

        async with self._session_factory() as session:
            user = await self._resolve_user(session, platform, platform_user_id)
            if user is None:
                return "No data found."

            fact_repo = FactRepository(session)

            if arg.strip().lower() == "all":
                count = await fact_repo.deactivate_all_facts(user_id=user.id)
                await session.commit()
                if count == 0:
                    return "📭 No facts to forget."
                return f"🗑️ Forgot {count} fact(s). Starting fresh!"

            # Try to parse as integer fact ID
            try:
                fact_id = int(arg.strip())
            except ValueError:
                return "❌ Invalid fact ID. Use /facts to see your facts and their IDs."

            success = await fact_repo.deactivate_fact(fact_id=fact_id, user_id=user.id)
            await session.commit()

            if success:
                return f"🗑️ Forgot fact `[{fact_id}]`."
            else:
                return f"❌ Fact `[{fact_id}]` not found or already forgotten."

    async def handle_model(self) -> str:
        """Show the current model configuration for all tasks."""
        tasks = ["chat", "fact_extraction", "summarization", "embeddings", "vision"]
        lines = ["⚙️ *Current AI Model Configuration*\n"]

        for task_name in tasks:
            info = await self.llm_router.get_task_info(task_name)
            if "error" in info:
                lines.append(f"  *{task_name}*: not configured")
            else:
                lines.append(
                    f"  *{task_name}*: {info['provider']}/{info['model']}"
                )
                if info.get("fallbacks"):
                    for i, fb in enumerate(info["fallbacks"], 1):
                        lines.append(f"    ↳ fallback {i}: {fb['provider']}/{fb['model']}")

        return "\n".join(lines)

    async def handle_stats(
        self, platform: str, platform_user_id: str
    ) -> str:
        """Show memory statistics for the user."""
        async with self._session_factory() as session:
            user = await self._resolve_user(session, platform, platform_user_id)
            if user is None:
                return "No data found yet. Start chatting!"

            fact_repo = FactRepository(session)
            msg_repo = MessageRepository(session)

            fact_count = await fact_repo.count_active_facts(user_id=user.id)

            # Count total messages across all conversations
            from sqlalchemy import select, func
            from src.db.models import Message, MessageEmbedding
            msg_stmt = select(func.count(Message.id)).where(Message.user_id == user.id)
            msg_result = await session.execute(msg_stmt)
            msg_count = msg_result.scalar_one()

            # Count embeddings
            emb_stmt = select(func.count(MessageEmbedding.id)).where(
                MessageEmbedding.user_id == user.id
            )
            emb_result = await session.execute(emb_stmt)
            emb_count = emb_result.scalar_one()

            return (
                f"📊 *Your Memory Stats*\n"
                f"\n"
                f"  💬 Messages: {msg_count}\n"
                f"  🧠 Facts: {fact_count}\n"
                f"  🔗 Embeddings: {emb_count}\n"
                f"  📝 Working memory: last {self.config.memory.working_memory_size} messages\n"
                f"  🎯 Episodic recall: top {self.config.memory.episodic_top_k} similar\n"
                f"  📦 Context budget: {self.config.memory.max_context_tokens} tokens"
            )

    async def _resolve_user(
        self, session: AsyncSession, platform: str, platform_user_id: str
    ):
        """Find the user in the DB. Returns None if not found."""
        user_repo = UserRepository(session)
        from sqlalchemy import select
        from src.db.models import User
        stmt = select(User).where(
            User.platform == platform,
            User.platform_user_id == platform_user_id,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
