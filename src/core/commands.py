"""
Command Handler — processes bot commands like /facts, /search, /forget, /model, /help.

Handles all command logic and DB interactions, called by the gateway layer.
Each method returns a formatted string response to send back to the user.
Also handles cross-platform linking (/link on WhatsApp, /connect on Telegram).
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone

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

        # In-memory store for cross-platform link codes
        # Maps code -> {"user_id": int, "platform": str, "platform_user_id": str, "expires_at": datetime}
        self._link_codes: dict[str, dict] = {}

    async def handle_help(self) -> str:
        """Return a help message listing all available commands."""
        return (
            "🧠 <b>Remember Bot — Commands</b>\n"
            "\n"
            "💬 <b>Memory</b>\n"
            "/facts — Show all stored facts about you\n"
            "/search &lt;query&gt; — Search your facts by keyword\n"
            "/forget &lt;id|all&gt; — Forget a specific fact or all facts\n"
            "\n"
            "📄 <b>Documents</b>\n"
            "/documents — List your uploaded documents\n"
            "\n"
            "⚙️ <b>Info</b>\n"
            "/model — Show current AI model configuration\n"
            "/stats — Show your memory statistics\n"
            "/export — Download a ZIP of all your data\n"
            "/help — Show this help message\n"
            "\n"
            "🔗 <b>Cross-Platform Linking</b>\n"
            "/link — (WhatsApp) Generate a code to link accounts\n"
            "/connect &lt;code&gt; — (Telegram) Link your WhatsApp account\n"
            "\n"
            "💡 Chat naturally, send voice messages 🎤, photos 📷, or documents 📄\n"
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

            lines = [f"🧠 <b>Your stored facts</b> ({len(facts)} total):\n"]
            for fact in facts:
                tags_str = f"  [{', '.join(fact.tags)}]" if fact.tags else ""
                # Escape HTML in content just in case
                content = fact.content.replace("<", "&lt;").replace(">", "&gt;")
                lines.append(f"  <code>[{fact.id}]</code> {content}{tags_str}")

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

            lines = [f"🔍 <b>Facts matching \"{query.strip()}\"</b> ({len(results)}):\n"]
            for fact in results:
                tags_str = f"  [{', '.join(fact.tags)}]" if fact.tags else ""
                content = fact.content.replace("<", "&lt;").replace(">", "&gt;")
                lines.append(f"  <code>[{fact.id}]</code> {content}{tags_str}")

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
                return f"🗑️ Forgot fact <code>[{fact_id}]</code>."
            else:
                return f"❌ Fact <code>[{fact_id}]</code> not found or already forgotten."

    async def handle_model(self) -> str:
        """Show the current model configuration for all tasks."""
        tasks = ["chat", "fact_extraction", "summarization", "embeddings", "vision"]
        lines = ["⚙️ <b>Current AI Model Configuration</b>\n"]

        for task_name in tasks:
            info = await self.llm_router.get_task_info(task_name)
            if "error" in info:
                lines.append(f"  <b>{task_name}</b>: not configured")
            else:
                lines.append(
                    f"  <b>{task_name}</b>: {info['provider']}/{info['model']}"
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
                f"📊 <b>Your Memory Stats</b>\n"
                f"\n"
                f"  💬 Messages: {msg_count}\n"
                f"  🧠 Facts: {fact_count}\n"
                f"  🔗 Embeddings: {emb_count}\n"
                f"  📝 Working memory: last {self.config.memory.working_memory_size} messages\n"
                f"  🎯 Episodic recall: top {self.config.memory.episodic_top_k} similar\n"
                f"  📦 Context budget: {self.config.memory.max_context_tokens} tokens"
            )

    async def handle_export(self, platform: str, platform_user_id: str) -> dict | str:
        """Export all user data as a dictionary. Returns error string if user not found."""
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        from src.db.models import User, Fact, Conversation, MessageEmbedding

        async with self._session_factory() as session:
            # Load user
            stmt = select(User).where(
                User.platform == platform,
                User.platform_user_id == platform_user_id,
            )
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()

            if user is None:
                return "No data found to export."

            # Fetch facts
            facts_stmt = select(Fact).where(Fact.user_id == user.id).order_by(Fact.created_at)
            facts_result = await session.execute(facts_stmt)
            facts = facts_result.scalars().all()

            # Fetch conversations and messages
            convs_stmt = select(Conversation).where(Conversation.user_id == user.id).options(
                selectinload(Conversation.messages)
            ).order_by(Conversation.created_at)
            convs_result = await session.execute(convs_stmt)
            conversations = convs_result.scalars().all()

            # Fetch embeddings
            embs_stmt = select(MessageEmbedding).where(MessageEmbedding.user_id == user.id).order_by(MessageEmbedding.created_at)
            embs_result = await session.execute(embs_stmt)
            embeddings = embs_result.scalars().all()

            # Build dict
            data = {
                "user": {
                    "id": user.id,
                    "platform": user.platform,
                    "platform_user_id": user.platform_user_id,
                    "display_name": user.display_name,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                },
                "facts": [
                    {
                        "id": f.id,
                        "content": f.content,
                        "tags": f.tags,
                        "relevance_score": f.relevance_score,
                        "is_active": f.is_active,
                        "created_at": f.created_at.isoformat() if f.created_at else None,
                        "updated_at": f.updated_at.isoformat() if f.updated_at else None,
                    }
                    for f in facts
                ],
                "conversations": [
                    {
                        "id": c.id,
                        "created_at": c.created_at.isoformat() if c.created_at else None,
                        "messages": [
                            {
                                "id": m.id,
                                "role": m.role,
                                "content": m.content,
                                "created_at": m.created_at.isoformat() if m.created_at else None,
                            }
                            for m in c.messages
                        ]
                    }
                    for c in conversations
                ],
                "embeddings_history": [
                    {
                        "id": e.id,
                        "chunk_text": e.chunk_text,
                        "message_id": e.message_id,
                        "created_at": e.created_at.isoformat() if e.created_at else None,
                    }
                    for e in embeddings
                ]
            }

            return data

    # ------------------------------------------------------------------
    # Cross-platform linking
    # ------------------------------------------------------------------

    async def handle_link(self, platform: str, platform_user_id: str) -> str:
        """Generate a temporary link code (for WhatsApp users to link to Telegram)."""
        # Clean up expired codes first
        now = datetime.now(timezone.utc)
        self._link_codes = {
            k: v for k, v in self._link_codes.items()
            if v["expires_at"] > now
        }

        # Check if this user already has a pending code
        for code, data in self._link_codes.items():
            if (
                data["platform"] == platform
                and data["platform_user_id"] == platform_user_id
            ):
                remaining = int((data["expires_at"] - now).total_seconds())
                return (
                    f"🔗 You already have an active link code:\n\n"
                    f"  <code>{code}</code>\n\n"
                    f"Send this command in your <b>Telegram</b> chat:\n"
                    f"  /connect {code}\n\n"
                    f"⏳ Expires in {remaining} seconds."
                )

        # Resolve user in DB
        async with self._session_factory() as session:
            user = await self._resolve_user(session, platform, platform_user_id)
            if user is None:
                return "❌ Send me a message first so I can create your account."

            if user.linked_to is not None:
                return "✅ Your account is already linked!"

            # Generate a 6-character alphanumeric code
            code = secrets.token_hex(3).upper()  # e.g. "A1B2C3"
            self._link_codes[code] = {
                "user_id": user.id,
                "platform": platform,
                "platform_user_id": platform_user_id,
                "expires_at": now + timedelta(minutes=5),
            }

            return (
                f"🔗 <b>Link Code Generated!</b>\n\n"
                f"Your code: <code>{code}</code>\n\n"
                f"Now open your <b>Telegram</b> chat with me and send:\n"
                f"  /connect {code}\n\n"
                f"⏳ This code expires in 5 minutes."
            )

    async def handle_connect(
        self, platform: str, platform_user_id: str, code: str
    ) -> str:
        """Connect a WhatsApp account to this Telegram account using a link code."""
        code = code.strip().upper()

        if not code:
            return (
                "Usage: /connect <code>\n\n"
                "First, send /link in your WhatsApp chat to get a code."
            )

        # Look up the code
        now = datetime.now(timezone.utc)
        link_data = self._link_codes.get(code)

        if link_data is None:
            return "❌ Invalid or expired code. Send /link on WhatsApp to get a new one."

        if link_data["expires_at"] <= now:
            del self._link_codes[code]
            return "❌ This code has expired. Send /link on WhatsApp to get a new one."

        # Don't link to yourself
        if (
            link_data["platform"] == platform
            and link_data["platform_user_id"] == platform_user_id
        ):
            return "❌ You can't link an account to itself."

        async with self._session_factory() as session:
            user_repo = UserRepository(session)

            # The Telegram user is the primary
            primary_user = await self._resolve_user(
                session, platform, platform_user_id
            )
            if primary_user is None:
                return "❌ Send me a message first so I can create your account."

            # The WhatsApp user is the secondary (to be merged)
            secondary_user = await user_repo.get_by_id(link_data["user_id"])
            if secondary_user is None:
                del self._link_codes[code]
                return "❌ The WhatsApp account was not found."

            if secondary_user.linked_to is not None:
                del self._link_codes[code]
                return "❌ That WhatsApp account is already linked."

            if primary_user.linked_to is not None:
                return "❌ Your Telegram account is already linked to another account."

            # Merge: move all WhatsApp data under the Telegram user
            await user_repo.merge_users(
                primary_id=primary_user.id,
                secondary_id=secondary_user.id,
            )
            await session.commit()

            # Remove the used code
            del self._link_codes[code]

            return (
                f"✅ <b>Accounts linked!</b>\n\n"
                f"Your WhatsApp ({link_data['platform_user_id']}) and "
                f"Telegram accounts now share the same memory.\n\n"
                f"All past facts and conversations have been merged. "
                f"From now on, anything you tell me on either platform "
                f"will be remembered across both! 🧠"
            )

    # ------------------------------------------------------------------
    # Document commands
    # ------------------------------------------------------------------

    async def handle_documents(
        self, platform: str, platform_user_id: str
    ) -> str:
        """List the user's uploaded documents and their processing status."""
        from src.db.repositories.documents import DocumentRepository

        async with self._session_factory() as session:
            user = await self._resolve_user(session, platform, platform_user_id)
            if user is None:
                return "No data found. Start chatting and I'll learn about you!"

            doc_repo = DocumentRepository(session)
            documents = await doc_repo.get_user_documents(user_id=user.id, limit=20)

            if not documents:
                return (
                    "📄 No documents uploaded yet.\n\n"
                    "Send me a PDF, DOCX, Markdown, or text file "
                    "and I'll process and remember its contents!"
                )

            status_icons = {
                "pending": "⏳",
                "processing": "🔄",
                "completed": "✅",
                "failed": "❌",
            }

            lines = [f"📄 <b>Your uploaded documents</b> ({len(documents)}):\n"]
            for doc in documents:
                icon = status_icons.get(doc.status, "❓")
                size_str = (
                    f"{doc.file_size_bytes / 1024:.0f} KB"
                    if doc.file_size_bytes < 1024 * 1024
                    else f"{doc.file_size_bytes / 1024 / 1024:.1f} MB"
                )
                status_extra = ""
                if doc.status == "completed" and doc.total_chunks:
                    status_extra = f" ({doc.total_chunks} chunks)"
                elif doc.status == "processing" and doc.total_chunks:
                    status_extra = f" ({doc.processed_chunks}/{doc.total_chunks} chunks)"
                elif doc.status == "failed" and doc.error_message:
                    status_extra = f" — {doc.error_message[:60]}"

                lines.append(
                    f"  {icon} <b>{doc.filename}</b> ({size_str}){status_extra}"
                )

            return "\n".join(lines)

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
