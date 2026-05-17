"""
SQLAlchemy ORM models for all database tables.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    platform = Column(String(10), nullable=False)          # 'telegram' | 'whatsapp'
    platform_user_id = Column(String(64), nullable=False)  # Telegram user ID / WhatsApp phone
    display_name = Column(String(255), nullable=True)
    settings = Column(JSONB, default=dict)
    linked_to = Column(Integer, ForeignKey("users.id"), nullable=True)  # Cross-platform link to primary user
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    messages = relationship("Message", back_populates="user")
    facts = relationship("Fact", back_populates="user")

    __table_args__ = (
        Index("uq_user_platform", "platform", "platform_user_id", unique=True),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    platform = Column(String(10), nullable=False)
    platform_chat_id = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at")

    __table_args__ = (
        Index("uq_conv_platform_chat", "platform", "platform_chat_id", unique=True),
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String(10), nullable=False)   # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB, default=dict)  # tokens, provider, model, latency
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    user = relationship("User", back_populates="messages")
    embedding = relationship("MessageEmbedding", back_populates="message", uselist=False)

    __table_args__ = (
        Index("idx_messages_conv_time", "conversation_id", "created_at"),
        Index("idx_messages_user", "user_id", "created_at"),
    )


class MessageEmbedding(Base):
    __tablename__ = "message_embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    message_id = Column(BigInteger, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(3072), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    message = relationship("Message", back_populates="embedding")

    __table_args__ = (
        Index("idx_embeddings_user", "user_id"),
        # Note: HNSW index limited to 2000 dims, gemini-embedding-002 outputs 3072.
        # Exact search via <=> operator is fine at this scale. Add IVFFlat index
        # later when data grows (requires training on existing rows).
    )


class Fact(Base):
    __tablename__ = "facts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)                    # Natural language fact
    tags = Column(ARRAY(Text), default=list)                  # Free-form tags
    embedding = Column(Vector(3072), nullable=True)           # Semantic embedding for vector search
    relevance_score = Column(Float, default=1.0)
    source_message_id = Column(BigInteger, ForeignKey("messages.id"), nullable=True)
    superseded_by = Column(Integer, ForeignKey("facts.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="facts")

    __table_args__ = (
        Index("idx_facts_user_active", "user_id", postgresql_where=(is_active == True)),
        Index("idx_facts_tags", "tags", postgresql_using="gin"),
    )


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    summary_text = Column(Text, nullable=False)
    message_range_start = Column(BigInteger, ForeignKey("messages.id"), nullable=True)
    message_range_end = Column(BigInteger, ForeignKey("messages.id"), nullable=True)
    embedding = Column(Vector(3072), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_summaries_user", "user_id"),
    )
