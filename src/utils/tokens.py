"""
Token counting utilities for context budget management.
"""

from __future__ import annotations

import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for a text string.
    Uses tiktoken with a GPT-4 encoding as a reasonable approximation
    for all providers (actual counts may vary slightly).
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def count_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate total tokens across a list of chat messages."""
    total = 0
    for msg in messages:
        total += 4  # per-message overhead (role, separators)
        total += count_tokens(msg.get("content", ""))
    total += 2  # priming overhead
    return total
