"""
Text chunker — splits large text into embedding-sized chunks.

Chunks respect sentence boundaries and include configurable token
overlap between consecutive chunks to preserve context across
boundaries.  Uses the project's ``src.utils.tokens`` module for
token counting.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence-splitting regex
# ---------------------------------------------------------------------------

# Splits on sentence-ending punctuation (.!?) followed by whitespace or end
# of string.  Negative lookbehind avoids splitting on common abbreviations
# (Mr., Mrs., Ms., Dr., Prof., Sr., Jr., vs., etc., e.g., i.e.).
_SENTENCE_BOUNDARY = re.compile(
    r"(?<!\bMr)"
    r"(?<!\bMs)"
    r"(?<!\bDr)"
    r"(?<!\bSr)"
    r"(?<!\bJr)"
    r"(?<!\bvs)"
    r"(?<!\betc)"
    r"(?<!\be\.g)"
    r"(?<!\bi\.e)"
    r"(?<!\bMrs)"
    r"(?<!\bProf)"
    r"[.!?]"
    r"(?:\s+|$)"
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk produced by :class:`TextChunker`.

    Attributes
    ----------
    index:
        Zero-based position of this chunk in the output sequence.
    text:
        The chunk's text content.
    token_count:
        Estimated token count for *text*.
    """

    index: int
    text: str
    token_count: int


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class TextChunker:
    """Split text into token-bounded chunks that respect sentence boundaries.

    Parameters
    ----------
    chunk_size:
        Target number of tokens per chunk.
    overlap:
        Number of overlap tokens to repeat at the beginning of each
        successive chunk (drawn from the tail of the previous chunk).
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        """Split *text* into a list of :class:`Chunk` objects.

        Strategy
        --------
        1. Split text into sentences using a regex that handles common
           abbreviations.
        2. Accumulate sentences until the running token count reaches
           ``chunk_size``.
        3. Emit the chunk, then start the next chunk with enough trailing
           sentences from the previous chunk to approximate ``overlap``
           tokens.
        4. Single sentences that exceed ``chunk_size`` are emitted as
           their own chunk (never skipped).

        Returns an empty list for empty or whitespace-only input.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Pre-compute token counts for every sentence
        sentence_tokens = [count_tokens(s) for s in sentences]

        chunks: list[Chunk] = []
        idx = 0  # current sentence cursor

        while idx < len(sentences):
            start_idx = idx  # remember where this chunk started

            # --- Build one chunk ---
            chunk_sentences: list[str] = []
            chunk_token_total = 0

            while idx < len(sentences):
                stok = sentence_tokens[idx]

                # If adding this sentence would exceed the budget and we
                # already have content, stop (but always accept at least
                # one sentence to avoid infinite loops).
                if chunk_sentences and chunk_token_total + stok > self.chunk_size:
                    break

                chunk_sentences.append(sentences[idx])
                chunk_token_total += stok
                idx += 1

            chunk_text = " ".join(chunk_sentences)
            final_token_count = count_tokens(chunk_text)
            chunks.append(Chunk(
                index=len(chunks),
                text=chunk_text,
                token_count=final_token_count,
            ))

            # --- Compute overlap for the next chunk ---
            if idx < len(sentences) and self.overlap > 0:
                rewound_idx = self._rewind_for_overlap(
                    sentences, sentence_tokens, idx, chunk_sentences,
                )
                # CRITICAL: never rewind to or before the start of the
                # current chunk — that would cause an infinite loop.
                if rewound_idx > start_idx:
                    idx = rewound_idx
                # else: no rewind, idx stays at its current position

        logger.info(
            "Chunked text into %d chunks (chunk_size=%d, overlap=%d, "
            "total_tokens≈%d)",
            len(chunks),
            self.chunk_size,
            self.overlap,
            sum(c.token_count for c in chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split *text* into sentences, preserving the terminating
        punctuation on each sentence and stripping extra whitespace.

        Strategy:
        1. Split on newlines first — this is the primary boundary for
           structured documents (syllabi, lists, etc.) and works for
           all languages including RTL (Hebrew, Arabic).
        2. For each resulting paragraph, apply the sentence-boundary
           regex to split further if needed.
        3. This ensures we never merge unrelated lines into a single
           giant "sentence" just because they lack English-style
           sentence-ending punctuation.
        """
        parts: list[str] = []

        # First split on newlines — this handles structured documents,
        # lists, and RTL text that uses line breaks as separators.
        lines = text.split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Try to split this line further using sentence boundaries
            sub_parts: list[str] = []
            last_end = 0

            for match in _SENTENCE_BOUNDARY.finditer(stripped):
                end = match.end()
                sentence = stripped[last_end:end].strip()
                if sentence:
                    sub_parts.append(sentence)
                last_end = end

            # Trailing text after the last sentence boundary
            tail = stripped[last_end:].strip()
            if tail:
                sub_parts.append(tail)

            # If no sentence boundaries were found, the whole line is one unit
            if not sub_parts:
                sub_parts = [stripped]

            parts.extend(sub_parts)

        return parts

    def _rewind_for_overlap(
        self,
        sentences: list[str],
        sentence_tokens: list[int],
        next_idx: int,
        prev_chunk_sentences: list[str],
    ) -> int:
        """Move *next_idx* backwards so the next chunk begins with
        approximately ``self.overlap`` tokens of repeated content from
        the previous chunk.

        Returns the adjusted sentence index.
        """
        overlap_tokens = 0
        rewind_count = 0

        # Walk backwards through the sentences that formed the previous chunk
        for i in range(len(prev_chunk_sentences) - 1, -1, -1):
            # Map the chunk-local index back to the global sentence list
            global_idx = next_idx - len(prev_chunk_sentences) + i
            stok = sentence_tokens[global_idx]

            if overlap_tokens + stok > self.overlap and rewind_count > 0:
                break
            overlap_tokens += stok
            rewind_count += 1

        return next_idx - rewind_count

    def __repr__(self) -> str:
        return (
            f"TextChunker(chunk_size={self.chunk_size}, "
            f"overlap={self.overlap})"
        )
