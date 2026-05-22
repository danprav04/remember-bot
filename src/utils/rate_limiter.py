"""
Async sliding-window rate limiter for API call throttling.

Tracks requests-per-minute (RPM), tokens-per-minute (TPM), and
requests-per-day (RPD) using sliding windows backed by deques of
(timestamp, token_count) tuples.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

# Sliding-window durations in seconds
_MINUTE: int = 60
_DAY: int = 86_400


class RateLimiter:
    """Per-API rate limiter with RPM, TPM, RPD sliding windows.

    Uses ``asyncio.Lock`` internally so a single instance is safe to share
    across concurrent coroutines.  Timestamps come from
    ``time.monotonic()`` to avoid issues with wall-clock adjustments.

    Parameters
    ----------
    rpm:
        Maximum requests allowed per 60-second window.
    tpm:
        Maximum tokens allowed per 60-second window.
    rpd:
        Maximum requests allowed per 86 400-second (24-hour) window.
    name:
        Human-readable label used in log messages (e.g. ``"bg_embedding"``).
    """

    def __init__(
        self,
        rpm: int,
        tpm: int,
        rpd: int,
        name: str = "default",
    ) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.name = name

        # Each entry is (monotonic_timestamp, token_count)
        self._minute_window: deque[tuple[float, int]] = deque()
        self._day_window: deque[tuple[float, int]] = deque()

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self, estimated_tokens: int = 1) -> None:
        """Block until capacity is available, then record the usage.

        The method loops with 1-second sleeps while any of the three
        limits (RPM, TPM, RPD) would be exceeded, logging each wait.

        Parameters
        ----------
        estimated_tokens:
            Number of tokens this request is expected to consume.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                self._cleanup_window(self._minute_window, _MINUTE)
                self._cleanup_window(self._day_window, _DAY)

                wait = self._compute_wait(now, estimated_tokens)
                if wait <= 0:
                    # Capacity available — record and return
                    self._minute_window.append((now, estimated_tokens))
                    self._day_window.append((now, estimated_tokens))
                    return

            # Release the lock while sleeping so other coroutines can proceed
            logger.info(
                "RateLimiter[%s]: throttled — waiting %.1fs "
                "(rpm=%d/%d, tpm=%d/%d, rpd=%d/%d)",
                self.name,
                wait,
                self._current_rpm,
                self.rpm,
                self._current_tpm,
                self.tpm,
                self._current_rpd,
                self.rpd,
            )
            await asyncio.sleep(min(wait, 1.0))

    async def try_acquire(self, estimated_tokens: int = 1) -> bool:
        """Non-blocking attempt to acquire capacity.

        Returns ``True`` if capacity was available and consumed,
        ``False`` otherwise.
        """
        async with self._lock:
            now = time.monotonic()
            self._cleanup_window(self._minute_window, _MINUTE)
            self._cleanup_window(self._day_window, _DAY)

            if self._compute_wait(now, estimated_tokens) > 0:
                return False

            self._minute_window.append((now, estimated_tokens))
            self._day_window.append((now, estimated_tokens))
            return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available_rpm(self) -> int:
        """Current remaining requests in the per-minute window."""
        self._cleanup_window(self._minute_window, _MINUTE)
        return max(0, self.rpm - self._current_rpm)

    @property
    def available_tpm(self) -> int:
        """Current remaining tokens in the per-minute window."""
        self._cleanup_window(self._minute_window, _MINUTE)
        return max(0, self.tpm - self._current_tpm)

    @property
    def available_rpd(self) -> int:
        """Current remaining requests in the per-day window."""
        self._cleanup_window(self._day_window, _DAY)
        return max(0, self.rpd - self._current_rpd)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def _current_rpm(self) -> int:
        """Number of requests recorded in the minute window."""
        return len(self._minute_window)

    @property
    def _current_tpm(self) -> int:
        """Sum of tokens recorded in the minute window."""
        return sum(tokens for _, tokens in self._minute_window)

    @property
    def _current_rpd(self) -> int:
        """Number of requests recorded in the day window."""
        return len(self._day_window)

    def _cleanup_window(self, window: deque[tuple[float, int]], seconds: int) -> None:
        """Remove entries older than *seconds* from the left of *window*."""
        cutoff = time.monotonic() - seconds
        while window and window[0][0] < cutoff:
            window.popleft()

    def _compute_wait(self, now: float, estimated_tokens: int) -> float:
        """Return seconds to wait (≤ 0 means capacity is available).

        Checks RPM, TPM, and RPD limits and returns the *longest* wait
        required to satisfy all three.
        """
        wait = 0.0

        # RPM check
        if self._current_rpm >= self.rpm:
            oldest_ts = self._minute_window[0][0]
            wait = max(wait, oldest_ts + _MINUTE - now)

        # TPM check
        if self._current_tpm + estimated_tokens > self.tpm:
            oldest_ts = self._minute_window[0][0]
            wait = max(wait, oldest_ts + _MINUTE - now)

        # RPD check
        if self._current_rpd >= self.rpd:
            oldest_ts = self._day_window[0][0]
            wait = max(wait, oldest_ts + _DAY - now)

        return wait

    def __repr__(self) -> str:
        return (
            f"RateLimiter(name={self.name!r}, "
            f"rpm={self.rpm}, tpm={self.tpm}, rpd={self.rpd})"
        )
