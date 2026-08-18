"""In-process token-bucket rate limiter (works for stdio and HTTP modes)."""

from __future__ import annotations

import time
from typing import Dict, Tuple


class RateLimitExceeded(RuntimeError):
    """Raised when a client exceeds its allowed request rate."""


class RateLimiter:
    """Token-bucket limiter keyed by client/tool name.

    ``requests_per_minute == 0`` disables rate limiting.
    """

    def __init__(self, requests_per_minute: int = 10):
        self.rpm = requests_per_minute
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def _refill(self, key: str) -> float:
        now = time.monotonic()
        if self.rpm <= 0:
            return self.rpm  # unlimited
        tokens, last = self._buckets.get(key, (float(self.rpm), now))
        tokens = min(float(self.rpm), tokens + (now - last) * (self.rpm / 60.0))
        self._buckets[key] = (tokens, now)
        return tokens

    async def check(self, key: str) -> None:
        """Consume one token; raise :class:`RateLimitExceeded` when empty."""
        if self.rpm <= 0:
            return
        tokens = self._refill(key)
        if tokens < 1.0:
            raise RateLimitExceeded(
                f"rate limit exceeded ({self.rpm}/min for {key!r})"
            )
        self._buckets[key] = (tokens - 1.0, self._buckets[key][1])

    def reset(self) -> None:
        self._buckets.clear()
