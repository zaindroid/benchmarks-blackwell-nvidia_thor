"""Tests for thor_mcp.rate_limit."""

import pytest

from thor_mcp.rate_limit import RateLimitExceeded, RateLimiter


async def test_rate_limiter_allows_rpm_requests():
    limiter = RateLimiter(requests_per_minute=3)
    for _ in range(3):
        await limiter.check("client-a")
    with pytest.raises(RateLimitExceeded):
        await limiter.check("client-a")


async def test_rate_limiter_is_per_key():
    limiter = RateLimiter(requests_per_minute=2)
    await limiter.check("client-a")
    await limiter.check("client-b")
    await limiter.check("client-b")
    with pytest.raises(RateLimitExceeded):
        await limiter.check("client-b")
    await limiter.check("client-a")  # still has 1 token


async def test_rate_limiter_disabled():
    limiter = RateLimiter(requests_per_minute=0)
    for _ in range(5):
        await limiter.check("any")  # never raises


async def test_reset():
    limiter = RateLimiter(requests_per_minute=1)
    await limiter.check("k")
    limiter.reset()
    await limiter.check("k")  # tokens restored
