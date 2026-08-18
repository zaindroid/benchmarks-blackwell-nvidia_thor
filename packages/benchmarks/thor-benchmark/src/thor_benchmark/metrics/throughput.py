"""Throughput metric helpers."""

from __future__ import annotations


def samples_per_second(total_samples: int, elapsed_s: float) -> float:
    """Throughput in samples (inferences) per second."""
    if elapsed_s <= 0:
        return 0.0
    return total_samples / elapsed_s


def tokens_per_second(total_tokens: int, elapsed_s: float) -> float:
    """Decode throughput in tokens per second."""
    if elapsed_s <= 0:
        return 0.0
    return total_tokens / elapsed_s


def latency_ms_per_token(total_ms: float, tokens: int) -> float:
    """Average decode latency per token (ms/token)."""
    if tokens <= 0:
        return 0.0
    return total_ms / tokens
