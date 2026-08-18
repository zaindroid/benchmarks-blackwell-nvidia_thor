"""Latency metric helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence

from thor_core.metrics import summarize_latency


def summarize_latencies(values_ms: Sequence[float]) -> Dict[str, float]:
    """Summarize per-inference latencies (ms) -> p50/p95/p99/min/max/std."""
    return summarize_latency(list(values_ms))


def average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
