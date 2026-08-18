"""Unified metric collection and summary statistics.

Mirrors the benchmark result schema used across the platform
(see docs/api-reference.md and website/database/migrations).
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (matches numpy default behaviour)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(ordered[int(k)])
    return float(ordered[lo] * (hi - k) + ordered[hi] * (k - lo))


def summarize_latency(values_ms: Sequence[float]) -> Dict[str, float]:
    """Summarize a list of per-inference latencies (ms)."""
    if not values_ms:
        return {
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
            "count": 0,
        }
    return {
        "p50_ms": round(percentile(values_ms, 50), 3),
        "p95_ms": round(percentile(values_ms, 95), 3),
        "p99_ms": round(percentile(values_ms, 99), 3),
        "min_ms": round(min(values_ms), 3),
        "max_ms": round(max(values_ms), 3),
        "std_ms": round(statistics.stdev(values_ms), 3) if len(values_ms) > 1 else 0.0,
        "count": len(values_ms),
    }


class MetricCollector:
    """Collects latency/throughput/power/memory/thermal samples and emits
    a result dict matching the platform benchmark schema."""

    def __init__(self) -> None:
        self._latencies: List[float] = []
        self._tokens_per_second: List[float] = []
        self._samples_per_second: List[float] = []
        self._power: Optional[Dict[str, Any]] = None
        self._power_elapsed_s: float = 0.0
        self._memory: Optional[Dict[str, Any]] = None
        self._thermal: Optional[Dict[str, Any]] = None
        self._max_batch_size: int = 0
        self._total_samples: int = 0

    # -- collection -----------------------------------------------------
    def add_latency(self, latencies_ms: Sequence[float]) -> None:
        self._latencies.extend(latencies_ms)

    def add_tokens_per_second(self, values: Sequence[float]) -> None:
        self._tokens_per_second.extend(values)

    def add_throughput(self, samples_per_second: float, batch_size: int, total_samples: int) -> None:
        self._samples_per_second.append(samples_per_second)
        self._max_batch_size = max(self._max_batch_size, batch_size)
        self._total_samples = max(self._total_samples, total_samples)

    def add_power(self, stats: Dict[str, Any], elapsed_s: float) -> None:
        self._power = stats
        self._power_elapsed_s = elapsed_s

    def add_memory(self, stats: Dict[str, Any]) -> None:
        self._memory = stats

    def add_thermal(self, stats: Dict[str, Any]) -> None:
        self._thermal = stats

    # -- derived values -------------------------------------------------
    def get_time_to_first_token(self) -> float:
        return percentile(self._latencies, 50) if self._latencies else 0.0

    def get_tokens_per_second(self) -> float:
        return statistics.mean(self._tokens_per_second) if self._tokens_per_second else 0.0

    def get_decode_latency(self) -> float:
        return percentile(self._latencies, 50) if self._latencies else 0.0

    def get_average_power(self) -> float:
        return (self._power or {}).get("average_watts", 0.0)

    def get_peak_memory(self) -> float:
        return (self._memory or {}).get("peak_mb", 0.0)

    # -- serialization --------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        avg_power = self.get_average_power()
        joules = avg_power * self._power_elapsed_s if self._power and avg_power else 0.0
        per_sample = joules / self._total_samples if self._total_samples else 0.0

        return {
            "latency": summarize_latency(self._latencies),
            "throughput": {
                "samples_per_second": round(
                    statistics.mean(self._samples_per_second), 2
                ) if self._samples_per_second else 0.0,
                "max_batch_size": self._max_batch_size,
                "tokens_per_second": round(self.get_tokens_per_second(), 2),
            },
            "power": {
                **(self._power or {}),
                "average_watts": round(avg_power, 2) if self._power else None,
                "joules_per_sample": round(per_sample, 4),
            },
            "memory": {
                **(self._memory or {}),
                "available": bool(self._memory),
            },
            "thermal": {
                **(self._thermal or {}),
                "available": bool(self._thermal),
            },
        }
