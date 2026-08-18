"""Memory metric helpers."""

from __future__ import annotations

from typing import Any, Dict, List


def memory_summary(samples_mb: List[float]) -> Dict[str, Any]:
    """Summarize sampled GPU memory usage (MB)."""
    if not samples_mb:
        return {"available": False}
    return {
        "available": True,
        "peak_mb": round(max(samples_mb), 1),
        "average_mb": round(sum(samples_mb) / len(samples_mb), 1),
        "allocation_pattern": [round(m, 1) for m in samples_mb[-200:]],
    }
