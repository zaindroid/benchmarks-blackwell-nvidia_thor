"""Power metric helpers."""

from __future__ import annotations

from typing import Dict, List, Optional


def power_summary(powers_watts: List[float]) -> Dict[str, Optional[float]]:
    """Summarize sampled power draws (watts)."""
    if not powers_watts:
        return {"available": False}
    return {
        "available": True,
        "average_watts": round(sum(powers_watts) / len(powers_watts), 2),
        "peak_watts": round(max(powers_watts), 2),
        "min_watts": round(min(powers_watts), 2),
        "idle_watts": round(min(powers_watts), 2),
        "samples": len(powers_watts),
    }


def energy_joules(average_watts: float, elapsed_s: float) -> float:
    """Energy consumed over a run: joules = watts * seconds."""
    return average_watts * elapsed_s


def joules_per_sample(average_watts: float, elapsed_s: float, total_samples: int) -> float:
    if total_samples <= 0:
        return 0.0
    return energy_joules(average_watts, elapsed_s) / total_samples
