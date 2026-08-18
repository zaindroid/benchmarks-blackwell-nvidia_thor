"""Shared request/response models for the leaderboard API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BenchmarkQuery(BaseModel):
    model_id: Optional[str] = None
    workload_type: Optional[str] = None
    metric: str = Field(default="latency_p50")
    top_k: int = Field(default=10, ge=1, le=100)
    timeframe_days: int = Field(default=30, ge=1, le=365)


class ModelComparison(BaseModel):
    model_ids: List[str]
    metrics: List[str] = ["latency_p50", "throughput", "power_watts"]
