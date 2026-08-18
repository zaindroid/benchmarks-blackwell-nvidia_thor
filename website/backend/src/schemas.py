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


class Submission(BaseModel):
    """Community model submission (pending moderation)."""

    model_id: str
    name: Optional[str] = None
    architecture: Optional[str] = None
    parameters: Optional[int] = None
    source: Optional[str] = "custom"
    contact_email: Optional[str] = None
    metrics: dict = Field(default_factory=dict)  # e.g. {"latency_p50_ms": 3.2, ...}
    notes: Optional[str] = None


class Review(BaseModel):
    status: str = Field(pattern="^(approved|rejected)$")
    comment: Optional[str] = None
