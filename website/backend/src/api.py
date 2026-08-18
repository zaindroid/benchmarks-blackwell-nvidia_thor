"""ThorBench leaderboard REST API.

Reads from the PostgreSQL ``benchmark_runs`` table when a database is
reachable (DATABASE_URL env var), otherwise returns empty datasets so
the scaffold runs anywhere. Results are written by the ThorMCP server
(or by pushing result JSON files).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models import BenchmarkQuery, ModelComparison

router = APIRouter()

DATABASE_URL = os.getenv("DATABASE_URL", "")


async def _connect():
    import asyncpg

    return await asyncpg.connect(DATABASE_URL, timeout=3)


def _pg_available() -> bool:
    return bool(DATABASE_URL)


@router.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/api/leaderboard")
async def get_leaderboard(
    model_id: Optional[str] = None,
    workload_type: Optional[str] = None,
    metric: str = "latency_p50",
    top_k: int = 10,
    timeframe_days: int = 30,
) -> Dict[str, Any]:
    """Leaderboard ordered by the requested metric."""
    if not _pg_available():
        return {"leaderboard": []}

    order = {
        "latency_p50": "MIN((results->'latency'->>'p50_ms')::float)",
        "throughput": "MAX((results->'throughput'->>'samples_per_second')::float)",
        "power_watts": "MIN((results->'power'->>'average_watts')::float)",
    }.get(metric, "MIN((results->'latency'->>'p50_ms')::float)")

    query = f"""
        SELECT
            model_info->>'name' as model_name,
            model_info->>'architecture' as architecture,
            model_info->>'parameters' as parameters,
            workload_info->>'type' as workload_type,
            MIN((results->'latency'->>'p50_ms')::float) as best_latency_ms,
            MAX((results->'throughput'->>'samples_per_second')::float) as best_throughput,
            MIN((results->'power'->>'average_watts')::float) as best_power_watts,
            COUNT(*) as benchmark_count
        FROM benchmark_runs
        WHERE timestamp > NOW() - INTERVAL '1 day' * $1
          AND ($2::text IS NULL OR model_info->>'name' = $2)
          AND ($3::text IS NULL OR workload_info->>'type' = $3)
        GROUP BY model_info->>'name', model_info->>'architecture',
                 model_info->>'parameters', workload_info->>'type'
        ORDER BY {order} ASC
        LIMIT $4
    """
    conn = await _connect()
    try:
        rows = await conn.fetch(query, timeframe_days, model_id, workload_type, top_k)
        return {"leaderboard": [dict(row) for row in rows]}
    finally:
        await conn.close()


@router.get("/api/models/{model_id}/history")
async def get_model_history(model_id: str, days: int = 30) -> Dict[str, Any]:
    """Benchmark history for a specific model."""
    if not _pg_available():
        return {"history": []}

    query = """
        SELECT run_id, timestamp, model_info, workload_info, results, git_commit
        FROM benchmark_runs
        WHERE model_info->>'name' = $1
          AND timestamp > NOW() - INTERVAL '1 day' * $2
        ORDER BY timestamp DESC
    """
    conn = await _connect()
    try:
        rows = await conn.fetch(query, model_id, days)
        return {"history": [dict(row) for row in rows]}
    finally:
        await conn.close()


@router.post("/api/compare")
async def compare_models(comparison: ModelComparison) -> Dict[str, Any]:
    """Compare multiple models across metrics."""
    if not _pg_available():
        return {"comparison": []}

    query = """
        SELECT
            model_info->>'name' as model_name,
            MIN((results->'latency'->>'p50_ms')::float) as min_latency_ms,
            MAX((results->'throughput'->>'samples_per_second')::float) as max_throughput,
            MIN((results->'power'->>'average_watts')::float) as min_power_watts,
            MIN((results->'memory'->>'peak_mb')::float) as min_memory_mb
        FROM benchmark_runs
        WHERE model_info->>'name' = ANY($1)
        GROUP BY model_info->>'name'
    """
    conn = await _connect()
    try:
        rows = await conn.fetch(query, comparison.model_ids)
        return {"comparison": [dict(row) for row in rows]}
    finally:
        await conn.close()


@router.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    """Overall platform statistics."""
    if not _pg_available():
        return {
            "total_models": 0, "total_benchmarks": 0,
            "first_benchmark": None, "last_benchmark": None,
            "architectures": 0, "database": "not configured (set DATABASE_URL)",
        }

    query = """
        SELECT
            COUNT(DISTINCT model_info->>'name') as total_models,
            COUNT(*) as total_benchmarks,
            MIN(timestamp) as first_benchmark,
            MAX(timestamp) as last_benchmark,
            COUNT(DISTINCT model_info->>'architecture') as architectures
        FROM benchmark_runs
    """
    conn = await _connect()
    try:
        row = await conn.fetchrow(query)
        return dict(row)
    finally:
        await conn.close()
