"""Benchmark results resource."""

from __future__ import annotations

from typing import Any, Dict
from urllib.parse import parse_qs, urlparse


async def get_benchmark_results(uri: str, ctx: Any) -> Dict[str, Any]:
    """Return benchmark results, honoring query filters on the URI."""
    query = parse_qs(urlparse(uri).query)
    base, _, _ = uri.partition("?")
    rest = base.removeprefix("thor://benchmarks/results")
    if rest.startswith("/"):
        run_id = rest.strip("/")
        run = await ctx.store.get_run(run_id)
        if run is None:
            return {"uri": uri, "error": f"unknown run: {run_id}"}
        return {"uri": uri, "run": run}

    runs = await ctx.store.list_runs(
        model_id=_first(query, "model_id"),
        workload_type=_first(query, "workload_type"),
        limit=int(_first(query, "limit") or 100),
    )
    return {"uri": uri, "count": len(runs), "runs": runs}


def _first(query: Dict[str, Any], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None
