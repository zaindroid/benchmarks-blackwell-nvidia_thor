"""MCP resources — queryable data exposed under ``thor://`` URIs."""

from __future__ import annotations

from typing import Any, Dict, List

from mcp.types import Resource

from thor_mcp.resources import experiments, models, results, telemetry

_RESOURCES: List[Dict[str, str]] = [
    {
        "uri": "thor://benchmarks/results",
        "name": "Benchmark Results",
        "mimeType": "application/json",
        "description": "All benchmark results from Thor",
    },
    {
        "uri": "thor://models/registry",
        "name": "Model Registry",
        "mimeType": "application/json",
        "description": "Registered models for Thor",
    },
    {
        "uri": "thor://hardware/telemetry",
        "name": "Hardware Telemetry",
        "mimeType": "application/json",
        "description": "Real-time hardware telemetry",
    },
    {
        "uri": "thor://experiments/history",
        "name": "Experiment History",
        "mimeType": "application/json",
        "description": "Tracked research experiments",
    },
]


def resource_list() -> List[Resource]:
    return [
        Resource(
            uri=r["uri"],
            name=r["name"],
            mimeType=r["mimeType"],
            description=r["description"],
        )
        for r in _RESOURCES
    ]


async def read_resource(uri: str, ctx: Any) -> Dict[str, Any]:
    """Resolve a ``thor://`` URI to its JSON payload."""
    if uri.startswith("thor://benchmarks/results"):
        return await results.get_benchmark_results(uri, ctx)
    if uri.startswith("thor://models/registry"):
        return models.get_model_registry(uri, ctx)
    if uri.startswith("thor://hardware/telemetry"):
        return telemetry.get_hardware_telemetry(uri, ctx)
    if uri.startswith("thor://experiments/history"):
        return experiments.get_experiment_history(uri, ctx)
    raise ValueError(f"Unknown resource: {uri}")
