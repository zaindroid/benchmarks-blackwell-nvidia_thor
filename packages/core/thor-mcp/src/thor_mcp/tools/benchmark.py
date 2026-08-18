"""Benchmark tools: run / compare / list."""

from __future__ import annotations

import asyncio
import csv
import io
import os
from typing import Any, Dict, List

from thor_mcp.tools import ToolError

SPECS: List[Dict[str, Any]] = [
    {
        "name": "benchmark_run",
        "description": "Run a benchmark on NVIDIA Thor",
        "properties": {
            "model_id": {"type": "string", "description": "Model identifier (e.g. 'meta-llama/Llama-3-8B', 'ultralytics/yolov8n')"},
            "workload_type": {"type": "string", "enum": ["vision", "language", "multimodal", "segmentation", "classification"]},
            "precision": {"type": "string", "enum": ["fp32", "fp16", "int8", "int4", "fp8"]},
            "batch_sizes": {"type": "array", "items": {"type": "integer"}, "default": [1, 4, 8]},
            "iterations": {"type": "integer", "default": 100},
            "warmup_iterations": {"type": "integer", "default": 10},
            "collect_power": {"type": "boolean", "default": True},
            "collect_memory": {"type": "boolean", "default": True},
            "collect_thermal": {"type": "boolean", "default": True},
            "custom_config": {"type": "object", "description": "Workload-specific config; set custom_config.simulate=true for a GPU-free synthetic run"},
        },
        "required": ["model_id", "workload_type"],
    },
    {
        "name": "benchmark_compare",
        "description": "Compare benchmark results across models or configurations",
        "properties": {
            "benchmark_ids": {"type": "array", "items": {"type": "string"}},
            "metrics": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["latency_p50", "latency_p99", "throughput", "power_watts", "memory_mb", "tokens_per_second"],
                },
            },
            "format": {"type": "string", "enum": ["json", "csv", "markdown"]},
        },
        "required": ["benchmark_ids"],
    },
    {
        "name": "benchmark_list",
        "description": "List benchmark runs, optionally filtered by model or workload",
        "properties": {
            "model_id": {"type": "string"},
            "workload_type": {"type": "string"},
            "limit": {"type": "integer", "default": 50},
        },
        "required": [],
    },
]

HANDLERS: Dict[str, Any] = {}

_METRIC_PATHS = {
    "latency_p50": ("latency", "p50_ms"),
    "latency_p99": ("latency", "p99_ms"),
    "throughput": ("throughput", "samples_per_second"),
    "tokens_per_second": ("throughput", "tokens_per_second"),
    "power_watts": ("power", "average_watts"),
    "memory_mb": ("memory", "peak_mb"),
}


def _extract(results: Dict[str, Any], metric: str) -> Any:
    path = _METRIC_PATHS.get(metric)
    if path is None:
        return None
    node: Any = results
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


async def benchmark_run(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Run a benchmark and store the result."""
    await ctx.limiter.check("benchmark_run")
    model_id = args.get("model_id")
    if not model_id:
        raise ToolError("model_id is required")
    workload_type = args.get("workload_type", "vision")

    custom = dict(args.get("custom_config") or {})
    simulate = bool(custom.pop("simulate", False)) or os.getenv("THOR_SIMULATE") == "1"

    result = await asyncio.to_thread(
        ctx.runner.run,
        model_id=model_id,
        workload_type=workload_type,
        precision=args.get("precision", "fp16"),
        batch_sizes=args.get("batch_sizes"),
        iterations=args.get("iterations"),
        warmup_iterations=args.get("warmup_iterations"),
        collect_power=args.get("collect_power", True),
        collect_memory=args.get("collect_memory", True),
        collect_thermal=args.get("collect_thermal", True),
        custom_config=custom,
        simulate=simulate,
    )
    data = result.to_dict()
    await ctx.store.save_run(data)

    results_ = data.get("results", {})
    ctx.registry.update_best_metrics(model_id, {
        "latency_p50_ms": results_.get("latency", {}).get("p50_ms"),
        "throughput_sps": results_.get("throughput", {}).get("samples_per_second"),
        "power_watts": results_.get("power", {}).get("average_watts"),
        "memory_peak_mb": results_.get("memory", {}).get("peak_mb"),
    })
    return {"status": "success", "run_id": data["run_id"], **data}


async def benchmark_compare(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    benchmark_ids = args.get("benchmark_ids") or []
    if not benchmark_ids:
        raise ToolError("benchmark_ids is required")
    metrics = args.get("metrics") or ["latency_p50", "throughput", "power_watts", "memory_mb"]
    fmt = args.get("format", "json")

    runs: List[Dict[str, Any]] = []
    for run_id in benchmark_ids:
        run = await ctx.store.get_run(run_id)
        if run is None:
            raise ToolError(f"unknown benchmark id: {run_id}")
        runs.append(run)

    rows = []
    for run in runs:
        row: Dict[str, Any] = {
            "run_id": run["run_id"],
            "model": run.get("model", {}).get("name"),
            "precision": run.get("model", {}).get("precision"),
            "workload": run.get("workload", {}).get("type"),
            "timestamp": run.get("timestamp"),
            "simulated": run.get("simulated", False),
        }
        for metric in metrics:
            row[metric] = _extract(run.get("results", {}), metric)
        rows.append(row)

    if fmt == "markdown":
        return {"comparison": _to_markdown(rows), "metrics": metrics}
    if fmt == "csv":
        return {"comparison": _to_csv(rows), "metrics": metrics}
    return {"comparison": rows, "metrics": metrics}


async def benchmark_list(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    runs = await ctx.store.list_runs(
        model_id=args.get("model_id"),
        workload_type=args.get("workload_type"),
        limit=args.get("limit", 50),
    )
    summary = [
        {
            "run_id": r["run_id"],
            "timestamp": r.get("timestamp"),
            "model": r.get("model", {}).get("name"),
            "workload": r.get("workload", {}).get("type"),
            "precision": r.get("model", {}).get("precision"),
            "latency_p50_ms": r.get("results", {}).get("latency", {}).get("p50_ms"),
            "throughput_sps": r.get("results", {}).get("throughput", {}).get("samples_per_second"),
            "simulated": r.get("simulated", False),
        }
        for r in runs
    ]
    return {"count": len(summary), "runs": summary}


def _to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No results."
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("-" if v is None else str(v) for v in row.values()) + " |")
    return "\n".join(lines)


def _to_csv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


HANDLERS.update({
    "benchmark_run": benchmark_run,
    "benchmark_compare": benchmark_compare,
    "benchmark_list": benchmark_list,
})
