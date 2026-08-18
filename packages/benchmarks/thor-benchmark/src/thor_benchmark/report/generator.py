"""Report generation from benchmark results (markdown/json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from thor_core.logging import get_logger

logger = get_logger(__name__)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _kv_table(rows: Dict[str, Any]) -> str:
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, value in rows.items():
        lines.append(f"| {key} | {_fmt(value)} |")
    return "\n".join(lines)


def generate_markdown(result: Dict[str, Any]) -> str:
    """Render a benchmark result as a markdown report."""
    model = result["model"]
    workload = result["workload"]
    hardware = result["hardware"]
    results = result["results"]

    lines: list[str] = []
    lines.append(f"# Benchmark Report — {model.get('name', 'unknown')}")
    lines.append("")
    lines.append(
        f"- **run_id**: `{result['run_id']}`  \n"
        f"- **timestamp**: {result['timestamp']}  \n"
        f"- **simulated**: {result.get('simulated', False)}"
    )
    lines.append("")

    lines.append("## Model")
    lines.append(_kv_table(model))
    lines.append("")

    lines.append("## Workload")
    lines.append(_kv_table({
        "type": workload.get("type"),
        "precision": model.get("precision"),
        "batch_sizes": ", ".join(map(str, workload.get("batch_sizes", []))),
        "iterations": workload.get("iterations"),
    }))
    lines.append("")

    lines.append("## Hardware")
    lines.append(_kv_table({
        "device": hardware.get("device"),
        "gpu_name": hardware.get("gpu_name"),
        "driver": hardware.get("driver_version"),
        "cuda": hardware.get("cuda_version"),
        "tensorrt": hardware.get("tensorrt_version"),
        "gpu_temp_c": hardware.get("gpu_temp_c"),
        "elapsed_s": hardware.get("elapsed_s"),
    }))
    lines.append("")

    lines.append("## Latency")
    lines.append(_kv_table(results.get("latency", {})))
    lines.append("")

    lines.append("## Throughput")
    lines.append(_kv_table(results.get("throughput", {})))
    lines.append("")

    for section in ("power", "memory", "thermal"):
        lines.append(f"## {section.title()}")
        lines.append(_kv_table(results.get(section, {})))
        lines.append("")

    return "\n".join(lines)


def generate_report(result: Dict[str, Any], fmt: str = "markdown") -> str:
    """Generate a report in the requested format."""
    fmt = fmt.lower()
    if fmt == "json":
        return json.dumps(result, indent=2, default=str)
    if fmt in ("md", "markdown"):
        return generate_markdown(result)
    raise ValueError(f"Unsupported report format: {fmt!r} (use json or markdown)")


def write_report(result: Dict[str, Any], path: str | Path, fmt: str = "markdown") -> Path:
    """Write a report to disk and return the path."""
    out = Path(path)
    if out.suffix in ("", ".md") and fmt == "json":
        out = out.with_suffix(".json")
    out.write_text(generate_report(result, fmt), encoding="utf-8")
    logger.info("report written", path=str(out), fmt=fmt)
    return out
