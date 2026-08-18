"""Report generation tools."""

from __future__ import annotations

from typing import Any, Dict, List

from thor_mcp.tools import ToolError

SPECS: List[Dict[str, Any]] = [
    {
        "name": "reports_generate",
        "description": "Generate a report from a benchmark run",
        "properties": {
            "benchmark_id": {"type": "string", "description": "run_id of a stored benchmark"},
            "format": {"type": "string", "enum": ["markdown", "json"], "default": "markdown"},
            "include_charts": {"type": "boolean", "default": False},
            "template": {"type": "string", "default": "default"},
        },
        "required": ["benchmark_id"],
    },
]

HANDLERS: Dict[str, Any] = {}


async def reports_generate(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    from thor_benchmark.report.generator import generate_report

    benchmark_id = args.get("benchmark_id")
    if not benchmark_id:
        raise ToolError("benchmark_id is required")
    run = await ctx.store.get_run(benchmark_id)
    if run is None:
        raise ToolError(f"unknown benchmark id: {benchmark_id}")

    fmt = args.get("format", "markdown")
    try:
        content = generate_report(run, fmt)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc

    report_id = await ctx.store.save_report({
        "benchmark_id": benchmark_id,
        "format": fmt,
        "content": content,
        "template": args.get("template", "default"),
        "created_at": run.get("timestamp"),
    })
    return {
        "status": "success",
        "report_id": report_id,
        "benchmark_id": benchmark_id,
        "format": fmt,
        "content": content,
    }


HANDLERS.update({"reports_generate": reports_generate})
