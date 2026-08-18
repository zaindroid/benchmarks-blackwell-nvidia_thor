"""Experiment tracking tools."""

from __future__ import annotations

from typing import Any, Dict, List

from thor_mcp.tools import ToolError

SPECS: List[Dict[str, Any]] = [
    {
        "name": "experiments_track",
        "description": "Track a research experiment",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "hypothesis": {"type": "string"},
            "config": {"type": "object"},
            "results": {"type": "object"},
            "metrics": {"type": "object"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name"],
    },
    {
        "name": "experiments_list",
        "description": "List tracked experiments",
        "properties": {
            "status": {"type": "string", "enum": ["pending", "running", "completed", "failed"]},
        },
        "required": [],
    },
]

HANDLERS: Dict[str, Any] = {}


async def experiments_track(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    name = args.get("name")
    if not name:
        raise ToolError("name is required")
    experiment = ctx.experiments.track(
        name=name,
        description=args.get("description", ""),
        hypothesis=args.get("hypothesis", ""),
        config=args.get("config"),
        results=args.get("results"),
        metrics=args.get("metrics"),
        tags=args.get("tags"),
    )
    return {"status": "success", "experiment": experiment}


async def experiments_list(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    experiments = ctx.experiments.list(status=args.get("status"))
    return {"count": len(experiments), "experiments": experiments}


HANDLERS.update({
    "experiments_track": experiments_track,
    "experiments_list": experiments_list,
})
