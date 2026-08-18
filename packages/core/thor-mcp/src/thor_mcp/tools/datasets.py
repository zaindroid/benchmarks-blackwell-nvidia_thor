"""Dataset management tools: list / register."""

from __future__ import annotations

from typing import Any, Dict, List

from thor_mcp.tools import ToolError

SPECS: List[Dict[str, Any]] = [
    {
        "name": "datasets_list",
        "description": "List registered datasets",
        "properties": {},
        "required": [],
    },
    {
        "name": "datasets_register",
        "description": "Register a dataset used by benchmarks",
        "properties": {
            "dataset_id": {"type": "string"},
            "name": {"type": "string"},
            "task": {"type": "string", "enum": ["detection", "segmentation", "language", "multimodal"]},
            "source": {"type": "string"},
            "license": {"type": "string"},
            "metadata": {"type": "object"},
        },
        "required": ["dataset_id"],
    },
]

HANDLERS: Dict[str, Any] = {}


async def datasets_list(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    datasets = await ctx.store.list_datasets()
    return {"count": len(datasets), "datasets": datasets}


async def datasets_register(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    dataset_id = args.get("dataset_id")
    if not dataset_id:
        raise ToolError("dataset_id is required")
    entry = await ctx.store.register_dataset(
        dataset_id,
        {
            "name": args.get("name", dataset_id),
            "task": args.get("task"),
            "source": args.get("source"),
            "license": args.get("license"),
            "metadata": args.get("metadata") or {},
        },
    )
    return {"status": "success", "dataset": entry}


HANDLERS.update({
    "datasets_list": datasets_list,
    "datasets_register": datasets_register,
})
