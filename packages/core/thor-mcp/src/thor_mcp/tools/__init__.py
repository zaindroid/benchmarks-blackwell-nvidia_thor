"""MCP tool registry — specs and handlers for ThorMCP."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from mcp.types import Tool


class ToolError(RuntimeError):
    """Raised by tool handlers; surfaced as an MCP tool error."""


from thor_mcp.tools import benchmark  # noqa: E402
from thor_mcp.tools import datasets  # noqa: E402
from thor_mcp.tools import experiments  # noqa: E402
from thor_mcp.tools import hardware  # noqa: E402
from thor_mcp.tools import models  # noqa: E402
from thor_mcp.tools import reports  # noqa: E402


# name -> (spec dict, handler)
_GROUPS = [benchmark, models, datasets, reports, experiments, hardware]

SPECS: List[Dict[str, Any]] = [spec for group in _GROUPS for spec in group.SPECS]
HANDLERS: Dict[str, Callable[..., Any]] = {}
for group in _GROUPS:
    HANDLERS.update(group.HANDLERS)


def build_tools() -> List[Tool]:
    """Build MCP Tool definitions from the registry."""
    return [
        Tool(
            name=spec["name"],
            description=spec["description"],
            inputSchema={
                "type": "object",
                "properties": spec.get("properties", {}),
                "required": spec.get("required", []),
            },
        )
        for spec in SPECS
    ]


async def dispatch(name: str, arguments: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Dispatch a tool call to its handler."""
    handler = HANDLERS.get(name)
    if handler is None:
        raise ToolError(f"Unknown tool: {name}")
    return await handler(arguments or {}, ctx)
