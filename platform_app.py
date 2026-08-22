"""ThorAI platform app — MCP endpoint + leaderboard API + web UI.

Single composed FastAPI application served in production:

* ``/mcp``                 — MCP streamable-HTTP endpoint (13 tools)
* ``/api/*``               — leaderboard REST API (+ submission portal)
* ``/``                    — React web UI (website/frontend/dist)
* ``/health`` ``/ready``   — deployment liveness/readiness
* ``/version``             — build metadata
* ``/openapi.json``        — API spec

Run: ``python platform_app.py`` or ``uvicorn platform_app:app``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from thor_mcp.http_mcp import create_streamable_http_app
from thor_mcp.server import ThorMCPServer

_FRONTEND_DIR = Path(__file__).resolve().parent / "website" / "frontend" / "dist"


def build_app() -> FastAPI:
    """Compose the MCP server, leaderboard API and web UI into one app."""
    mcp_server = ThorMCPServer()
    app = create_streamable_http_app(mcp_server.server)

    # REST API for the web UI — reuses the same handlers as the MCP server
    # so the UI and MCP clients behave identically (including dispatch to
    # the remote Thor device for real benchmarks).
    @app.post("/api/benchmark/run")
    async def api_benchmark_run(payload: Dict[str, Any]) -> Dict[str, Any]:
        from fastapi import HTTPException

        from thor_mcp.tools.benchmark import ToolError, benchmark_run

        try:
            return await benchmark_run(payload, mcp_server.ctx)
        except ToolError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/tools")
    async def api_tools() -> Dict[str, Any]:
        from thor_mcp.tools import build_tools

        return {
            "tools": [
                {"name": t.name, "description": t.description}
                for t in build_tools()
            ]
        }

    @app.get("/api/hardware")
    async def api_hardware() -> Dict[str, Any]:
        from thor_mcp.tools.hardware import hardware_status

        return await hardware_status({}, mcp_server.ctx)

    # Leaderboard REST API (reads DATABASE_URL when provided).
    try:
        from api import router as leaderboard_router

        app.include_router(leaderboard_router)
    except ImportError:  # pragma: no cover - optional component
        pass

    # React web UI (served last so API/MCP routes take precedence).
    if _FRONTEND_DIR.exists():
        app.mount(
            "/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="web"
        )
    return app


app = build_app()


def run() -> None:
    import uvicorn

    uvicorn.run("platform_app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
