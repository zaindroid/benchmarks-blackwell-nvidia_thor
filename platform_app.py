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

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from thor_mcp.http_mcp import create_streamable_http_app
from thor_mcp.server import ThorMCPServer

_FRONTEND_DIR = Path(__file__).resolve().parent / "website" / "frontend" / "dist"


def build_app() -> FastAPI:
    """Compose the MCP server, leaderboard API and web UI into one app."""
    mcp_server = ThorMCPServer()
    app = create_streamable_http_app(mcp_server.server)

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
