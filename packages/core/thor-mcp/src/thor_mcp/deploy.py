"""Platform deployment endpoints (hosting contract).

Satisfies the endpoints required by typical deployment platforms:

* ``GET /health``    -> ``{"status": "ok"}`` — must NOT touch the database
* ``GET /ready``     -> 200 once dependencies (DB etc.) are reachable
* ``GET /version``   -> ``{"sha": ..., "built": ...}``
* ``GET /openapi.json`` — provided automatically by FastAPI

Also honours deployment conventions: JSON logs to stdout (when
``APP_ENV`` is set) and environment-only configuration.
"""

from __future__ import annotations

import os
from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter, Response, status

ReadyCheck = Callable[[], Awaitable[bool]]


def build_info() -> dict[str, Any]:
    """Return build metadata (git sha + build time) from the environment."""
    return {
        "sha": os.getenv("THOR_BUILD_SHA", "dev"),
        "built": os.getenv("THOR_BUILD_TIME", "dev"),
    }


async def default_ready_check() -> bool:
    """Ready when the configured database is reachable (if any).

    Without a DATABASE_URL the app has no external dependencies and is
    always ready. With one, a short connection probe decides.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        return True
    try:
        import asyncpg

        conn = await asyncpg.connect(url, timeout=2, command_timeout=2)
        await conn.close()
        return True
    except Exception:
        return False


def platform_router(check_ready: Optional[ReadyCheck] = None) -> APIRouter:
    """Build the /health, /ready, /version router for a deployable app."""
    ready_check = check_ready or default_ready_check
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, Any]:
        # Contract: 200 {"status": "ok"} — never touches the database.
        return {"status": "ok"}

    @router.get("/ready")
    async def ready(response: Response) -> dict[str, Any]:
        try:
            ok = await ready_check()
        except Exception:
            ok = False
        if not ok:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "not_ready"}
        return {"status": "ready"}

    @router.get("/version")
    async def version() -> dict[str, Any]:
        return build_info()

    return router
