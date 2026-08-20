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
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter, Response, status

ReadyCheck = Callable[[], Awaitable[bool]]


def _read_build_file(name: str) -> Optional[str]:
    """Read a stamped build-metadata file (``/app`` or repo root)."""
    candidates = [
        Path("/app") / name,
        Path(__file__).resolve().parents[5] / name,
    ]
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError:
            continue
    return None


def build_info() -> dict[str, Any]:
    """Return build metadata (git sha + build time).

    Resolution order: ``THOR_BUILD_SHA``/``THOR_BUILD_TIME`` env vars
    (from build args), then files stamped at image build time
    (``.build_sha`` / ``.build_time``), then ``dev``.
    """
    sha = os.getenv("THOR_BUILD_SHA") or ""
    built = os.getenv("THOR_BUILD_TIME") or ""
    if not sha or sha == "dev":
        sha = _read_build_file(".build_sha") or "dev"
    if not built or built == "dev":
        built = _read_build_file(".build_time") or "dev"
    return {"sha": sha, "built": built}


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
