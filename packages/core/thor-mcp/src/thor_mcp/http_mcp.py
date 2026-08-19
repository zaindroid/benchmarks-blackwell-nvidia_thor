"""MCP streamable-HTTP transport (server side).

Serves the ThorMCP server over the standard MCP streamable-HTTP
protocol (JSON-RPC over POST + SSE stream for responses), so any
MCP-compatible client (Claude Desktop, Codex, Cursor, opencode, ...)
can connect to a remote/hosted ThorMCP endpoint.

Run with ``thor-mcp --http-mcp --port 8000``; the MCP endpoint lives
at ``/mcp``.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.lowlevel import Server
from mcp.server.models import InitializationOptions
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.types import PromptsCapability, ResourcesCapability, ServerCapabilities, ToolsCapability
from starlette.routing import Route as StarletteRoute
from starlette.routing import compile_path


class ASGIRoute(StarletteRoute):
    """A route that passes the raw ASGI scope to its handler.

    Starlette's ``Mount`` only matches ``/mcp/...`` (it requires a
    trailing slash), so the exact path ``/mcp`` falls through to any
    catch-all (e.g. a static file mount at ``/``) and is shadowed.
    This route matches the exact path and hands the raw ASGI scope to
    the streamable-HTTP transport (POST/GET/DELETE/OPTIONS).
    """

    def __init__(self, path: str, asgi_app: Any, methods: list[str] | None = None,
                 name: str | None = None):
        self.path = path
        self.endpoint = asgi_app
        self.asgi_app = asgi_app
        self.name = name or "asgi"
        self.methods = set(methods or ["GET", "POST", "DELETE", "OPTIONS"])
        self.path_regex, self.param_convertors, _ = compile_path(path)

    async def handle(self, scope: Any, receive: Any, send: Any) -> None:
        await self.asgi_app(scope, receive, send)


class StreamableHTTPServer:
    """Wraps an mcp ``Server`` with a streamable-HTTP transport.

    The transport's ``connect()`` context provides the read/write
    streams for ``Server.run``; ``handle_request`` is the ASGI handler
    that consumes HTTP requests (POST JSON-RPC + GET SSE streams).
    """

    def __init__(self, mcp_server: Server,
                 server_name: str = "thor-mcp",
                 server_version: str = "0.1.0"):
        self._mcp = mcp_server
        self._server_name = server_name
        self._server_version = server_version
        self._transport = StreamableHTTPServerTransport(mcp_session_id=None)
        self._task: asyncio.Task | None = None

    # -- lifecycle ------------------------------------------------------
    async def start(self) -> None:
        """Start the MCP session loop in the background."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._session_loop(), name="thor-mcp-http")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _session_loop(self) -> None:
        async with self._transport.connect() as (read, write):
            await self._mcp.run(
                read,
                write,
                InitializationOptions(
                    server_name=self._server_name,
                    server_version=self._server_version,
                    capabilities=ServerCapabilities(
                        tools=ToolsCapability(listChanged=False),
                        resources=ResourcesCapability(subscribe=False, listChanged=False),
                        prompts=PromptsCapability(listChanged=False),
                    ),
                ),
            )

    # -- ASGI -----------------------------------------------------------
    async def handle_request(self, scope: Any, receive: Any, send: Any) -> None:
        """ASGI endpoint for the streamable-HTTP transport."""
        await self.start()
        await self._transport.handle_request(scope, receive, send)


def create_streamable_http_app(mcp_server: Server, server_name: str = "thor-mcp",
                               server_version: str = "0.1.0",
                               allow_origins: list[str] | None = None) -> FastAPI:
    """Build the FastAPI app hosting the MCP endpoint at ``/mcp``."""
    http_server = StreamableHTTPServer(mcp_server, server_name, server_version)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await http_server.start()
        yield
        await http_server.stop()

    app = FastAPI(title=f"{server_name} (streamable HTTP)", version=server_version,
                  lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins or ["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id", "mcp-session-id"],
    )

    # Platform contract endpoints (/health, /ready, /version).
    from thor_mcp.deploy import platform_router

    app.include_router(platform_router())

    # Raw ASGI route: the transport handles full HTTP/SSE semantics.
    # An exact-path route (not a Mount) so a catch-all static mount at
    # "/" can never shadow it (Starlette Mounts require a trailing slash).
    app.router.routes.insert(0, ASGIRoute("/mcp", http_server.handle_request))
    return app
