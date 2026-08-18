"""FastAPI REST bridge for ThorMCP (HTTP mode).

Exposes the same tools/resources over JSON endpoints with bearer-token
auth and rate limiting. For MCP-protocol HTTP transports, use the stdio
mode with an MCP-aware client.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from thor_mcp import resources
from thor_mcp.rate_limit import RateLimitExceeded


class ToolCallRequest(BaseModel):
    arguments: Dict[str, Any] = {}


def create_app(server: Any) -> FastAPI:
    """Build the FastAPI app wrapping a ThorMCPServer instance."""
    app = FastAPI(title="ThorMCP REST Bridge", version="0.1.0")

    @app.exception_handler(PermissionError)
    async def _permission_error(request: Request, exc: PermissionError):
        return JSONResponse(status_code=401, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "server": "thor-mcp"}

    @app.post("/auth/token")
    async def issue_token(request: Request) -> Dict[str, Any]:
        raw = await request.body()
        body = json.loads(raw) if raw else {}
        token = server.auth.issue_token(subject=body.get("subject", "thor-client"))
        return {"token": token, "expires_in": 3600}

    @app.get("/tools")
    async def list_tools() -> Dict[str, Any]:
        from thor_mcp.tools import SPECS

        return {"tools": [
            {"name": s["name"], "description": s["description"]} for s in SPECS
        ]}

    @app.post("/tools/{name}")
    async def call_tool(name: str, payload: ToolCallRequest,
                        authorization: Optional[str] = Header(None)) -> Any:
        server.auth.require_token(authorization)
        try:
            await server.limiter.check(f"http:{name}")
        except RateLimitExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        result = await server.invoke(name, payload.arguments)
        if result.isError:
            detail = result.content[0].text if result.content else "tool error"
            raise HTTPException(status_code=400, detail=detail)
        return json.loads(result.content[0].text) if result.content else {}

    @app.get("/resources")
    async def list_resources() -> Dict[str, Any]:
        return {"resources": [
            {"uri": r.uri, "name": r.name} for r in resources.resource_list()
        ]}

    @app.get("/resources/{uri:path}")
    async def read_resource(uri: str,
                            authorization: Optional[str] = Header(None)) -> Any:
        server.auth.require_token(authorization)
        return await server.read(uri)

    return app
