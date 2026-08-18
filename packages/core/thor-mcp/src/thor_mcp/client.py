"""ThorMCPClient — async MCP client used by examples and automation.

Spawns the ``thor-mcp`` server over stdio and wraps the MCP session in
a small typed API::

    async with ThorMCPClient(config_path="thor-config.yaml") as client:
        status = await client.call_tool("hardware.status", {})
        runs = await client.read_resource("thor://benchmarks/results")
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ThorMCPClient:
    """Programmatic MCP client for the Thor server."""

    def __init__(self, command: str = "thor-mcp",
                 args: Optional[List[str]] = None,
                 config_path: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None):
        if args is None:
            args = ["--stdio"]
            if config_path:
                args = ["--config", config_path]
        self._params = StdioServerParameters(command=command, args=args, env=env)
        self._client_ctx = None
        self._session: Optional[ClientSession] = None

    async def __aenter__(self) -> "ThorMCPClient":
        self._client_ctx = stdio_client(self._params)
        read, write = await self._client_ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session is not None:
            await self._session.__aexit__(*exc)
            self._session = None
        if self._client_ctx is not None:
            await self._client_ctx.__aexit__(*exc)
            self._client_ctx = None

    async def call_tool(self, name: str,
                        arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call an MCP tool and return its JSON payload."""
        assert self._session is not None, "use ThorMCPClient() as an async context manager"
        result = await self._session.call_tool(name, arguments or {})
        if result.isError:
            detail = result.content[0].text if result.content else f"tool {name} failed"
            raise RuntimeError(detail)
        return json.loads(result.content[0].text) if result.content else {}

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read an MCP resource (thor://...) and return its JSON payload."""
        assert self._session is not None, "use ThorMCPClient() as an async context manager"
        result = await self._session.read_resource(uri)
        text = result.contents[0].text if result.contents else "{}"
        return json.loads(text)

    query_resource = read_resource

    async def list_tools(self) -> List[str]:
        assert self._session is not None, "use ThorMCPClient() as an async context manager"
        result = await self._session.list_tools()
        return [tool.name for tool in result.tools]

    async def list_resources(self) -> List[str]:
        assert self._session is not None, "use ThorMCPClient() as an async context manager"
        result = await self._session.list_resources()
        return [resource.uri for resource in result.resources]

    async def close(self) -> None:
        await self.__aexit__(None, None, None)
