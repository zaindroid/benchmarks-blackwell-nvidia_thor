"""ThorMCPClient — async MCP client used by examples and automation.

Connects to a ThorMCP server either over stdio (spawns ``thor-mcp``)
or to any remote streamable-HTTP MCP endpoint (a hosted ThorMCP) ::

    async with ThorMCPClient(config_path="thor-config.yaml") as client:
        status = await client.call_tool("hardware.status", {})

    async with ThorMCPClient(url="https://mcp.example.com/mcp",
                             headers={"Authorization": "Bearer <token>"}) as client:
        tools = await client.list_tools()
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


class ThorMCPClient:
    """Programmatic MCP client for the Thor server (stdio or remote HTTP)."""

    def __init__(self, command: str = "thor-mcp",
                 args: Optional[List[str]] = None,
                 config_path: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None,
                 url: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None):
        self._url = url
        self._headers = headers
        if url is None and args is None:
            args = ["--stdio"]
            if config_path:
                args = ["--config", config_path]
        self._params = StdioServerParameters(command=command, args=args or [], env=env)
        self._client_ctx = None
        self._session: Optional[ClientSession] = None

    async def __aenter__(self) -> "ThorMCPClient":
        if self._url:
            self._client_ctx = streamablehttp_client(self._url, headers=self._headers)
        else:
            self._client_ctx = stdio_client(self._params)
        yielded = await self._client_ctx.__aenter__()
        # stdio yields (read, write); streamable-http yields (read, write, session_id)
        read, write = yielded[0], yielded[1]
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
