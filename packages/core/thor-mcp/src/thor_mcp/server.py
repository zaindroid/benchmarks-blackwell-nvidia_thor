"""ThorMCP server — MCP protocol server plus shared context.

Implements tools, resources and prompts using the low-level ``mcp``
SDK (``Server`` + ``stdio_server``), so it works with any MCP client
(Claude Desktop, Cursor, LangChain, ...). ``--http`` starts the
FastAPI REST bridge instead (see ``thor_mcp.http``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from mcp.server import Server
from mcp.server.lowlevel.server import ReadResourceContents
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    PromptsCapability,
    ResourcesCapability,
    ServerCapabilities,
    TextContent,
    ToolsCapability,
)

from thor_benchmark.runner import BenchmarkRunner
from thor_core.config import ThorConfig
from thor_core.experiments import ExperimentStore, ExperimentTracker
from thor_core.logging import configure_logging, get_logger
from thor_models.registry import ModelRegistry
from thor_models.zoo import BUILTIN_ZOO
from thor_sdk.device import ThorDevice

from thor_mcp import resources
from thor_mcp.auth import ThorAuth
from thor_mcp.rate_limit import RateLimiter
from thor_mcp.storage import BenchmarkStore
from thor_mcp.tools import build_tools, dispatch

logger = get_logger(__name__)


@dataclass
class ThorContext:
    """Shared state handed to tool/resource handlers."""

    config: ThorConfig
    store: BenchmarkStore
    registry: ModelRegistry
    experiments: ExperimentTracker
    device: ThorDevice
    runner: BenchmarkRunner
    limiter: RateLimiter
    auth: ThorAuth


_PROMPTS: Dict[str, Dict[str, Any]] = {
    "benchmark-new-model": {
        "description": "Template for benchmarking a new model",
        "arguments": [
            PromptArgument(name="model_id", description="Model identifier", required=True),
            PromptArgument(name="precision", description="Target precision", required=False),
        ],
        "template": "Please benchmark {model_id} on Thor with precision {precision}.",
    },
    "optimize-for-thor": {
        "description": "Optimize a model for Thor deployment",
        "arguments": [
            PromptArgument(name="model_id", description="Model identifier", required=True),
            PromptArgument(name="optimization_type", description="tensorrt|quantization|pruning|distillation", required=False),
        ],
        "template": "Optimize {model_id} for Thor using {optimization_type}.",
    },
    "generate-report": {
        "description": "Generate a report for a benchmark run",
        "arguments": [
            PromptArgument(name="run_id", description="Benchmark run id", required=True),
            PromptArgument(name="format", description="markdown|json", required=False),
        ],
        "template": "Generate a {format} report for benchmark run {run_id}.",
    },
}


class ThorMCPServer:
    """MCP server exposing Thor benchmarking/model tools, resources and prompts."""

    def __init__(self, config_path: Optional[str] = None,
                 force_memory: bool = False, log_level: str = "INFO"):
        configure_logging(log_level)
        self.config = ThorConfig.load(config_path)
        # Deployment: MCP_SECRET_KEY comes from the environment. Per the
        # hosting contract we fail loudly in production instead of
        # defaulting silently.
        secret = os.getenv("MCP_SECRET_KEY")
        if secret is None and os.getenv("APP_ENV"):
            raise RuntimeError(
                "MCP_SECRET_KEY is required in production (set via environment)"
            )
        self.auth = ThorAuth(secret or self.config.server.secret_key)
        self.limiter = RateLimiter(self.config.rate_limit_rpm)
        self.store = BenchmarkStore(self.config, force_memory=force_memory)
        self.registry = ModelRegistry()
        self.registry.seed_zoo(BUILTIN_ZOO)
        self.experiments = ExperimentTracker(store=ExperimentStore())
        self.device = ThorDevice(self.config.hardware)
        self.runner = BenchmarkRunner(self.config)
        self.ctx = ThorContext(
            config=self.config,
            store=self.store,
            registry=self.registry,
            experiments=self.experiments,
            device=self.device,
            runner=self.runner,
            limiter=self.limiter,
            auth=self.auth,
        )
        self.server = Server("thor-mcp")
        self._register()

    # -- registration ---------------------------------------------------
    def _register(self) -> None:
        self._register_tools()
        self._register_resources()
        self._register_prompts()

    def _register_tools(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            return ListToolsResult(tools=build_tools())

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            return await self.invoke(name, arguments)

    def _register_resources(self) -> None:
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            return ListResourcesResult(resources=resources.resource_list())

        @self.server.read_resource()
        async def read_resource(uri: Any) -> Iterable[ReadResourceContents]:
            payload = await self.read(str(uri))
            return [ReadResourceContents(
                content=json.dumps(payload, indent=2, default=str),
                mime_type="application/json",
            )]

    def _register_prompts(self) -> None:
        @self.server.list_prompts()
        async def list_prompts() -> ListPromptsResult:
            return ListPromptsResult(prompts=[
                Prompt(
                    name=name,
                    description=meta["description"],
                    arguments=meta["arguments"],
                )
                for name, meta in _PROMPTS.items()
            ])

        @self.server.get_prompt()
        async def get_prompt(name: str,
                             arguments: Optional[Dict[str, Any]] = None) -> GetPromptResult:
            meta = _PROMPTS.get(name)
            if meta is None:
                raise ValueError(f"Unknown prompt: {name}")
            text = meta["template"].format(**(arguments or {}))
            return GetPromptResult(messages=[
                PromptMessage(role="user",
                              content=TextContent(type="text", text=text))
            ])

    # -- public API (used by the HTTP bridge, tests and the session) ----
    async def invoke(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Call a tool handler; returns an MCP CallToolResult."""
        logger.info("tool called", tool=name)
        try:
            result = await dispatch(name, arguments, self.ctx)
            return CallToolResult(content=[TextContent(
                type="text", text=json.dumps(result, indent=2, default=str)
            )])
        except Exception as exc:
            logger.warning("tool failed", tool=name, error=str(exc))
            return CallToolResult(isError=True, content=[TextContent(
                type="text", text=f"Error: {exc}"
            )])

    async def read(self, uri: str) -> Dict[str, Any]:
        """Resolve a thor:// resource URI to its JSON payload."""
        return await resources.read_resource(uri, self.ctx)

    # -- transports -----------------------------------------------------
    async def run_stdio(self) -> None:
        """Serve MCP over stdio."""
        logger.info("starting thor-mcp (stdio)")
        async with stdio_server() as (read, write):
            await self.server.run(
                read,
                write,
                InitializationOptions(
                    server_name="thor-mcp",
                    server_version="0.1.0",
                    capabilities=ServerCapabilities(
                        tools=ToolsCapability(listChanged=False),
                        resources=ResourcesCapability(subscribe=False, listChanged=False),
                        prompts=PromptsCapability(listChanged=False),
                    ),
                ),
            )

    def run_http(self, host: str = "0.0.0.0", port: int = 3000) -> None:
        """Serve the FastAPI REST bridge (see ``thor_mcp.http``)."""
        import uvicorn

        from thor_mcp.http import create_app

        uvicorn.run(create_app(self), host=host, port=port)

    def run_streamable_http(self, host: str = "0.0.0.0", port: int = 8000,
                            allow_origins: list[str] | None = None) -> None:
        """Serve the MCP streamable-HTTP transport (see ``thor_mcp.http_mcp``).

        The MCP endpoint is at ``http://<host>:<port>/mcp`` — a standard
        streamable-HTTP MCP server address, usable from any MCP client.
        """
        import uvicorn

        from thor_mcp.http_mcp import create_streamable_http_app

        uvicorn.run(
            create_streamable_http_app(
                self.server,
                allow_origins=allow_origins,
            ),
            host=host,
            port=port,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Thor MCP Server")
    parser.add_argument("--config", help="Path to config file", default=None)
    parser.add_argument("--stdio", action="store_true", help="Run in stdio mode (default)")
    parser.add_argument("--http", action="store_true",
                        help="Run in HTTP mode (REST JSON bridge)")
    parser.add_argument("--http-mcp", action="store_true",
                        help="Run the MCP streamable-HTTP transport (endpoint at /mcp)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host")
    parser.add_argument("--port", type=int, default=3000, help="HTTP port")
    parser.add_argument("--allow-origins", default=None,
                        help="Comma-separated CORS origins for --http-mcp")
    parser.add_argument("--memory", action="store_true", help="Force in-memory storage")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    server = ThorMCPServer(
        config_path=args.config,
        force_memory=args.memory,
        log_level=args.log_level,
    )
    if args.http_mcp:
        origins = [o.strip() for o in args.allow_origins.split(",")] if args.allow_origins else None
        server.run_streamable_http(host=args.host, port=args.port, allow_origins=origins)
    elif args.http:
        server.run_http(host=args.host, port=args.port)
    else:
        asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()
