"""Remote benchmark worker dispatch.

Executes real (non-simulated) benchmarks on a remote MCP worker — for
example a DRIVE Thor device reachable over the network or tailnet —
when the local node has no GPU. The worker runs its own ThorMCP server
(``thor-mcp --http-mcp``) with the model runtimes installed; this
module proxies ``benchmark_run`` calls to it and returns the result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from thor_core.logging import get_logger

logger = get_logger(__name__)


class RemoteDeviceError(RuntimeError):
    """Raised when a remote benchmark dispatch fails."""


class RemoteDeviceRunner:
    """Proxy that runs benchmarks on a remote MCP worker."""

    def __init__(self, url: str, token: str = ""):
        if not url:
            raise RemoteDeviceError("remote device url is not configured")
        self.url = url
        self._headers = {"Authorization": f"Bearer {token}"} if token else None

    async def run_benchmark(
        self,
        model_id: str,
        workload_type: str = "vision",
        precision: str = "fp16",
        batch_sizes: Optional[List[int]] = None,
        iterations: Optional[int] = None,
        warmup_iterations: Optional[int] = None,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch a real benchmark to the device worker."""
        try:
            from thor_mcp.client import ThorMCPClient
        except ImportError:  # pragma: no cover - thor-mcp is installed
            raise RemoteDeviceError("thor-mcp client is required for remote dispatch") from None

        arguments: Dict[str, Any] = {
            "model_id": model_id,
            "workload_type": workload_type,
            "precision": precision,
            "batch_sizes": batch_sizes,
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "custom_config": custom_config,
        }
        # drop None so worker defaults apply
        arguments = {k: v for k, v in arguments.items() if v is not None}

        logger.info("dispatching benchmark to remote device",
                    url=self.url, model_id=model_id, workload=workload_type)
        try:
            async with ThorMCPClient(url=self.url, headers=self._headers) as client:
                return await client.call_tool("benchmark_run", arguments)
        except Exception as exc:
            raise RemoteDeviceError(f"remote benchmark failed: {exc}") from exc
