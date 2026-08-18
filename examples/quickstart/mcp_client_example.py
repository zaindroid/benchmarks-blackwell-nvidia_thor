"""Connect to the ThorMCP server and drive it via MCP.

Spawns ``thor-mcp`` over stdio (must be on PATH, e.g. after
``pip install -e packages/core/thor-mcp``).
"""

import asyncio
import json

from thor_mcp.client import ThorMCPClient


async def main() -> None:
    async with ThorMCPClient(config_path="../../thor-config.yaml") as client:
        # 1. Hardware status
        status = await client.call_tool("hardware_status", {})
        print(f"Hardware status: {status['status']} / gpu_available={status['gpu']['available']}")

        # 2. List available tools
        tools = await client.list_tools()
        print(f"Available tools ({len(tools)}): {tools}")

        # 3. Run a benchmark (simulate for GPU-less machines)
        result = await client.call_tool(
            "benchmark_run",
            arguments={
                "model_id": "ultralytics/yolov8n",
                "workload_type": "vision",
                "precision": "fp16",
                "batch_sizes": [1, 4],
                "iterations": 10,
                "custom_config": {"simulate": True},
            },
        )
        print(f"Benchmark {result['run_id']} p50: "
              f"{result['results']['latency']['p50_ms']} ms")

        # 4. Read stored results
        runs = await client.read_resource("thor://benchmarks/results")
        print(f"Stored runs: {runs['count']}")

        # 5. Compare with another model
        second = await client.call_tool(
            "benchmark_run",
            arguments={
                "model_id": "meta-llama/Llama-3-8B",
                "workload_type": "language",
                "precision": "int4",
                "batch_sizes": [1],
                "iterations": 5,
                "custom_config": {"simulate": True},
            },
        )
        comparison = await client.call_tool(
            "benchmark_compare",
            arguments={
                "benchmark_ids": [result["run_id"], second["run_id"]],
                "metrics": ["latency_p50", "throughput"],
                "format": "markdown",
            },
        )
        print("\nComparison:\n" + comparison["comparison"])


if __name__ == "__main__":
    asyncio.run(main())
