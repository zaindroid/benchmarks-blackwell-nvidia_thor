"""Complete research workflow using ThorMCP.

1. Check hardware status
2. Register a new model
3. Create an optimization profile
4. Run a comprehensive benchmark
5. Compare with baselines
6. Generate a report
7. Track everything as an experiment
"""

import asyncio
import json

from thor_mcp.client import ThorMCPClient

MODEL_ID = "custom/vlm-novel"


async def complete_research_workflow() -> None:
    async with ThorMCPClient(config_path="../../thor-config.yaml") as client:
        print("STEP 1: hardware status")
        status = await client.call_tool("hardware_status", {})
        print(json.dumps(status, indent=2)[:400])

        print("\nSTEP 2: register model")
        await client.call_tool(
            "models_register",
            arguments={
                "model_id": MODEL_ID,
                "source": "custom",
                "architecture": "vision-transformer",
                "parameters": 13000000000,
                "metadata": {"license": "MIT"},
            },
        )

        print("\nSTEP 3: optimization profile")
        optimization = await client.call_tool(
            "models_optimize",
            arguments={
                "model_id": MODEL_ID,
                "optimization_type": "tensorrt",
                "precision": "int8",
                "target_latency_ms": 50,
                "enable_sparsity": True,
            },
        )
        print(f"profile: {optimization['profile_id']} status={optimization['status']}")

        print("\nSTEP 4: comprehensive benchmark")
        benchmark = await client.call_tool(
            "benchmark_run",
            arguments={
                "model_id": MODEL_ID,
                "workload_type": "multimodal",
                "precision": "int8",
                "batch_sizes": [1, 2, 4],
                "iterations": 20,
                "collect_power": True,
                "collect_memory": True,
                "collect_thermal": True,
                "custom_config": {
                    "simulate": True,  # remove on a real Thor device
                    "max_new_tokens": 64,
                },
            },
        )
        print(f"run_id: {benchmark['run_id']}, "
              f"p50={benchmark['results']['latency']['p50_ms']} ms")

        print("\nSTEP 5: comparison")
        comparison = await client.call_tool(
            "benchmark_compare",
            arguments={
                "benchmark_ids": [benchmark["run_id"]],
                "metrics": ["latency_p50", "throughput", "power_watts"],
                "format": "markdown",
            },
        )
        print(comparison["comparison"])

        print("\nSTEP 6: report")
        report = await client.call_tool(
            "reports_generate",
            arguments={"benchmark_id": benchmark["run_id"], "format": "markdown"},
        )
        print(f"report_id: {report['report_id']}")

        print("\nSTEP 7: track experiment")
        experiment = await client.call_tool(
            "experiments_track",
            arguments={
                "name": "Novel VLM benchmark on Thor",
                "hypothesis": "Custom VLM achieves <50ms latency with int8 quantization",
                "config": {"model": MODEL_ID, "optimization": optimization},
                "results": {"benchmark": benchmark, "comparison": comparison},
                "tags": ["vlm", "int8", "tensorrt"],
            },
        )
        print(f"experiment_id: {experiment['experiment']['experiment_id']}")

        print("\nWORKFLOW COMPLETE")


if __name__ == "__main__":
    asyncio.run(complete_research_workflow())
