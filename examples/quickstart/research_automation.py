"""Automated research loop: benchmark new models, compare, track experiments."""

import asyncio

from thor_mcp.client import ThorMCPClient

# Models we want evaluated this session (stand-in for "trending on HF").
CANDIDATES = [
    {"id": "ultralytics/yolov8n", "workload_type": "vision"},
    {"id": "meta-llama/Llama-3-8B", "workload_type": "language"},
    {"id": "microsoft/Phi-3-mini-4k-instruct", "workload_type": "language"},
]


async def auto_benchmark_new_models() -> None:
    async with ThorMCPClient(config_path="../../thor-config.yaml") as client:
        for model in CANDIDATES:
            # Skip models that already have results
            existing = await client.read_resource(
                f"thor://benchmarks/results?model_id={model['id']}"
            )
            if existing.get("count", 0) > 0:
                print(f"Skipping {model['id']} (already benchmarked)")
                continue

            print(f"Benchmarking {model['id']} ...")
            results = await client.call_tool(
                "benchmark_run",
                arguments={
                    "model_id": model["id"],
                    "workload_type": model["workload_type"],
                    "precision": "int8",
                    "batch_sizes": [1, 4],
                    "iterations": 20,
                    "custom_config": {"simulate": True},
                },
            )

            experiment = await client.call_tool(
                "experiments_track",
                arguments={
                    "name": f"Benchmark {model['id']}",
                    "hypothesis": f"{model['id']} achieves competitive performance on Thor",
                    "config": {"model": model["id"], "precision": "int8"},
                    "results": {"benchmark": results},
                    "tags": ["auto-benchmark"],
                },
            )
            print(f"Tracked experiment {experiment['experiment']['experiment_id']}")

    history = await client.read_resource("thor://experiments/history")
    print(f"\nTotal experiments tracked: {history['count']}")


if __name__ == "__main__":
    asyncio.run(auto_benchmark_new_models())
