"""Tests for ThorMCP server tools and resources (in-memory storage, no GPU)."""

import json

import pytest

from thor_mcp.server import ThorMCPServer


@pytest.fixture
def server():
    return ThorMCPServer(force_memory=True, log_level="WARNING")


async def test_hardware_status(server):
    result = await server.invoke("hardware_status", {})
    assert result.isError is False
    data = json.loads(result.content[0].text)
    assert data["status"] == "ok"
    assert data["device"] == "NVIDIA DRIVE Thor"
    assert "gpu" in data and "cpu" in data and "software" in data


async def test_benchmark_run_simulated(server):
    result = await server.invoke("benchmark_run", {
        "model_id": "ultralytics/yolov8n",
        "workload_type": "vision",
        "precision": "fp16",
        "batch_sizes": [1],
        "iterations": 3,
        "custom_config": {"simulate": True},
    })
    assert result.isError is False, result.content[0].text
    data = json.loads(result.content[0].text)
    assert data["status"] == "success"
    assert data["run_id"].startswith("run-")
    assert "latency" in data["results"]


async def test_benchmark_compare(server):
    await server.invoke("benchmark_run", {
        "model_id": "ultralytics/yolov8n", "workload_type": "vision",
        "batch_sizes": [1], "iterations": 3,
        "custom_config": {"simulate": True},
    })
    await server.invoke("benchmark_run", {
        "model_id": "meta-llama/Llama-3-8B", "workload_type": "language",
        "batch_sizes": [1], "iterations": 3,
        "custom_config": {"simulate": True},
    })
    runs = await server.read("thor://benchmarks/results")
    assert runs["count"] == 2

    ids = [r["run_id"] for r in runs["runs"]]
    result = await server.invoke("benchmark_compare", {
        "benchmark_ids": ids,
        "metrics": ["latency_p50", "throughput"],
        "format": "markdown",
    })
    assert result.isError is False
    data = json.loads(result.content[0].text)
    assert "comparison" in data
    assert "| run_id |" in data["comparison"]


async def test_benchmark_compare_unknown_id(server):
    result = await server.invoke("benchmark_compare", {
        "benchmark_ids": ["run-does-not-exist"],
    })
    assert result.isError is True
    assert "unknown benchmark id" in result.content[0].text


async def test_models_tools(server):
    result = await server.invoke("models_register", {
        "model_id": "custom/vlm-novel",
        "architecture": "vision-transformer",
        "parameters": 13000000000,
    })
    assert result.isError is False
    assert json.loads(result.content[0].text)["status"] == "success"

    listed = json.loads((await server.invoke("models_list", {})).content[0].text)
    # registry is seeded with the built-in zoo
    assert listed["count"] >= 15

    opt = await server.invoke("models_optimize", {
        "model_id": "ultralytics/yolov8n",
        "optimization_type": "tensorrt",
        "precision": "int8",
        "target_latency_ms": 5.0,
    })
    assert opt.isError is False
    assert json.loads(opt.content[0].text)["status"] == "planned"


async def test_datasets_and_reports(server):
    ds = await server.invoke("datasets_register", {
        "dataset_id": "coco-2017", "task": "detection", "source": "coco",
    })
    assert json.loads(ds.content[0].text)["status"] == "success"

    run = await server.invoke("benchmark_run", {
        "model_id": "ultralytics/yolov8n", "workload_type": "vision",
        "batch_sizes": [1], "iterations": 3,
        "custom_config": {"simulate": True},
    })
    run_id = json.loads(run.content[0].text)["run_id"]

    report = await server.invoke("reports_generate", {
        "benchmark_id": run_id, "format": "markdown",
    })
    assert report.isError is False
    data = json.loads(report.content[0].text)
    assert data["report_id"].startswith("report-")
    assert "Benchmark Report" in data["content"]


async def test_experiments_track(server):
    result = await server.invoke("experiments_track", {
        "name": "VLM eval",
        "hypothesis": "latency < 50ms",
        "tags": ["vlm", "int8"],
    })
    assert result.isError is False
    data = json.loads(result.content[0].text)
    assert data["experiment"]["experiment_id"] is not None

    history = await server.read("thor://experiments/history")
    assert history["count"] >= 1


async def test_resources(server):
    # resource reads work directly
    registry = await server.read("thor://models/registry")
    assert "models" in registry

    telemetry = await server.read("thor://hardware/telemetry")
    assert "device" in telemetry and "host" in telemetry


async def test_unknown_tool_returns_error(server):
    result = await server.invoke("not_a_tool", {})
    assert result.isError is True
    assert "Unknown tool" in result.content[0].text
