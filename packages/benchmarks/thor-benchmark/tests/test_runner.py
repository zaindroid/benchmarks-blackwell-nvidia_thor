"""End-to-end tests for BenchmarkRunner (simulate mode — no GPU/torch needed)."""

from pathlib import Path

import pytest

from thor_benchmark.report.generator import generate_report, write_report
from thor_benchmark.runner import BenchmarkRunner
from thor_benchmark.workloads import WorkloadError

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_simulated_vision_benchmark_schema():
    runner = BenchmarkRunner()
    result = runner.run(
        model_id="ultralytics/yolov8n",
        workload_type="vision",
        precision="fp16",
        batch_sizes=[1, 4],
        iterations=10,
        warmup_iterations=2,
        simulate=True,
    )
    data = result.to_dict()
    assert data["run_id"].startswith("run-")
    assert data["simulated"] is True
    assert data["model"]["name"] == "ultralytics/yolov8n"
    assert data["model"]["architecture"] == "cnn"
    assert data["results"]["latency"]["count"] == 20
    assert data["results"]["throughput"]["max_batch_size"] == 4
    assert data["results"]["latency"]["p50_ms"] > 0
    # hardware section matches the platform schema
    assert data["hardware"]["device"] == "NVIDIA DRIVE Thor"
    for section in ("power", "memory", "thermal"):
        assert "available" in data["results"][section]


def test_simulated_llm_benchmark():
    runner = BenchmarkRunner()
    result = runner.run(
        model_id="meta-llama/Llama-3-8B",
        workload_type="language",
        precision="int4",
        batch_sizes=[1],
        iterations=5,
        warmup_iterations=0,
        simulate=True,
        custom_config={"max_new_tokens": 16},
    )
    data = result.to_dict()
    assert data["model"]["architecture"] == "transformer"
    assert data["model"]["parameters"] == 8000000000
    assert data["results"]["throughput"]["tokens_per_second"] > 0
    assert data["results"]["throughput"]["max_batch_size"] == 1


def test_unknown_model_raises_when_real():
    # Real mode rejects unknown models; simulate mode allows custom ids.
    runner = BenchmarkRunner()
    with pytest.raises(WorkloadError):
        runner.run(model_id="nonexistent/model", workload_type="vision", simulate=False)


def test_custom_model_allowed_in_simulate():
    runner = BenchmarkRunner()
    result = runner.run(
        model_id="custom/vlm-novel",
        workload_type="multimodal",
        batch_sizes=[1],
        iterations=3,
        warmup_iterations=0,
        simulate=True,
    )
    assert result.to_dict()["model"]["name"] == "custom/vlm-novel"


def test_unknown_workload_raises():
    runner = BenchmarkRunner()
    with pytest.raises(WorkloadError):
        runner.run(model_id="ultralytics/yolov8n", workload_type="quantum", simulate=True)


def test_run_from_config_simulated():
    runner = BenchmarkRunner()
    result = runner.run_from_config(CONFIGS / "yolov8n.yaml", simulate=True)
    data = result.to_dict()
    assert data["model"]["name"] == "ultralytics/yolov8n"
    assert data["workload"]["type"] == "vision"
    # batch sizes from config are kept
    assert data["workload"]["batch_sizes"] == [1, 4, 8, 16, 32]


def test_report_generation(tmp_path):
    runner = BenchmarkRunner()
    result = runner.run(
        model_id="ultralytics/yolov8n",
        workload_type="vision",
        batch_sizes=[1],
        iterations=3,
        warmup_iterations=0,
        simulate=True,
    )
    data = result.to_dict()

    md = generate_report(data, "markdown")
    assert data["run_id"] in md
    assert "## Latency" in md
    assert "## Throughput" in md

    js = generate_report(data, "json")
    assert '"run_id"' in js

    out = write_report(data, tmp_path / "report.md", "markdown")
    assert out.exists()
    assert "Benchmark Report" in out.read_text(encoding="utf-8")
