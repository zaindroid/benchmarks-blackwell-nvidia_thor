"""Tests for thor_core.timeseries (InfluxDB schema mapping)."""

import pytest

from thor_core.timeseries import run_to_points, influx_available

pytestmark = pytest.mark.skipif(not influx_available(), reason="influxdb-client required")

SAMPLE_RUN = {
    "run_id": "run-abc123",
    "timestamp": "2026-08-18T00:00:00+00:00",
    "hardware": {
        "device": "NVIDIA DRIVE Thor",
        "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        "gpu_utilization_pct": 42.0,
    },
    "model": {"name": "ultralytics/yolov8n", "precision": "fp16"},
    "workload": {"type": "vision", "batch_sizes": [1, 4]},
    "results": {
        "latency": {"p50_ms": 3.2, "p99_ms": 5.1},
        "throughput": {"samples_per_second": 250.0, "tokens_per_second": 0.0},
        "power": {"average_watts": 120.5},
        "memory": {"average_mb": 512.0, "peak_mb": 700.0},
        "thermal": {"peak_temp_c": 65.0},
    },
}


def test_run_to_points_shape():
    points = run_to_points(SAMPLE_RUN)
    assert len(points) == 1 + 2 + 1  # hardware + 2 batches + system
    measurements = [p._name for p in points]
    assert measurements.count("hardware_metrics") == 1
    assert measurements.count("inference_metrics") == 2
    assert measurements.count("system_metrics") == 1


def test_inference_points_carry_batch_tags():
    points = run_to_points(SAMPLE_RUN)
    inference = [p for p in points if p._name == "inference_metrics"]
    tags = [dict(p._tags) for p in inference]
    assert {t["batch_size"] for t in tags} == {"1", "4"}
    assert all(t["run_id"] == "run-abc123" for t in tags)
    assert all(t["precision"] == "fp16" for t in tags)


def test_hardware_point_fields():
    points = run_to_points(SAMPLE_RUN)
    hw = [p for p in points if p._name == "hardware_metrics"][0]
    fields = dict(hw._fields)
    assert fields["power_watts"] == 120.5
    assert fields["gpu_temp_c"] == 65.0
    assert fields["memory_used_mb"] == 512.0
    assert fields["gpu_utilization_pct"] == 42.0


def test_minimal_run_still_produces_points():
    run = {
        "run_id": "run-x",
        "model": {"name": "m", "precision": "fp16"},
        "workload": {"batch_sizes": [1]},
        "results": {"latency": {"p50_ms": 1.0}},
    }
    points = run_to_points(run)
    assert len(points) >= 2  # hardware + 1 inference (+ system)


def test_writer_requires_server_but_constructs():
    from thor_core.timeseries import TimeSeriesWriter

    writer = TimeSeriesWriter(url="http://localhost:8086", token="t",
                              org="o", bucket="b")
    assert writer.bucket == "b"
    writer.close()
