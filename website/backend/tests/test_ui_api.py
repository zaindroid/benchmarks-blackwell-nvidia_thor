"""Tests for the web UI REST API (composed platform app endpoints)."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from platform_app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_tools_lists_mcp_tools(client):
    response = client.get("/api/tools")
    assert response.status_code == 200
    tools = response.json()["tools"]
    names = {t["name"] for t in tools}
    assert "benchmark_run" in names
    assert len(tools) >= 10


def test_api_hardware(client):
    response = client.get("/api/hardware")
    assert response.status_code == 200
    assert "status" in response.json()


def test_api_benchmark_run_simulate(client):
    response = client.post("/api/benchmark/run", json={
        "model_id": "ultralytics/yolov8n",
        "workload_type": "vision",
        "precision": "fp16",
        "batch_sizes": [1, 4],
        "iterations": 5,
        "custom_config": {"simulate": True},
    })
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["run_id"].startswith("run-")
    assert body["simulated"] is True
    assert body["results"]["latency"]["p50_ms"] > 0


def test_api_benchmark_run_requires_model(client):
    response = client.post("/api/benchmark/run", json={"workload_type": "vision"})
    assert response.status_code == 400
