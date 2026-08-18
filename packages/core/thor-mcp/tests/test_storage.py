"""Tests for BenchmarkStore (in-memory backend; no live Postgres needed)."""

from datetime import datetime, timedelta, timezone

import pytest

from thor_mcp.storage import BenchmarkStore


@pytest.fixture
def store():
    return BenchmarkStore()  # config=None -> in-memory backend


async def test_is_persisted_false_without_config(store):
    assert await store.is_persisted() is False


async def test_save_and_get_run(store):
    run = {"run_id": "run-abc123", "timestamp": datetime.now(timezone.utc).isoformat(),
           "model": {"name": "ultralytics/yolov8n"}, "workload": {"type": "vision"},
           "results": {"latency": {"p50_ms": 4.2}}}
    run_id = await store.save_run(run)
    assert run_id == "run-abc123"

    fetched = await store.get_run("run-abc123")
    assert fetched["model"]["name"] == "ultralytics/yolov8n"

    assert await store.get_run("does-not-exist") is None


async def test_list_runs_filters_and_pagination(store):
    for i in range(5):
        await store.save_run({
            "run_id": f"run-{i}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": {"name": "ultralytics/yolov8n"},
            "workload": {"type": "vision"},
            "results": {},
        })
    await store.save_run({
        "run_id": "run-other-model",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {"name": "meta-llama/Llama-3-8B"},
        "workload": {"type": "language"},
        "results": {},
    })

    all_yolo = await store.list_runs(model_id="ultralytics/yolov8n")
    assert len(all_yolo) == 5

    page1 = await store.list_runs(model_id="ultralytics/yolov8n", limit=2, offset=0)
    page2 = await store.list_runs(model_id="ultralytics/yolov8n", limit=2, offset=2)
    assert len(page1) == 2 and len(page2) == 2
    assert {r["run_id"] for r in page1}.isdisjoint({r["run_id"] for r in page2})

    future_since = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    assert await store.list_runs(since=future_since) == []


async def test_save_and_get_experiment(store):
    exp_id = await store.save_experiment(
        "exp-1", config={"lr": 0.01}, results={"accuracy": 0.9}, name="lr sweep",
    )
    assert exp_id == "exp-1"

    fetched = await store.get_experiment("exp-1")
    assert fetched["name"] == "lr sweep"
    assert fetched["config"] == {"lr": 0.01}
    assert fetched["results"] == {"accuracy": 0.9}
    assert fetched["status"] == "completed"

    assert await store.get_experiment("missing") is None


async def test_save_experiment_upserts(store):
    await store.save_experiment("exp-2", config={}, results={}, status="pending")
    await store.save_experiment("exp-2", config={}, results={"accuracy": 0.5}, status="completed")

    fetched = await store.get_experiment("exp-2")
    assert fetched["status"] == "completed"
    assert fetched["results"] == {"accuracy": 0.5}
