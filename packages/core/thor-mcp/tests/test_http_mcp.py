"""Tests for the MCP streamable-HTTP transport and platform endpoints."""

import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from thor_mcp.client import ThorMCPClient
from thor_mcp.deploy import platform_router
from thor_mcp.http_mcp import create_streamable_http_app
from thor_mcp.server import ThorMCPServer

PORT = 8791


@pytest.fixture(scope="module")
def http_server():
    server = ThorMCPServer(force_memory=True, log_level="WARNING")
    app = create_streamable_http_app(server.server)
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if uvicorn_server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{PORT}/mcp"
    uvicorn_server.should_exit = True
    thread.join(timeout=5)


async def test_streamable_http_end_to_end(http_server):
    async with ThorMCPClient(url=http_server) as client:
        tools = await client.list_tools()
        assert len(tools) == 13

        status = await client.call_tool("hardware_status", {})
        assert status["status"] == "ok"

        result = await client.call_tool("benchmark_run", {
            "model_id": "ultralytics/yolov8n",
            "workload_type": "vision",
            "batch_sizes": [1],
            "iterations": 3,
            "custom_config": {"simulate": True},
        })
        assert result["run_id"].startswith("run-")

        runs = await client.read_resource("thor://benchmarks/results")
        assert runs["count"] == 1


def test_platform_endpoints(http_server):
    base = http_server.rsplit("/", 1)[0]  # http://127.0.0.1:PORT
    import httpx

    with httpx.Client(base_url=base) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        version = client.get("/version").json()
        assert "sha" in version and "built" in version
        assert client.get("/openapi.json").status_code == 200


def test_platform_router_in_fastapi():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(platform_router())
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200
        assert client.get("/version").json()["sha"] == "dev"
