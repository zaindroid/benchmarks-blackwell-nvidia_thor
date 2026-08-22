"""Tests for remote benchmark dispatch (Thor device worker)."""

import pytest

from thor_mcp.remote import RemoteDeviceError, RemoteDeviceRunner


class StubMCPClient:
    """Drop-in for ThorMCPClient that records the call and returns canned data."""

    def __init__(self, url, headers=None, result=None):
        self.url = url
        self.headers = headers
        self.result = result or {"status": "success", "run_id": "run-remote",
                                 "results": {"latency": {"p50_ms": 1.0}}}
        self.called = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def call_tool(self, name, arguments):
        self.called = (name, arguments)
        return self.result


async def test_remote_runner_dispatches_benchmark_run(monkeypatch):
    import thor_mcp.client as client_mod

    stub = StubMCPClient("http://device/mcp")
    monkeypatch.setattr(client_mod, "ThorMCPClient",
                        lambda url, headers=None: stub)

    runner = RemoteDeviceRunner("http://device/mcp", token="tok")
    result = await runner.run_benchmark(
        model_id="ultralytics/yolov8n",
        workload_type="vision",
        precision="fp16",
        batch_sizes=[1, 4],
        iterations=100,
        custom_config={},
    )
    assert result["run_id"] == "run-remote"
    name, args = stub.called
    assert name == "benchmark_run"
    assert args["model_id"] == "ultralytics/yolov8n"
    assert args["batch_sizes"] == [1, 4]
    assert "warmup_iterations" not in args  # None values are dropped


async def test_remote_runner_requires_url():
    with pytest.raises(RemoteDeviceError):
        RemoteDeviceRunner("")


async def test_benchmark_run_handler_uses_remote_device(monkeypatch):
    """A real (non-simulated) run with remote_device enabled dispatches."""
    from thor_core.config import RemoteDeviceConfig, ThorConfig

    from thor_mcp.tools.benchmark import benchmark_run

    saved = []

    class FakeLimiter:
        async def check(self, name):
            return None

    class FakeStore:
        async def save_run(self, data):
            saved.append(data)

        async def is_persisted(self):
            return True

    class FakeRegistry:
        def update_best_metrics(self, *a, **k):
            return None

    class FakeCtx:
        limiter = FakeLimiter()
        config = ThorConfig(
            remote_device=RemoteDeviceConfig(enabled=True, url="http://device/mcp")
        )
        store = FakeStore()
        registry = FakeRegistry()

    async def fake_run_benchmark(**kwargs):
        return {"status": "success", "run_id": "run-device",
                "results": {"latency": {"p50_ms": 3.3},
                            "throughput": {"samples_per_second": 200.0}}}

    import thor_mcp.remote as remote_mod

    monkeypatch.setattr(
        remote_mod, "RemoteDeviceRunner",
        lambda url, token: type("R", (), {"run_benchmark": staticmethod(fake_run_benchmark)})(),
    )

    out = await benchmark_run(
        {"model_id": "ultralytics/yolov8n", "workload_type": "vision",
         "custom_config": {}},
        FakeCtx(),
    )
    assert out["device"] == "remote"
    assert out["run_id"] == "run-device"
    assert saved and saved[0]["run_id"] == "run-device"


async def test_benchmark_run_simulate_stays_local(monkeypatch):
    """Simulated runs never dispatch to the remote device."""
    from thor_core.config import RemoteDeviceConfig, ThorConfig

    from thor_mcp.tools.benchmark import benchmark_run

    dispatched = []

    class FakeLimiter:
        async def check(self, name):
            return None

    class FakeStore:
        async def save_run(self, data):
            return None

        async def is_persisted(self):
            return False

    class FakeRegistry:
        def update_best_metrics(self, *a, **k):
            return None

    class FakeResult:
        def to_dict(self):
            return {"run_id": "run-local", "simulated": True,
                    "hardware": {}, "model": {}, "workload": {}, "results": {}}

    class FakeRunner:
        def run(self, **kwargs):
            return FakeResult()

    class FakeCtx:
        limiter = FakeLimiter()
        config = ThorConfig(
            remote_device=RemoteDeviceConfig(enabled=True, url="http://device/mcp")
        )
        store = FakeStore()
        registry = FakeRegistry()
        runner = FakeRunner()

    import thor_mcp.remote as remote_mod

    def fake_runner(url, token):
        dispatched.append(url)
        raise AssertionError("should not dispatch for simulated run")

    monkeypatch.setattr(remote_mod, "RemoteDeviceRunner", fake_runner)

    out = await benchmark_run(
        {"model_id": "ultralytics/yolov8n", "workload_type": "vision",
         "custom_config": {"simulate": True}},
        FakeCtx(),
    )
    assert dispatched == []
    assert out["run_id"] == "run-local"
