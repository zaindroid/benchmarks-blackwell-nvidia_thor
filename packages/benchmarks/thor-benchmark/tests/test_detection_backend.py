"""Tests for DetectionBenchmark's TensorRT engine auto-discovery.

Covers the backend-selection logic only (does a cached engine exist? does
an explicit tensorrt request fail loudly without one?). Real CUDA
inference execution (``_run_tensorrt``) needs an actual GPU and can't be
exercised here -- see thor-models/tests/test_trt_builder.py for the same
fake-``tensorrt``-module pattern used to test the build/load side.
"""

from __future__ import annotations

import sys
import types

import pytest

from thor_benchmark.workloads import WorkloadError
from thor_benchmark.workloads.vision.detection import DetectionBenchmark


class FakeTensorRT:
    class Logger:
        WARNING = "WARNING"

        def __init__(self, level):
            self.level = level

    class Runtime:
        def __init__(self, logger):
            self.logger = logger

        def deserialize_cuda_engine(self, blob):
            return types.SimpleNamespace(
                create_execution_context=lambda: types.SimpleNamespace()
            )


@pytest.fixture
def fake_trt(monkeypatch):
    module = FakeTensorRT()
    monkeypatch.setitem(sys.modules, "tensorrt", module)
    return module


def _bench(tmp_path, backend: str = "auto") -> DetectionBenchmark:
    bench = DetectionBenchmark({"backend": backend, "cache_dir": str(tmp_path)})
    bench.model_id = "ultralytics/yolov8n"
    bench.precision = "fp16"
    return bench


def test_try_load_engine_false_without_tensorrt_installed(tmp_path):
    # No fake tensorrt injected -> trt_available() is False in this env.
    bench = _bench(tmp_path)
    assert bench._try_load_engine("auto") is False
    assert bench.backend == "torch"


def test_try_load_engine_false_without_cached_plan(tmp_path, fake_trt):
    bench = _bench(tmp_path)
    assert bench._try_load_engine("auto") is False
    assert bench.backend == "torch"


def test_try_load_engine_succeeds_with_cached_plan(tmp_path, fake_trt):
    from thor_models.optimize.trt_builder import default_engine_path

    engine_path = default_engine_path(str(tmp_path), "ultralytics/yolov8n", "fp16")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(b"FAKE_ENGINE_BYTES")

    bench = _bench(tmp_path)
    assert bench._try_load_engine("auto") is True
    assert bench.backend == "tensorrt"
    assert bench._engine is not None


def test_prepare_model_tensorrt_backend_requires_cached_engine(tmp_path):
    bench = _bench(tmp_path, backend="tensorrt")
    with pytest.raises(WorkloadError, match="no cached engine"):
        bench.prepare_model("ultralytics/yolov8n", "fp16")
