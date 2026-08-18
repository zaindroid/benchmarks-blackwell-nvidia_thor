"""Tests for the TensorRT engine builder.

The build logic is exercised against a fake ``tensorrt`` module (the
real library requires an NVIDIA driver); ONNX export runs for real with
torch.
"""

import sys
import types
from pathlib import Path

import pytest

from thor_models.optimize.profiles import OptimizeError, optimize_model
from thor_models.optimize.trt_builder import (
    Int8Calibrator,
    build_engine,
    build_engine_from_model,
    export_to_onnx,
)


class FakeTensorRT:
    """Minimal fake of the TensorRT Python API surface used by the builder."""

    class Logger:
        WARNING = "WARNING"

        def __init__(self, level):
            self.level = level

    class Builder:
        platform_has_fast_fp16 = True
        platform_has_fast_int8 = True

        def __init__(self, logger):
            self.logger = logger
            self.networks = []
            self.configs = []
            self.profiles = []

        def create_network(self, flags):
            network = FakeTensorRT.Network(flags)
            self.networks.append(network)
            return network

        def create_builder_config(self):
            config = types.SimpleNamespace()
            config.memory_pool_limits = {}
            config.flags = set()
            config.int8_calibrator = None
            config.profiles = []
            config.set_memory_pool_limit = lambda pool, size: config.__dict__.update(
                memory_pool_limits={pool: size}
            )
            config.set_flag = lambda flag: config.flags.add(flag)
            config.add_optimization_profile = lambda profile: config.profiles.append(profile)
            self.configs.append(config)
            return config

        def create_optimization_profile(self):
            profile = types.SimpleNamespace(set_shape=lambda *a: None,
                                             set_shape_input=lambda *a: None)
            self.profiles.append(profile)
            return profile

        def build_serialized_network(self, network, config):
            if network.parse_ok:
                return b"FAKE_ENGINE_BYTES"
            return None

    class NetworkDefinitionCreationFlag:
        EXPLICIT_BATCH = 1

    class Network:
        def __init__(self, flags):
            self.flags = flags
            self.num_inputs = 1
            self.inputs = [types.SimpleNamespace(name="input", shape=(-1, 3, 640, 640))]
            self.parse_ok = True

        def get_input(self, i):
            return self.inputs[i]

    class OnnxParser:
        def __init__(self, network, logger):
            self.network = network
            self.logger = logger
            self.num_errors = 0

        def parse(self, data):
            return True

        def get_error(self, i):
            return types.SimpleNamespace(message=lambda: "fake error")

    class MemoryPoolType:
        WORKSPACE = "WORKSPACE"

    class BuilderFlag:
        FP16 = "FP16"
        INT8 = "INT8"
        SPARSE_WEIGHTS = "SPARSE_WEIGHTS"

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
    import thor_models.optimize.trt_builder as builder

    monkeypatch.setattr(builder, "_import_trt", lambda: module)
    return module


def _tiny_model(size: int = 32):
    import torch
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Flatten(),
        nn.Linear(8 * size * size, 10),
    )


def test_export_to_onnx(tmp_path):
    import torch

    model = _tiny_model().eval()
    path = tmp_path / "model.onnx"
    result = export_to_onnx(model, torch.randn(1, 3, 32, 32), path)
    assert Path(result).exists()
    assert Path(result).stat().st_size > 0


def test_build_engine_with_fake_trt(fake_trt, tmp_path):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx-bytes")
    result = build_engine(onnx_path, precision="fp16", output_path=tmp_path / "model.plan")
    assert result["backend"] == "tensorrt"
    assert result["precision"] == "fp16"
    assert result["input_batch_range"] == [1, 8, 32]
    assert (tmp_path / "model.plan").read_bytes() == b"FAKE_ENGINE_BYTES"
    assert result["size_bytes"] == len(b"FAKE_ENGINE_BYTES")


def test_build_engine_int8_with_calibrator(fake_trt, tmp_path):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx-bytes")
    calibrator = Int8Calibrator(cache_file=str(tmp_path / "calib.bin"))
    result = build_engine(onnx_path, precision="int8", calibrator=calibrator)
    assert result["precision"] == "int8"


def test_build_engine_bad_batch_range(fake_trt, tmp_path):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx-bytes")
    with pytest.raises(OptimizeError):
        build_engine(onnx_path, batch_range=(8, 1, 32))


def test_build_engine_missing_onnx(fake_trt, tmp_path):
    with pytest.raises(OptimizeError):
        build_engine(tmp_path / "nope.onnx")


def test_build_engine_from_model_end_to_end(fake_trt, tmp_path):
    import torch

    model = _tiny_model().eval()
    model_path = tmp_path / "tiny.pt"
    torch.save(model, model_path)
    result = build_engine_from_model(
        model_path, precision="fp16", input_shape=[1, 3, 32, 32],
        output_dir=tmp_path,
    )
    assert result["engine_path"].endswith(".plan")
    assert (tmp_path / "tiny.onnx").exists()
    assert (tmp_path / "tiny.plan").exists()


def test_optimize_model_tensorrt_requires_model_path():
    with pytest.raises(OptimizeError):
        optimize_model("test/tiny", "tensorrt", precision="fp16", execute=True)


def test_trt_missing_raises(tmp_path):
    # Without the fake trt injected, execution must fail cleanly.
    import thor_models.optimize.trt_builder as builder

    if builder.trt_available():
        pytest.skip("real tensorrt present")
    with pytest.raises(OptimizeError):
        build_engine(tmp_path / "model.onnx")
