"""Tests for quantization execution (real torch int8 dynamic quantization)."""

import io

import pytest

from thor_models.optimize.profiles import OptimizeError, optimize_model
from thor_models.optimize.quantize import (
    QuantizeError,
    quantize_model_file,
    quantize_torch_model,
    torch_available,
)

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch required")


def _tiny_model():
    import torch
    import torch.nn as nn

    return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))


def test_int8_dynamic_quantization_executes():
    import torch.nn as nn

    model = _tiny_model()
    result = quantize_torch_model(model, precision="int8")
    assert result["status"] == "ok"
    assert result["method"] == "dynamic_quant"
    quantized = result["model"]
    # Quantization applied: packed-parameter submodules appear and the
    # fp32 payload shrinks (torch 2.13 keeps the Linear class name).
    assert any("PackedParams" in type(m).__name__ for m in quantized.modules())
    assert result["size_bytes_after"] > 0
    # int8 weights should shrink the fp32 payload
    assert result["compression_ratio"] > 1.0


def test_quantize_model_file(tmp_path):
    import torch

    model = _tiny_model()
    path = tmp_path / "tiny.pt"
    torch.save(model, path)
    result = quantize_model_file(str(path), precision="int8")
    assert result["status"] == "ok"
    assert result["size_bytes_after"] < result["size_bytes_before"]


def test_fp16_needs_no_quantization():
    result = quantize_torch_model(_tiny_model(), precision="fp16")
    assert result["status"] == "ok"
    assert result["method"] == "none"


def test_int4_execution_staged():
    with pytest.raises(QuantizeError):
        quantize_torch_model(_tiny_model(), precision="int4")


def test_optimize_model_executes_quantization(tmp_path):
    import torch

    model = _tiny_model()
    path = tmp_path / "tiny.pt"
    torch.save(model, path)

    result = optimize_model(
        model_id="test/tiny",
        optimization_type="quantization",
        precision="int8",
        execute=True,
        model_path=str(path),
    )
    assert result["status"] == "ready"
    assert result["quantization"]["method"] == "dynamic_quant"
    assert result["performance_gain"]["compression_ratio"] > 1.0


def test_optimize_model_quantization_requires_model_path():
    with pytest.raises(OptimizeError):
        optimize_model(
            model_id="test/tiny",
            optimization_type="quantization",
            precision="int8",
            execute=True,
        )
