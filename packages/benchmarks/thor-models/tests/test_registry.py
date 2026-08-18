"""Tests for thor_models registry and optimization profiles."""

import pytest

from thor_models.optimize.profiles import OptimizeError, optimize_model
from thor_models.registry import ModelRegistry
from thor_models.zoo import BUILTIN_ZOO


def test_registry_register_and_get():
    registry = ModelRegistry()
    entry = registry.register(
        model_id="custom/vlm-novel",
        architecture="vision-transformer",
        parameters=13000000000,
        source="custom",
    )
    assert entry["model_id"] == "custom/vlm-novel"
    assert registry.get("custom/vlm-novel")["source"] == "custom"
    assert registry.get("missing") is None


def test_registry_seed_zoo():
    registry = ModelRegistry()
    count = registry.seed_zoo(BUILTIN_ZOO)
    assert count == len(BUILTIN_ZOO)
    assert registry.get("ultralytics/yolov8n") is not None
    assert registry.get("meta-llama/Llama-3-8B") is not None
    # seeding again registers nothing new
    assert registry.seed_zoo(BUILTIN_ZOO) == 0


def test_registry_update_best_metrics():
    registry = ModelRegistry()
    registry.seed_zoo(BUILTIN_ZOO)
    registry.update_best_metrics("ultralytics/yolov8n", {"latency_p50": 3.2})
    assert registry.get("ultralytics/yolov8n")["best_metrics"]["latency_p50"] == 3.2
    assert registry.get("ultralytics/yolov8n")["last_benchmarked"] is not None


def test_registry_list_filters():
    registry = ModelRegistry()
    registry.seed_zoo(BUILTIN_ZOO)
    cnn = registry.list(architecture="cnn")
    assert all(m["architecture"] == "cnn" for m in cnn)
    assert len(cnn) >= 5


def test_optimize_profile_created():
    result = optimize_model(
        model_id="ultralytics/yolov8n",
        optimization_type="tensorrt",
        precision="int8",
        target_latency_ms=5.0,
    )
    assert result["profile_id"].startswith("opt-")
    assert result["status"] == "planned"
    assert result["targets"]["target_latency_ms"] == 5.0
    assert "note" in result


def test_optimize_invalid_type_raises():
    with pytest.raises(OptimizeError):
        optimize_model("ultralytics/yolov8n", optimization_type="fusion")


def test_optimize_execute_requires_trt():
    from thor_models.optimize.trt_builder import trt_available

    if not trt_available():
        with pytest.raises(OptimizeError):
            optimize_model(
                "ultralytics/yolov8n", "tensorrt", precision="fp16", execute=True
            )
