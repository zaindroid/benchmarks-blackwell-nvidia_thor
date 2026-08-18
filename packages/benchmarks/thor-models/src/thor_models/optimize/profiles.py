"""Optimization profiles — plans for TensorRT/quantization/pruning/distillation.

In the MVP, ``optimize_model`` produces a profile (plan) with the
requested targets. Executing the build requires the TensorRT toolchain
(a Thor device with TensorRT installed) and is staged for the
optimization sprint.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from thor_core.logging import get_logger

logger = get_logger(__name__)

VALID_TYPES = ("tensorrt", "quantization", "pruning", "distillation")
VALID_PRECISIONS = ("fp32", "fp16", "int8", "int4", "fp8")


class OptimizeError(RuntimeError):
    """Raised when an optimization cannot be performed."""


@dataclass
class OptimizationProfile:
    """A model optimization profile/plan."""

    profile_id: str
    model_id: str
    optimization_type: str
    precision: str
    config: Dict[str, Any] = field(default_factory=dict)
    targets: Dict[str, Any] = field(default_factory=dict)
    status: str = "planned"  # planned | building | ready | failed
    performance_gain: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "optimization_type": self.optimization_type,
            "precision": self.precision,
            "config": self.config,
            "targets": self.targets,
            "status": self.status,
            "performance_gain": self.performance_gain,
            "created_at": self.created_at,
        }


def create_profile(
    model_id: str,
    optimization_type: str,
    precision: str = "fp16",
    target_latency_ms: Optional[float] = None,
    target_throughput: Optional[float] = None,
    target_memory_mb: Optional[float] = None,
    enable_sparsity: bool = False,
) -> OptimizationProfile:
    """Create an optimization profile for a model."""
    if optimization_type not in VALID_TYPES:
        raise OptimizeError(
            f"optimization_type must be one of {VALID_TYPES}, got {optimization_type!r}"
        )
    if precision not in VALID_PRECISIONS:
        raise OptimizeError(f"precision must be one of {VALID_PRECISIONS}, got {precision!r}")

    targets = {
        "target_latency_ms": target_latency_ms,
        "target_throughput": target_throughput,
        "target_memory_mb": target_memory_mb,
        "enable_sparsity": enable_sparsity,
    }
    config = {
        "optimization_type": optimization_type,
        "precision": precision,
        "enable_sparsity": enable_sparsity,
    }
    return OptimizationProfile(
        profile_id=f"opt-{uuid.uuid4().hex[:12]}",
        model_id=model_id,
        optimization_type=optimization_type,
        precision=precision,
        config=config,
        targets=targets,
    )


def optimize_model(
    model_id: str,
    optimization_type: str,
    precision: str = "fp16",
    target_latency_ms: Optional[float] = None,
    target_throughput: Optional[float] = None,
    target_memory_mb: Optional[float] = None,
    enable_sparsity: bool = False,
    execute: bool = False,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an optimization profile and optionally execute the build.

    ``execute=True`` behaviour:

    * ``quantization`` + ``int8`` — real dynamic quantization via
      torch; requires a local ``model_path`` to a full model save.
    * ``tensorrt`` — requires the TensorRT toolchain (Thor device).
    * other types (pruning/distillation) — staged.
    """
    profile = create_profile(
        model_id=model_id,
        optimization_type=optimization_type,
        precision=precision,
        target_latency_ms=target_latency_ms,
        target_throughput=target_throughput,
        target_memory_mb=target_memory_mb,
        enable_sparsity=enable_sparsity,
    )
    if not execute:
        logger.info("optimization profile created", profile_id=profile.profile_id,
                    model_id=model_id, type=optimization_type)
        data = profile.to_dict()
        data["note"] = (
            "Profile created. Execution requires the relevant toolchain "
            "(TensorRT on the device, torch for int8 quantization)."
        )
        return data

    if optimization_type == "quantization":
        from thor_models.optimize.quantize import QuantizeError, quantize_model_file

        if precision in ("fp16", "fp32"):
            profile.status = "ready"
            profile.performance_gain = {"estimated_speedup": 1.0, "measured": False}
            data = profile.to_dict()
            data["note"] = f"{precision} requires no quantization."
            return data
        if not model_path:
            raise OptimizeError(
                "executing int8 quantization requires model_path to a local "
                "torch model save (torch.save(model))"
            )
        try:
            result = quantize_model_file(
                model_path, precision=precision, enable_sparsity=enable_sparsity
            )
        except QuantizeError as exc:
            raise OptimizeError(str(exc)) from exc
        profile.status = "ready"
        profile.performance_gain = {
            "compression_ratio": result.get("compression_ratio"),
            "size_bytes_before": result.get("size_bytes_before"),
            "size_bytes_after": result.get("size_bytes_after"),
            "measured": False,
        }
        data = profile.to_dict()
        data["quantization"] = {k: v for k, v in result.items() if k != "model"}
        return data

    from thor_models.optimize.trt_builder import OptimizeError as TRTOptimizeError
    from thor_models.optimize.trt_builder import build_engine_from_model

    if not model_path:
        raise OptimizeError(
            "executing tensorrt requires model_path to a torch model save"
        )
    profile.status = "building"
    try:
        engine = build_engine_from_model(
            model_path,
            precision=precision,
            enable_sparsity=enable_sparsity,
        )
    except TRTOptimizeError as exc:
        raise OptimizeError(str(exc)) from exc
    profile.status = "ready"
    profile.performance_gain = {"estimated_speedup": 1.0, "measured": False}
    data = profile.to_dict()
    data["engine"] = engine
    return data
