"""TensorRT engine builder (staged).

Building an engine requires the ``tensorrt`` extra and a device with a
NVIDIA driver. Until the optimization sprint lands, ``build_engine``
returns a descriptor for the planned engine and raises only when the
toolchain is genuinely missing and execution was requested.
"""

from __future__ import annotations

from typing import Any, Dict

from thor_core.logging import get_logger

from thor_models.optimize.profiles import OptimizeError

logger = get_logger(__name__)


def trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


def build_engine(model_id: str, precision: str,
                 enable_sparsity: bool = False) -> Dict[str, Any]:
    """Build (or describe) a TensorRT engine for a model.

    Returns an engine descriptor. Full engine serialization (plan files)
    is implemented in the optimization sprint; a descriptor keeps the
    MCP toolchain functional in the MVP.
    """
    if not trt_available():
        raise OptimizeError(
            "TensorRT is not installed. Install with `pip install "
            "thor-models[tensorrt]` on the Thor device."
        )
    logger.info("building tensorrt engine", model_id=model_id, precision=precision,
                sparsity=enable_sparsity)
    # TODO(optimization-sprint): ONNX export -> TensorRT builder -> plan file.
    return {
        "model_id": model_id,
        "precision": precision,
        "enable_sparsity": enable_sparsity,
        "backend": "tensorrt",
        "status": "descriptor",
        "note": "Engine serialization lands in the optimization sprint.",
    }
