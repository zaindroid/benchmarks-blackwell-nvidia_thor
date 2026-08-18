"""TensorRT engine builder.

Real implementation: torch -> ONNX export, then a TensorRT engine build
with optimization profiles (min/opt/max batch) and fp16/int8 flags.
INT8 calibration is provided via :class:`Int8Calibrator` (feed device
calibration batches through ``get_batch``).

Requires ``pip install thor-models[tensorrt]`` and an NVIDIA driver.
TensorRT is only importable on devices with the driver installed, so
execution raises a clear error elsewhere; the build logic is covered by
tests against a fake ``tensorrt`` module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from thor_core.logging import get_logger

from thor_models.optimize.profiles import OptimizeError

logger = get_logger(__name__)

DEFAULT_BATCH_RANGE = (1, 8, 32)


def trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


def _import_trt() -> Any:
    try:
        import tensorrt

        return tensorrt
    except ImportError:
        raise OptimizeError(
            "TensorRT is not installed. Install with `pip install "
            "thor-models[tensorrt]` on the Thor device."
        ) from None


def export_to_onnx(model: Any, dummy_input: Any, onnx_path: str | Path,
                   opset: int = 17) -> str:
    """Export a torch module to ONNX with a dynamic batch axis."""
    try:
        import torch
    except ImportError:
        raise OptimizeError("torch is required for ONNX export") from None

    onnx_path = str(onnx_path)
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)) or ".", exist_ok=True)
    # torch 2.x defaults to the dynamo exporter, which dislikes
    # dynamic_axes; use the legacy exporter where available.
    export_kwargs: dict[str, Any] = {}
    if "dynamo" in torch.onnx.export.__code__.co_varnames:
        export_kwargs["dynamo"] = False
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        **export_kwargs,
    )
    logger.info("onnx export complete", path=onnx_path)
    return onnx_path


class Int8Calibrator:
    """TensorRT INT8 calibrator (subclass or feed ``get_batch`` data).

    The base implementation persists/loads the calibration cache; device
    calibration data must be supplied by the caller (automotive datasets
    are device-specific).
    """

    def __init__(self, cache_file: Optional[str] = None):
        self.cache_file = cache_file
        self._cache: Optional[bytes] = None
        self._batch: Any = None

    def set_batch(self, data: Any) -> None:
        self._batch = data

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self) -> Any:
        # Return None when calibration batches are exhausted.
        return self._batch

    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_file and Path(self.cache_file).exists():
            return Path(self.cache_file).read_bytes()
        return self._cache

    def write_calibration_cache(self, cache: bytes) -> None:
        self._cache = cache
        if self.cache_file:
            Path(self.cache_file).write_bytes(cache)


def build_engine(
    onnx_path: str | Path,
    precision: str = "fp16",
    batch_range: tuple[int, int, int] = DEFAULT_BATCH_RANGE,
    workspace_mb: int = 2048,
    enable_sparsity: bool = False,
    output_path: Optional[str | Path] = None,
    calibrator: Optional[Int8Calibrator] = None,
) -> Dict[str, Any]:
    """Build a TensorRT engine from an ONNX file.

    Returns an engine descriptor (path, precision, batch range, size).
    """
    trt = _import_trt()
    onnx_path = str(onnx_path)
    if not Path(onnx_path).exists():
        raise OptimizeError(f"onnx file not found: {onnx_path}")
    min_batch, opt_batch, max_batch = batch_range
    if not (1 <= min_batch <= opt_batch <= max_batch):
        raise OptimizeError("batch_range must satisfy 1 <= min <= opt <= max")

    logger.info("building tensorrt engine", onnx=onnx_path, precision=precision)
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i).message()
                      for i in range(parser.num_errors)][:5]
            raise OptimizeError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            logger.warning("platform lacks fast fp16; fp16 flag still set")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            logger.warning("platform lacks fast int8; int8 flag still set")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator or Int8Calibrator()
    if enable_sparsity:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    # Optimization profile for a dynamic batch axis.
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = list(tensor.shape)
        if shape and shape[0] in (-1, 0):
            shape[0] = 1
        rest = shape[1:]
        profile.set_shape(
            tensor.name,
            (min_batch, *rest),
            (opt_batch, *rest),
            (max_batch, *rest),
        )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise OptimizeError("TensorRT engine build failed (no engine produced)")

    output_path = output_path or Path(onnx_path).with_suffix(".plan")
    Path(output_path).write_bytes(serialized)
    logger.info("engine built", path=str(output_path), size_bytes=len(serialized))
    return {
        "engine_path": str(output_path),
        "precision": precision,
        "backend": "tensorrt",
        "input_batch_range": [min_batch, opt_batch, max_batch],
        "size_bytes": len(serialized),
        "sparse_weights": enable_sparsity,
    }


def default_engine_path(cache_dir: str | Path, model_id: str, precision: str) -> Path:
    """Conventional cache location for a built engine.

    Shared by the ``thor-models optimize`` CLI (default ``--output``) and
    :class:`thor_benchmark.workloads.vision.detection.DetectionBenchmark`'s
    engine auto-discovery, so a build lands exactly where the benchmark
    runner will look for it without any extra wiring.
    """
    slug = model_id.replace("/", "_")
    return Path(cache_dir) / "engines" / f"{slug}_{precision}.plan"


def build_engine_from_model(
    model_path: str | Path,
    precision: str = "fp16",
    input_shape: Optional[List[int]] = None,
    batch_range: tuple[int, int, int] = DEFAULT_BATCH_RANGE,
    enable_sparsity: bool = False,
    output_dir: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Load a torch model save, export to ONNX and build a TRT engine.

    ``output_path`` pins the final ``.plan`` location (e.g. the CLI's
    ``--output``); ``output_dir`` only controls where the intermediate
    ONNX file is written when ``output_path`` is not given.
    """
    try:
        import torch
    except ImportError:
        raise OptimizeError("torch is required for the tensorrt pipeline") from None

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    if not hasattr(model, "eval"):
        raise OptimizeError("model_path must point to a full torch model save")
    model.eval()

    input_shape = input_shape or [1, 3, 640, 640]
    dummy = torch.randn(*input_shape)
    out_dir = Path(output_dir) if output_dir else Path(model_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{Path(model_path).stem}.onnx"
    export_to_onnx(model, dummy, onnx_path)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return build_engine(
        onnx_path,
        precision=precision,
        batch_range=batch_range,
        enable_sparsity=enable_sparsity,
        output_path=output_path,
    )


def load_engine(engine_path: str | Path) -> tuple[Any, Any, Any]:
    """Deserialize a .plan file into (runtime, engine, context)."""
    trt = _import_trt()
    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise OptimizeError(f"failed to deserialize engine: {engine_path}")
    context = engine.create_execution_context()
    return runtime, engine, context
