"""Object detection benchmark for Thor (YOLOv8 / DETR family).

Real inference requires the ``vision`` extra
(``pip install thor-benchmark[vision]``) and a CUDA device; without
them the workload runs in deterministic ``simulate`` mode.
"""

from __future__ import annotations

from typing import Any, Dict

from thor_core.logging import get_logger

from thor_benchmark.workloads import Workload, WorkloadError

logger = get_logger(__name__)


class DetectionBenchmark(Workload):
    """Object detection benchmark for Thor."""

    TASK = "detection"

    ALIASES = {
        "yolov8n": "ultralytics/yolov8n",
        "yolov8s": "ultralytics/yolov8s",
        "yolov8m": "ultralytics/yolov8m",
        "yolov8l": "ultralytics/yolov8l",
        "detr-resnet50": "hustvl/detr-resnet50",
        "rt-detr": "hustvl/rt-detr",
    }

    MODELS: Dict[str, Dict[str, Any]] = {
        "ultralytics/yolov8n": {
            "input_size": [640, 640], "num_classes": 80,
            "source": "ultralytics", "architecture": "cnn", "parameters": 3150000,
        },
        "ultralytics/yolov8s": {
            "input_size": [640, 640], "num_classes": 80,
            "source": "ultralytics", "architecture": "cnn", "parameters": 11100000,
        },
        "ultralytics/yolov8m": {
            "input_size": [640, 640], "num_classes": 80,
            "source": "ultralytics", "architecture": "cnn", "parameters": 25900000,
        },
        "ultralytics/yolov8l": {
            "input_size": [640, 640], "num_classes": 80,
            "source": "ultralytics", "architecture": "cnn", "parameters": 43600000,
        },
        "hustvl/detr-resnet50": {
            "input_size": [800, 800], "num_classes": 91,
            "source": "huggingface", "architecture": "transformer", "parameters": 41000000,
        },
        "hustvl/rt-detr": {
            "input_size": [640, 640], "num_classes": 80,
            "source": "huggingface", "architecture": "transformer", "parameters": 20000000,
        },
    }

    def prepare_model(self, model_id: str, precision: str) -> None:
        self.model_id = self.resolve_model_id(model_id)
        self.precision = precision
        if self.simulate:
            logger.info("simulated detection benchmark",
                        model=self.model_id, precision=precision)
            return

        try:
            import torch  # noqa: F401
        except ImportError:
            raise WorkloadError(
                "Detection benchmarks require torch. Install with "
                "`pip install thor-benchmark[vision]` or run with --simulate."
            ) from None

        spec = self.MODELS[self.model_id]
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("preparing detection model", model=self.model_id,
                    precision=precision, device=self._device)

        requested_backend = self.config.get("backend", "auto")
        if requested_backend != "torch" and precision in ("fp16", "int8"):
            if self._try_load_engine(requested_backend):
                return
            if requested_backend == "tensorrt":
                cache_dir = self.config.get("cache_dir", "/data/cache")
                from thor_models.optimize.trt_builder import default_engine_path

                raise WorkloadError(
                    f"backend=tensorrt requested but no cached engine at "
                    f"{default_engine_path(cache_dir, self.model_id, precision)}; "
                    f"build one first with `thor-models optimize --model {self.model_id} "
                    f"--precision {precision}`"
                )
            # auto + no cached engine: fall through to the torch backend below.

        if self.model_id.startswith("ultralytics/"):
            try:
                from ultralytics import YOLO

                self._model = YOLO(self.model_id.removeprefix("ultralytics/") + ".pt")
            except ImportError:
                raise WorkloadError(
                    "ultralytics is required for YOLO benchmarks. "
                    "Install with `pip install thor-benchmark[vision]`."
                ) from None
        else:
            try:
                from transformers import AutoModelForObjectDetection

                self._model = AutoModelForObjectDetection.from_pretrained(
                    self.model_id
                ).to(self._device)
            except ImportError:
                raise WorkloadError(
                    "transformers is required for DETR benchmarks. "
                    "Install with `pip install thor-benchmark[vision]`."
                ) from None

    def _try_load_engine(self, requested_backend: str) -> bool:
        """Load a cached TensorRT engine for (model_id, precision) if one
        exists; returns True and sets ``self.backend = "tensorrt"`` on
        success. No engine, or TensorRT itself unavailable, returns False
        so the caller falls back to (or errors out about) torch."""
        from thor_models.optimize.trt_builder import default_engine_path, load_engine, trt_available

        if not trt_available():
            if requested_backend == "tensorrt":
                logger.warning("backend=tensorrt requested but TensorRT is not installed")
            return False

        cache_dir = self.config.get("cache_dir", "/data/cache")
        engine_path = default_engine_path(cache_dir, self.model_id, self.precision)
        if not engine_path.exists():
            return False

        try:
            self._trt_runtime, self._trt_engine, self._trt_context = load_engine(engine_path)
        except Exception as exc:
            logger.warning("failed to load cached tensorrt engine; falling back to torch",
                            engine_path=str(engine_path), error=str(exc))
            return False

        self._engine = self._trt_context
        self.backend = "tensorrt"
        logger.info("loaded tensorrt engine", engine_path=str(engine_path))
        return True

    def _forward(self, batch_size: int) -> None:
        import torch

        spec = self.MODELS[self.model_id]
        input_size = tuple(spec["input_size"])
        tensor = torch.randn(batch_size, 3, *input_size, device=self._device)
        with torch.no_grad():
            if self._engine is not None:
                self._run_tensorrt(tensor)
            else:
                self._model(tensor)

    def _run_tensorrt(self, tensor: Any) -> None:
        """Run inference through the loaded engine, binding by name.

        Tensor names ("input"/"output") match
        ``thor_models.optimize.trt_builder.export_to_onnx``'s fixed
        naming, so no engine introspection is needed to find them.
        """
        import torch

        context = self._trt_context
        context.set_input_shape("input", tuple(tensor.shape))
        output_shape = tuple(context.get_tensor_shape("output"))
        output = torch.empty(output_shape, dtype=torch.float32, device=self._device)

        context.set_tensor_address("input", tensor.data_ptr())
        context.set_tensor_address("output", output.data_ptr())
        stream = torch.cuda.current_stream(device=self._device).cuda_stream
        context.execute_async_v3(stream)
        torch.cuda.synchronize()
