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

        # TensorRT engine building is provided by the optimization tooling
        # (thor_models.optimize.trt_builder); torch is the MVP backend.
        if precision in ("fp16", "int8"):
            logger.warning(
                "tensorrt engine build not yet active; running with torch backend",
                precision=precision,
            )

    def _forward(self, batch_size: int) -> None:
        import torch

        spec = self.MODELS[self.model_id]
        input_size = tuple(spec["input_size"])
        tensor = torch.randn(batch_size, 3, *input_size, device=self._device)
        with torch.no_grad():
            if self._engine is not None:  # TensorRT path (Sprint 5+)
                self._run_tensorrt(tensor)
            else:
                self._model(tensor)

    def _run_tensorrt(self, tensor: Any) -> None:  # pragma: no cover
        raise NotImplementedError("TensorRT execution lands with the optimization tooling")
