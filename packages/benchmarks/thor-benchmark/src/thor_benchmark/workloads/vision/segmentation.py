"""Semantic segmentation benchmark for Thor.

Real inference requires ``thor-benchmark[vision]`` + CUDA; otherwise the
workload runs in deterministic ``simulate`` mode. A full reference
segmentation pipeline (BEV) lives in ``examples/thor-sense``.
"""

from __future__ import annotations

from typing import Any, Dict

from thor_core.logging import get_logger

from thor_benchmark.workloads import Workload, WorkloadError

logger = get_logger(__name__)


class SegmentationBenchmark(Workload):
    """Semantic segmentation benchmark for Thor."""

    TASK = "segmentation"

    ALIASES = {"yolov8n-seg": "ultralytics/yolov8n-seg"}

    MODELS: Dict[str, Dict[str, Any]] = {
        "ultralytics/yolov8n-seg": {
            "input_size": [640, 640], "num_classes": 80,
            "source": "ultralytics", "architecture": "cnn", "parameters": 3260000,
        },
        "nvidia/segformer-b0-finetuned-ade-512-512": {
            "input_size": [512, 512], "num_classes": 150,
            "source": "huggingface", "architecture": "transformer", "parameters": 3700000,
        },
    }

    def prepare_model(self, model_id: str, precision: str) -> None:
        self.model_id = self.resolve_model_id(model_id)
        self.precision = precision
        if self.simulate:
            logger.info("simulated segmentation benchmark",
                        model=self.model_id, precision=precision)
            return
        try:
            import torch  # noqa: F401
        except ImportError:
            raise WorkloadError(
                "Segmentation benchmarks require torch. Install with "
                "`pip install thor-benchmark[vision]` or run with --simulate."
            ) from None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from transformers import AutoModelForSemanticSegmentation

            self._model = AutoModelForSemanticSegmentation.from_pretrained(
                self.model_id
            ).to(self._device)
        except ImportError:
            raise WorkloadError(
                "transformers is required for segmentation benchmarks. "
                "Install with `pip install thor-benchmark[vision]`."
            ) from None

    def _forward(self, batch_size: int) -> None:
        import torch

        spec = self.MODELS[self.model_id]
        input_size = tuple(spec["input_size"])
        tensor = torch.randn(batch_size, 3, *input_size, device=self._device)
        with torch.no_grad():
            self._model(tensor)
