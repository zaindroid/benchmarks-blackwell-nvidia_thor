"""Image classification benchmark for Thor.

Real inference requires ``thor-benchmark[vision]`` + CUDA; otherwise the
workload runs in deterministic ``simulate`` mode.
"""

from __future__ import annotations

from typing import Any, Dict

from thor_core.logging import get_logger

from thor_benchmark.workloads import Workload, WorkloadError

logger = get_logger(__name__)


class ClassificationBenchmark(Workload):
    """Image classification benchmark for Thor."""

    TASK = "classification"

    ALIASES = {"resnet50": "timm/resnet50"}

    MODELS: Dict[str, Dict[str, Any]] = {
        "timm/resnet50": {
            "input_size": [224, 224], "num_classes": 1000,
            "source": "timm", "architecture": "cnn", "parameters": 25600000,
        },
        "google/vit-base-patch16-224": {
            "input_size": [224, 224], "num_classes": 1000,
            "source": "huggingface", "architecture": "transformer", "parameters": 86000000,
        },
    }

    def prepare_model(self, model_id: str, precision: str) -> None:
        self.model_id = self.resolve_model_id(model_id)
        self.precision = precision
        if self.simulate:
            logger.info("simulated classification benchmark",
                        model=self.model_id, precision=precision)
            return
        try:
            import torch  # noqa: F401
        except ImportError:
            raise WorkloadError(
                "Classification benchmarks require torch. Install with "
                "`pip install thor-benchmark[vision]` or run with --simulate."
            ) from None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from transformers import AutoModelForImageClassification

            self._model = AutoModelForImageClassification.from_pretrained(
                self.model_id
            ).to(self._device)
        except ImportError:
            raise WorkloadError(
                "transformers is required for classification benchmarks. "
                "Install with `pip install thor-benchmark[vision]`."
            ) from None

    def _forward(self, batch_size: int) -> None:
        import torch

        spec = self.MODELS[self.model_id]
        input_size = tuple(spec["input_size"])
        tensor = torch.randn(batch_size, 3, *input_size, device=self._device)
        with torch.no_grad():
            self._model(tensor)
