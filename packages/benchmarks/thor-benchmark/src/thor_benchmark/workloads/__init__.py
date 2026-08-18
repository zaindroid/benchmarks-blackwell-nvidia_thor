"""Workload base class and registry."""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from thor_core.logging import get_logger

logger = get_logger(__name__)


class WorkloadError(RuntimeError):
    """Raised for invalid workload configuration or missing dependencies."""


class Workload(ABC):
    """Base class for benchmark workloads.

    Subclasses implement :meth:`prepare_model` and :meth:`_forward`.
    With ``simulate=True`` (config key) no model is loaded and
    deterministic synthetic latencies are produced, so benchmarks and
    tests run on machines without a Thor device.
    """

    MODELS: Dict[str, Dict[str, Any]] = {}
    TASK: str = "generic"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_id: str = ""
        self.precision: str = "fp16"
        self.simulate = bool(self.config.get("simulate", False))
        self._sim_base_ms = float(self.config.get("simulated_latency_ms", 8.0))
        self._engine: Any = None  # set to a TensorRT execution context when active
        self.backend: str = "torch"

    # -- interface ------------------------------------------------------
    @abstractmethod
    def prepare_model(self, model_id: str, precision: str) -> None:
        """Load (or resolve) the model for benchmarking."""

    @abstractmethod
    def _forward(self, batch_size: int) -> None:
        """Run a single inference pass; used by the default run loop."""

    # -- default run loop ----------------------------------------------
    def run_inference(self, batch_size: int, iterations: int) -> Dict[str, Any]:
        """Run ``iterations`` inferences at ``batch_size``.

        Returns ``{"latencies_ms": [...], "samples": int,
        "tokens_per_second": [...]}``.
        """
        if self.simulate:
            return self._simulate_result(batch_size, iterations)
        latencies = self._timed_loop(lambda: self._forward(batch_size), iterations)
        return {
            "latencies_ms": [round(ms, 3) for ms in latencies],
            "samples": iterations * batch_size,
            "tokens_per_second": [],
        }

    def _timed_loop(self, fn: Callable[[], Any], iterations: int) -> List[float]:
        latencies: List[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            latencies.append((time.perf_counter() - start) * 1000.0)
        return latencies

    # -- helpers --------------------------------------------------------
    def _simulate_latencies(self, iterations: int) -> List[float]:
        rng = random.Random(int(self.config.get("seed", 42)))
        return [
            round(self._sim_base_ms * (1 + rng.uniform(-0.08, 0.08)), 3)
            for _ in range(iterations)
        ]

    def _simulate_result(self, batch_size: int, iterations: int) -> Dict[str, Any]:
        # Sleep proportionally so measured throughput stays realistic.
        time.sleep(iterations * self._sim_base_ms / 1000.0)
        return {
            "latencies_ms": self._simulate_latencies(iterations),
            "samples": iterations * batch_size,
            "tokens_per_second": [],
        }

    def model_info(self) -> Dict[str, Any]:
        spec = self.MODELS.get(self.model_id, {})
        return {
            "name": self.model_id,
            "source": spec.get("source", "custom"),
            "architecture": spec.get("architecture", "unknown"),
            "parameters": spec.get("parameters"),
            "precision": self.precision,
            "input_shape": spec.get("input_size"),
            "task": spec.get("task", self.TASK),
            "backend": self.backend,
        }

    def resolve_model_id(self, model_id: str) -> str:
        """Resolve short aliases (e.g. ``yolov8n``) to registry ids.

        In simulated mode arbitrary custom model ids are allowed so
        registered custom models (e.g. ``custom/vlm-novel``) can be
        benchmarked end-to-end without a model download.
        """
        if model_id in self.MODELS:
            return model_id
        aliases = getattr(self, "ALIASES", {})
        if model_id in aliases:
            return aliases[model_id]
        if self.simulate:
            return model_id
        raise WorkloadError(
            f"Unknown model {model_id!r} for {self.TASK} workload. "
            f"Known models: {sorted(self.MODELS)}"
        )


def create_workload(workload_type: str,
                    config: Optional[Dict[str, Any]] = None) -> Workload:
    """Instantiate a workload by type name.

    Types: ``vision``, ``segmentation``, ``classification``,
    ``language``, ``multimodal``.
    """
    from thor_benchmark.workloads.language.llm import LLMBenchmark
    from thor_benchmark.workloads.multimodal.vlm import VLMBenchmark
    from thor_benchmark.workloads.vision.classification import ClassificationBenchmark
    from thor_benchmark.workloads.vision.detection import DetectionBenchmark
    from thor_benchmark.workloads.vision.segmentation import SegmentationBenchmark

    registry: Dict[str, type] = {
        "vision": DetectionBenchmark,
        "segmentation": SegmentationBenchmark,
        "classification": ClassificationBenchmark,
        "language": LLMBenchmark,
        "multimodal": VLMBenchmark,
    }
    cls = registry.get(workload_type)
    if cls is None:
        raise WorkloadError(
            f"Unknown workload type: {workload_type!r}. "
            f"Available: {sorted(registry)}"
        )
    return cls(config or {})


def list_workload_types() -> List[str]:
    return ["vision", "segmentation", "classification", "language", "multimodal"]
