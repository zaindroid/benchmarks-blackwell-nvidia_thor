"""Benchmark orchestrator — runs workloads, aggregates metrics, emits results
matching the platform benchmark schema (see docs/api-reference.md)."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from thor_core.config import ThorConfig
from thor_core.hardware import HardwareMonitor, detect_hardware
from thor_core.logging import get_logger
from thor_core.metrics import MetricCollector

from thor_benchmark.workloads import WorkloadError, create_workload, list_workload_types

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """A single benchmark run, serializable to the platform schema."""

    run_id: str
    timestamp: str
    hardware: Dict[str, Any]
    model: Dict[str, Any]
    workload: Dict[str, Any]
    results: Dict[str, Any]
    simulated: bool = False
    git_commit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """Orchestrates benchmark runs across workloads and batch sizes."""

    def __init__(self, config: Optional[ThorConfig] = None):
        self.config = config or ThorConfig()

    def run(
        self,
        model_id: str,
        workload_type: str = "vision",
        precision: str = "fp16",
        batch_sizes: Optional[List[int]] = None,
        iterations: Optional[int] = None,
        warmup_iterations: Optional[int] = None,
        collect_power: bool = True,
        collect_memory: bool = True,
        collect_thermal: bool = True,
        custom_config: Optional[Dict[str, Any]] = None,
        simulate: bool = False,
        influx: Optional[Any] = None,
    ) -> BenchmarkResult:
        """Run a benchmark and return a :class:`BenchmarkResult`.

        ``influx`` — optional :class:`thor_core.timeseries.TimeSeriesWriter`
        used to write telemetry after the run.
        """
        batch_sizes = batch_sizes or [1, 4, 8]
        iterations = iterations or self.config.benchmarks.default_iterations
        warmup_iterations = warmup_iterations or self.config.benchmarks.default_warmup

        if not batch_sizes or any(b < 1 for b in batch_sizes):
            raise WorkloadError("batch_sizes must be a non-empty list of positive ints")
        if iterations < 1 or warmup_iterations < 0:
            raise WorkloadError("iterations must be >= 1 and warmup_iterations >= 0")

        workload_cfg: Dict[str, Any] = {"simulate": simulate}
        workload_cfg.update(custom_config or {})
        workload = create_workload(workload_type, workload_cfg)
        workload.prepare_model(model_id, precision)

        collect_any = collect_power or collect_memory or collect_thermal
        monitor = HardwareMonitor(cuda_device=self.config.hardware.cuda_device)
        if collect_any:
            monitor.start()

        collector = MetricCollector()
        total_elapsed = 0.0
        total_samples = 0

        try:
            for batch_size in batch_sizes:
                logger.info("benchmarking batch size", batch_size=batch_size,
                            iterations=iterations)
                if warmup_iterations:
                    workload.run_inference(batch_size, warmup_iterations)

                start = time.perf_counter()
                out = workload.run_inference(batch_size, iterations)
                elapsed = time.perf_counter() - start
                total_elapsed += elapsed
                total_samples += out["samples"]

                collector.add_latency(out["latencies_ms"])
                sps = out["samples"] / elapsed if elapsed > 0 else 0.0
                collector.add_throughput(sps, batch_size, out["samples"])
                if out.get("tokens_per_second"):
                    collector.add_tokens_per_second(out["tokens_per_second"])
        finally:
            if collect_any:
                monitor.stop()

        hw_stats = monitor.get_stats()
        if collect_power:
            collector.add_power(hw_stats["power"], total_elapsed)
        if collect_memory:
            collector.add_memory(hw_stats["memory"])
        if collect_thermal:
            collector.add_thermal(hw_stats["thermal"])

        hw = detect_hardware(self.config.hardware.cuda_device).to_dict()
        hw.update(hw_stats["hardware"])
        hw["elapsed_s"] = round(total_elapsed, 2)

        result = BenchmarkResult(
            run_id=f"run-{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            hardware=hw,
            model=workload.model_info(),
            workload={
                "type": workload_type,
                "config": workload_cfg,
                "batch_sizes": batch_sizes,
                "iterations": iterations,
                "warmup_iterations": warmup_iterations,
            },
            results=collector.to_dict(),
            simulated=simulate,
        )

        if influx is not None:
            influx.write_run(result.to_dict())

        return result

    def run_from_config(self, config_path: str | Path,
                        simulate: bool = False) -> BenchmarkResult:
        """Run a benchmark defined in a YAML config file (configs/*.yaml)."""
        data: Dict[str, Any] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        model = data.get("model", {})
        workload = data.get("workload", {})
        optimization = data.get("optimization", {})
        return self.run(
            model_id=model.get("id", "ultralytics/yolov8n"),
            workload_type=workload.get("type", "vision"),
            precision=optimization.get("quantization") or optimization.get("precision")
            or "fp16",
            batch_sizes=workload.get("batch_sizes"),
            iterations=workload.get("num_iterations")
            or workload.get("iterations"),
            custom_config={
                k: v for k, v in {
                    "max_new_tokens": workload.get("max_new_tokens"),
                    "prompt_tokens": workload.get("prompt_tokens"),
                    "benchmark_types": workload.get("benchmark_types"),
                }.items() if v is not None
            },
            simulate=simulate,
        )

    @staticmethod
    def list_workloads() -> List[str]:
        return list_workload_types()

