"""thor-benchmark command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from thor_core.config import ThorConfig
from thor_core.hardware import detect_hardware

from thor_benchmark.runner import BenchmarkRunner
from thor_benchmark.workloads import WorkloadError

app = typer.Typer(help="Benchmark orchestrator for NVIDIA DRIVE Thor", no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)


def _parse_batch_sizes(raw: str) -> list[int]:
    try:
        values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    except ValueError:
        raise typer.BadParameter("batch sizes must be comma-separated integers")
    if not values:
        raise typer.BadParameter("batch sizes must not be empty")
    return values


def _load_benchmark_config(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@app.command()
def run(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Benchmark YAML config (configs/*.yaml)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model id (overrides config)"
    ),
    workload: Optional[str] = typer.Option(
        None, "--workload", "-w", help="Workload type: vision|segmentation|classification|language|multimodal"
    ),
    precision: Optional[str] = typer.Option(
        None, "--precision", "-p", help="fp32|fp16|int8|int4|fp8"
    ),
    batch_sizes: str = typer.Option("1,4,8", "--batch-sizes", help="Comma-separated batch sizes"),
    iterations: int = typer.Option(100, "--iterations", min=1),
    warmup: int = typer.Option(10, "--warmup", min=0),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write JSON result to this path"
    ),
    report: Optional[Path] = typer.Option(
        None, "--report", help="Write a markdown report to this path"
    ),
    simulate: bool = typer.Option(
        False, "--simulate", help="Deterministic synthetic run (no GPU required)"
    ),
    influx: bool = typer.Option(
        False, "--influx", help="Write telemetry to InfluxDB (thor-core[timeseries])"
    ),
    platform_config: Optional[Path] = typer.Option(
        None, "--platform-config", help="thor-config.yaml for hardware settings"
    ),
):
    """Run a benchmark on NVIDIA Thor."""
    cfg = ThorConfig.load(platform_config)
    bench_cfg = _load_benchmark_config(config)

    model_id = model or bench_cfg.get("model", {}).get("id")
    workload_type = workload or bench_cfg.get("workload", {}).get("type", "vision")
    precision = precision or bench_cfg.get("optimization", {}).get("quantization") \
        or bench_cfg.get("optimization", {}).get("precision") or "fp16"

    if model_id is None:
        err_console.print("[red]Error:[/red] model id required (--model or --config)")
        raise typer.Exit(1)

    runner = BenchmarkRunner(cfg)
    workload_cfg = bench_cfg.get("workload", {})
    influx_writer = None
    if influx:
        try:
            from thor_core.timeseries import TimeSeriesWriter

            influx_writer = TimeSeriesWriter.from_config(cfg.database.influxdb)
        except Exception as exc:
            err_console.print(f"[red]Error:[/red] --influx unavailable: {exc}")
            raise typer.Exit(1) from exc
    try:
        result = runner.run(
            model_id=model_id,
            workload_type=workload_type,
            precision=precision,
            batch_sizes=_parse_batch_sizes(batch_sizes),
            iterations=iterations,
            warmup_iterations=warmup,
            custom_config={
                k: v for k, v in {
                    "max_new_tokens": workload_cfg.get("max_new_tokens"),
                    "prompt_tokens": workload_cfg.get("prompt_tokens"),
                }.items() if v is not None
            },
            simulate=simulate,
            influx=influx_writer,
        )
    except WorkloadError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    data = result.to_dict()
    _print_summary(data)

    if output:
        Path(output).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        console.print(f"\n[green]Results written to {output}[/green]")
    if report:
        from thor_benchmark.report.generator import write_report

        path = write_report(data, report, fmt="markdown")
        console.print(f"[green]Report written to {path}[/green]")


def _print_summary(data: dict) -> None:
    """Print a compact summary of a benchmark result."""
    results = data.get("results", {})
    table = Table(title=f"Benchmark {data.get('run_id', '')}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("model", str(data.get("model", {}).get("name")))
    table.add_row("workload", str(data.get("workload", {}).get("type")))
    table.add_row("precision", str(data.get("model", {}).get("precision")))
    table.add_row("latency p50 (ms)", str(results.get("latency", {}).get("p50_ms")))
    table.add_row("latency p99 (ms)", str(results.get("latency", {}).get("p99_ms")))
    table.add_row("throughput (samples/s)", str(results.get("throughput", {}).get("samples_per_second")))
    power = results.get("power", {})
    table.add_row("power avg (W)", "-" if power.get("average_watts") is None else str(power.get("average_watts")))
    memory = results.get("memory", {})
    table.add_row("memory peak (MB)", "-" if memory.get("peak_mb") is None else str(memory.get("peak_mb")))
    console.print(table)


@app.command("list-workloads")
def list_workloads() -> None:
    """List available workload types and their models."""
    from thor_benchmark.workloads.language.llm import LLMBenchmark
    from thor_benchmark.workloads.multimodal.vlm import VLMBenchmark
    from thor_benchmark.workloads.vision.classification import ClassificationBenchmark
    from thor_benchmark.workloads.vision.detection import DetectionBenchmark
    from thor_benchmark.workloads.vision.segmentation import SegmentationBenchmark

    table = Table(title="Thor workloads")
    table.add_column("Workload", style="cyan")
    table.add_column("Models", style="green")
    for cls in (DetectionBenchmark, SegmentationBenchmark, ClassificationBenchmark,
                LLMBenchmark, VLMBenchmark):
        models = ", ".join(sorted(cls.MODELS)) or "-"
        table.add_row(cls.TASK, models)
    console.print(table)


@app.command()
def hardware(cuda_device: int = typer.Option(0, "--cuda-device")) -> None:
    """Show detected hardware information."""
    info = detect_hardware(cuda_device).to_dict()
    table = Table(title="Hardware detection")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for key, value in info.items():
        table.add_row(key, "-" if value is None else str(value))
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
