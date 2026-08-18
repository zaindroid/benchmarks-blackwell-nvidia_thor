"""thor-models command line interface."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from thor_core.config import ThorConfig

app = typer.Typer(help="Model optimization tooling for NVIDIA Thor", no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)


def _parse_int_tuple(raw: str, name: str, length: int) -> tuple[int, ...]:
    try:
        parts = tuple(int(v.strip()) for v in raw.split(","))
    except ValueError:
        raise typer.BadParameter(f"{name} must be comma-separated integers")
    if len(parts) != length:
        raise typer.BadParameter(f"{name} must have exactly {length} comma-separated integers")
    return parts


def _load_torch_model(model_id: str):
    """Load a real torch nn.Module for known model families, for callers
    that don't already have a local ``--model-path`` torch save."""
    if model_id.startswith("ultralytics/"):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise typer.BadParameter(
                "ultralytics is required to load this model without --model-path. "
                "Install with `pip install thor-models[zoo]` or pass --model-path."
            ) from None
        return YOLO(model_id.removeprefix("ultralytics/") + ".pt").model
    raise typer.BadParameter(
        f"Don't know how to load {model_id!r} without --model-path; pass "
        "--model-path to a local torch model save (torch.save(model, path))."
    )


@app.command()
def optimize(
    model: str = typer.Option(..., "--model", "-m", help="Model id (e.g. ultralytics/yolov8n)"),
    precision: str = typer.Option("fp16", "--precision", "-p", help="fp32|fp16|int8"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Engine .plan output path; defaults to the shared engine cache "
             "(thor-config.yaml models.cache_dir) so `thor-benchmark run "
             "--backend tensorrt` can auto-discover it",
    ),
    model_path: Optional[Path] = typer.Option(
        None, "--model-path",
        help="Local torch model save; loaded automatically for known model ids if omitted",
    ),
    input_shape: str = typer.Option("1,3,640,640", "--input-shape", help="batch,channels,h,w"),
    batch_range: str = typer.Option("1,8,32", "--batch-range", help="min,opt,max dynamic batch profile"),
    sparsity: bool = typer.Option(False, "--sparsity", help="Enable TensorRT structured sparsity"),
    platform_config: Optional[Path] = typer.Option(
        None, "--platform-config", help="thor-config.yaml (for models.cache_dir)"
    ),
):
    """Build a TensorRT engine for a model (torch -> ONNX -> .plan)."""
    from thor_models.optimize.trt_builder import (
        OptimizeError,
        build_engine_from_model,
        default_engine_path,
    )

    cfg = ThorConfig.load(platform_config)
    resolved_output = output or default_engine_path(cfg.models.cache_dir, model, precision)
    shape = list(_parse_int_tuple(input_shape, "--input-shape", 4))
    batches = _parse_int_tuple(batch_range, "--batch-range", 3)

    tmp_path: Optional[Path] = None
    path = model_path
    if path is None:
        import torch

        torch_model = _load_torch_model(model)
        torch_model.eval()
        tmp_path = Path(tempfile.mkdtemp()) / "model.pt"
        torch.save(torch_model, tmp_path)
        path = tmp_path

    try:
        result = build_engine_from_model(
            path,
            precision=precision,
            input_shape=shape,
            batch_range=batches,
            enable_sparsity=sparsity,
            output_path=resolved_output,
        )
    except OptimizeError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    table = Table(title=f"TensorRT engine: {model}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for key, value in result.items():
        table.add_row(key, str(value))
    console.print(table)
    console.print(f"\n[green]Engine written to {result['engine_path']}[/green]")


@app.command("list-engines")
def list_engines(
    platform_config: Optional[Path] = typer.Option(
        None, "--platform-config", help="thor-config.yaml (for models.cache_dir)"
    ),
):
    """List cached TensorRT engines (what `--backend tensorrt` can auto-discover)."""
    cfg = ThorConfig.load(platform_config)
    engine_dir = Path(cfg.models.cache_dir) / "engines"
    plans = sorted(engine_dir.glob("*.plan")) if engine_dir.exists() else []

    table = Table(title=f"Cached engines ({engine_dir})")
    table.add_column("File", style="cyan")
    table.add_column("Size", style="green")
    for plan in plans:
        table.add_row(plan.name, f"{plan.stat().st_size / (1 << 20):.1f} MB")
    console.print(table)
    if not plans:
        console.print("[yellow]No cached engines found.[/yellow]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
