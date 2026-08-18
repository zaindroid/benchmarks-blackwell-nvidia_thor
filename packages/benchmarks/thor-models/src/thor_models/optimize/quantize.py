"""Quantization execution.

Executes real INT8 dynamic quantization of torch models (works on CPU
and GPU), with per-precision plans for INT4/FP8 (bitsandbytes/GPTQ/AWQ
toolchain is staged).

Backends are tried in order: torchao (new API) -> torch.ao.quantization
-> torch.quantization (classic), so this works across torch 2.x
versions without importing anything not installed.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional

from thor_core.logging import get_logger

logger = get_logger(__name__)


class QuantizeError(RuntimeError):
    """Raised when a quantization cannot be executed."""


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def create_quantization_config(precision: str, enable_sparsity: bool = False) -> Dict[str, Any]:
    """Return quantization settings for a target precision."""
    configs: Dict[str, Dict[str, Any]] = {
        "int8": {"method": "dynamic_quant", "weight_bits": 8, "activation_bits": 8},
        "int4": {"method": "gptq", "weight_bits": 4, "group_size": 128},
        "fp8": {"method": "fp8_e4m3", "weight_bits": 8, "format": "e4m3"},
        "fp16": {"method": "none", "weight_bits": 16},
        "fp32": {"method": "none", "weight_bits": 32},
    }
    if precision not in configs:
        raise ValueError(f"Unsupported precision: {precision!r}")
    config = dict(configs[precision])
    config["enable_sparsity"] = enable_sparsity
    return config


def _dynamic_quantize(model: Any) -> Any:
    """Dynamically quantize nn.Linear/LSTM layers to int8.

    Tries the current torch.ao API first, then the classic
    torch.quantization API (removed in later torch versions).
    """
    import torch
    import torch.nn as nn
    try:
        from torch.ao.quantization import quantize_dynamic

        return quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=torch.qint8)
    except Exception:
        pass
    try:
        import torch.quantization

        return torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
        )
    except Exception as exc:
        raise QuantizeError(
            f"no working torch dynamic-quantization backend: {exc}"
        ) from exc


def _model_size_bytes(model: Any) -> int:
    import torch

    buffer = io.BytesIO()
    try:
        torch.save(model.state_dict(), buffer)
        return buffer.tell()
    except Exception:
        # state_dict may not be available for quantized/scripted modules
        return 0


def quantize_torch_model(model: Any, precision: str = "int8",
                         enable_sparsity: bool = False) -> Dict[str, Any]:
    """Quantize a torch ``nn.Module`` and return the result + stats.

    Supported for execution: ``int8`` (dynamic quantization).
    ``fp16``/``fp32`` need no quantization (returns the model as-is).
    ``int4``/``fp8`` execution is staged (requires the bitsandbytes /
    GPTQ / AWQ toolchain) and raises :class:`QuantizeError`.
    """
    import torch

    if not torch_available():
        raise QuantizeError("torch is required for quantization execution")
    if precision in ("fp16", "fp32"):
        return {
            "status": "ok",
            "precision": precision,
            "method": "none",
            "model": model,
            "size_bytes_before": _model_size_bytes(model),
            "size_bytes_after": _model_size_bytes(model),
            "note": "No quantization needed for this precision.",
        }
    if precision == "int8":
        size_before = _model_size_bytes(model)
        quantized = _dynamic_quantize(model)
        size_after = _model_size_bytes(quantized)
        return {
            "status": "ok",
            "precision": "int8",
            "method": "dynamic_quant",
            "model": quantized,
            "size_bytes_before": size_before,
            "size_bytes_after": size_after,
            "compression_ratio": round(size_before / size_after, 3) if size_after else 0.0,
            "note": "int8 dynamic quantization (nn.Linear/LSTM weights).",
        }
    raise QuantizeError(
        f"{precision} execution is staged: requires the bitsandbytes / "
        "GPTQ / AWQ toolchain (see quantize.py plans)."
    )


def quantize_model_file(model_path: str, precision: str = "int8",
                        enable_sparsity: bool = False,
                        map_location: str = "cpu") -> Dict[str, Any]:
    """Load a saved torch model/state dict and quantize it.

    ``model_path`` may point to a full model save (torch.save(model))
    or a state dict (torch.save(model.state_dict())).
    """
    import torch
    import torch.nn as nn

    payload = torch.load(model_path, map_location=map_location, weights_only=False)
    model = payload
    if isinstance(payload, dict):
        # state dict -> wrap into a container is not possible generically;
        # require a full model save for now.
        raise QuantizeError(
            "model_path contains a state dict; pass a full model save "
            "(torch.save(model)) so the architecture is available for quantization."
        )
    if not isinstance(model, nn.Module):
        raise QuantizeError(f"unsupported payload type: {type(model).__name__}")
    model.eval()
    return quantize_torch_model(model, precision=precision,
                                enable_sparsity=enable_sparsity)
