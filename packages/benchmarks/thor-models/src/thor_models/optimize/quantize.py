"""Quantization planning (per-precision configurations).

Quantization *execution* (bitsandbytes / GPTQ / AWQ weight transforms)
is staged for the optimization sprint; this module provides the
configurations used by optimization profiles.
"""

from __future__ import annotations

from typing import Any, Dict


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
