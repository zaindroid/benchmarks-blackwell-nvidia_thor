"""Multimodal (VLM) model zoo."""

from __future__ import annotations

from typing import Any, Dict

MULTIMODAL_MODELS: Dict[str, Dict[str, Any]] = {
    "llava-hf/llava-1.5-7b-hf": {
        "name": "LLaVA-1.5-7B", "task": "multimodal",
        "architecture": "vision-transformer", "parameters": 7000000000,
        "source": "huggingface", "license": "Apache-2.0",
        "input_size": [336, 336], "max_tokens": 512,
    },
    "Qwen/Qwen-VL-Chat": {
        "name": "Qwen-VL-Chat", "task": "multimodal",
        "architecture": "vision-transformer", "parameters": 9000000000,
        "source": "huggingface", "license": "Qianwen License",
        "input_size": [448, 448], "max_tokens": 512,
    },
}
