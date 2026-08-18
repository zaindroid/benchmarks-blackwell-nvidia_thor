"""Language model zoo."""

from __future__ import annotations

from typing import Any, Dict

LANGUAGE_MODELS: Dict[str, Dict[str, Any]] = {
    "meta-llama/Llama-3-8B": {
        "name": "Llama-3-8B", "task": "language", "architecture": "transformer",
        "parameters": 8000000000, "source": "huggingface",
        "license": "Llama-3 Community License", "max_seq_len": 8192,
    },
    "mistralai/Mistral-7B-v0.1": {
        "name": "Mistral-7B", "task": "language", "architecture": "transformer",
        "parameters": 7100000000, "source": "huggingface",
        "license": "Apache-2.0", "max_seq_len": 32768,
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "name": "Phi-3-mini", "task": "language", "architecture": "transformer",
        "parameters": 3800000000, "source": "huggingface",
        "license": "MIT", "max_seq_len": 4096,
    },
    "Qwen/Qwen2-7B": {
        "name": "Qwen2-7B", "task": "language", "architecture": "transformer",
        "parameters": 7600000000, "source": "huggingface",
        "license": "Apache-2.0", "max_seq_len": 32768,
    },
    "google/gemma-7b": {
        "name": "Gemma-7B", "task": "language", "architecture": "transformer",
        "parameters": 8500000000, "source": "huggingface",
        "license": "Gemma Terms of Use", "max_seq_len": 8192,
    },
}
