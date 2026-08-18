"""Built-in model zoo — deployment metadata for supported models."""

from __future__ import annotations

from typing import Any, Dict

from thor_models.zoo.language import LANGUAGE_MODELS
from thor_models.zoo.multimodal import MULTIMODAL_MODELS
from thor_models.zoo.vision import VISION_MODELS

BUILTIN_ZOO: Dict[str, Dict[str, Any]] = {}
BUILTIN_ZOO.update(VISION_MODELS)
BUILTIN_ZOO.update(LANGUAGE_MODELS)
BUILTIN_ZOO.update(MULTIMODAL_MODELS)

__all__ = ["BUILTIN_ZOO", "VISION_MODELS", "LANGUAGE_MODELS", "MULTIMODAL_MODELS"]
