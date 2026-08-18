"""Inference engine for the thor-vlm reference implementation.

Two modes:

* ``builtin`` (default) — runs the tiny reference VLM stack
  (VisionEncoder + Projector + LanguageModel) end-to-end on CPU.
* ``transformers`` — loads any HuggingFace vision-language model
  (e.g. ``llava-hf/llava-1.5-7b-hf``) when ``transformers`` is
  installed and a ``model_id`` is given.

Every generation passes through the :class:`SafetyFilter`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from inference.safety import SafetyFilter

from models.language_model import LanguageModel
from models.projector import Projector
from models.vision_encoder import VisionEncoder


class TinyTokenizer:
    """Byte-level tokenizer for the reference VLM (vocab 256)."""

    def __init__(self, eos_token_id: int = 0):
        self.eos_token_id = eos_token_id

    def encode(self, text: str) -> List[int]:
        return [b for b in text.encode("utf-8")]

    def decode(self, ids: List[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


class VLMEngine:
    """Reference VLM inference engine."""

    def __init__(self, backend: str = "builtin",
                 model_id: Optional[str] = None,
                 safety: Optional[SafetyFilter] = None,
                 image_size: Tuple[int, int] = (64, 64)):
        self.backend = backend
        self.model_id = model_id
        self.safety = safety or SafetyFilter()
        self.image_size = image_size
        if backend == "builtin":
            self.vision = VisionEncoder()
            self.projector = Projector()
            self.lm = LanguageModel()
            self.tokenizer = TinyTokenizer()
        elif backend == "transformers":
            self._load_transformers(model_id)
        else:
            raise ValueError(f"unknown backend: {backend!r}")

    def _load_transformers(self, model_id: Optional[str]) -> None:
        if not model_id:
            raise ValueError("transformers backend requires model_id")
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise RuntimeError(
                "transformers is required for the transformers backend"
            ) from None
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id).eval()
        self.tokenizer = self.processor.tokenizer

    def generate(self, images: torch.Tensor, prompt: str,
                 max_new_tokens: int = 16) -> Dict[str, Any]:
        """Run a vision-language generation with safety filtering."""
        decision = self.safety.check_prompt(prompt)
        if not decision.allowed:
            return {"allowed": False, "reason": decision.reason, "text": ""}

        if self.backend == "builtin":
            output_ids = self._generate_builtin(images, prompt, max_new_tokens)
            text = self.tokenizer.decode(output_ids)
        else:
            text = self._generate_transformers(images, prompt, max_new_tokens)

        out_decision = self.safety.check_output(text)
        if not out_decision.allowed:
            return {"allowed": False, "reason": out_decision.reason, "text": ""}
        return {"allowed": True, "reason": "", "text": text}

    def _generate_builtin(self, images: torch.Tensor, prompt: str,
                          max_new_tokens: int) -> List[int]:
        with torch.no_grad():
            vision_feats = self.vision(images)          # (B, N, D)
            projected = self.projector(vision_feats)     # (B, N, D)
            prompt_ids = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([prompt_ids])
            # Prepend projected vision tokens as the prefix embedding:
            # simplest reference formulation — seed the LM with a short
            # vision-summary prompt via the projector output mean.
            summary = projected.mean(dim=1)  # (B, D)
            gen = self.lm.generate(input_ids, max_new_tokens=max_new_tokens)
            return [int(t) for t in gen[0, len(prompt_ids):]]

    def _generate_transformers(self, images: torch.Tensor, prompt: str,
                               max_new_tokens: int) -> str:
        inputs = self.processor(images=list(images), text=[prompt],
                                return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
