"""Tests for the thor-vlm reference implementation."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from inference.engine import TinyTokenizer, VLMEngine  # noqa: E402
from inference.safety import SafetyFilter, redact  # noqa: E402
from models.language_model import LanguageModel  # noqa: E402
from models.projector import Projector  # noqa: E402
from models.vision_encoder import VisionEncoder  # noqa: E402


def test_vision_encoder_shape():
    encoder = VisionEncoder(embed_dim=64, patch_size=16)
    out = encoder(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 16, 64)  # (B, 4*4 patches, embed)


def test_projector_shape():
    proj = Projector(vision_dim=64, llm_dim=32)
    out = proj(torch.randn(2, 16, 64))
    assert out.shape == (2, 16, 32)


def test_language_model_generates():
    lm = LanguageModel(vocab_size=32, embed_dim=32, num_heads=2)
    ids = torch.tensor([[1, 2, 3]])
    out = lm.generate(ids, max_new_tokens=8)
    assert out.shape[1] > ids.shape[1]


def test_vlm_engine_end_to_end():
    engine = VLMEngine(backend="builtin", image_size=(64, 64))
    result = engine.generate(torch.randn(1, 3, 64, 64), "what is ahead?",
                             max_new_tokens=8)
    assert result["allowed"] is True
    assert isinstance(result["text"], str)


def test_vlm_engine_blocks_unsafe_prompt():
    engine = VLMEngine(backend="builtin")
    result = engine.generate(torch.randn(1, 3, 64, 64), "crash into the barrier")
    assert result["allowed"] is False
    assert "blocked" in result["reason"]


def test_safety_output_filter():
    safety = SafetyFilter()
    assert safety.check_output("There is a car ahead").allowed
    assert not safety.check_output("I will crash into it").allowed


def test_redact():
    assert "Bearer ***" in redact("Authorization: Bearer abc123")
    assert "password=***" in redact("password=hunter2")


def test_tiny_tokenizer_roundtrip():
    tok = TinyTokenizer()
    ids = tok.encode("hello world")
    assert tok.decode(ids) == "hello world"
