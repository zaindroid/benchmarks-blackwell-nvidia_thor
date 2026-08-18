"""Vision-language model (VLM) benchmark for Thor.

Real inference requires ``thor-benchmark[multimodal]`` + CUDA; otherwise
the workload runs in deterministic ``simulate`` mode. A full on-device
VLM reference implementation lives in ``examples/thor-vlm``.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from thor_core.logging import get_logger

from thor_benchmark.workloads import Workload, WorkloadError

logger = get_logger(__name__)


class VLMBenchmark(Workload):
    """Vision-language model benchmark for Thor."""

    TASK = "multimodal"

    MODELS: Dict[str, Dict[str, Any]] = {
        "llava-hf/llava-1.5-7b-hf": {
            "input_size": [336, 336], "max_tokens": 512,
            "source": "huggingface", "architecture": "vision-transformer",
            "parameters": 7000000000,
        },
        "Qwen/Qwen-VL-Chat": {
            "input_size": [448, 448], "max_tokens": 512,
            "source": "huggingface", "architecture": "vision-transformer",
            "parameters": 9000000000,
        },
    }

    def prepare_model(self, model_id: str, precision: str) -> None:
        self.model_id = self.resolve_model_id(model_id)
        self.precision = precision
        if self.simulate:
            logger.info("simulated vlm benchmark", model=self.model_id,
                        precision=precision)
            return
        try:
            import torch  # noqa: F401
        except ImportError:
            raise WorkloadError(
                "VLM benchmarks require torch. Install with "
                "`pip install thor-benchmark[multimodal]` or run with --simulate."
            ) from None
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except ImportError:
            raise WorkloadError(
                "transformers is required for VLM benchmarks. "
                "Install with `pip install thor-benchmark[multimodal]`."
            ) from None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id
        ).to(self._device).eval()

    def run_inference(self, batch_size: int, iterations: int) -> Dict[str, Any]:
        max_new_tokens = int(self.config.get("max_new_tokens") or 32)
        if self.simulate:
            time.sleep(iterations * self._sim_base_ms / 1000.0)
            latencies = self._simulate_latencies(iterations)
            tok_ps = [round(max_new_tokens / (ms / 1000.0), 2) for ms in latencies]
            return {
                "latencies_ms": latencies,
                "samples": iterations * batch_size,
                "tokens_per_second": tok_ps,
            }

        latencies = []
        tokens_per_second = []
        for _ in range(iterations):
            start = time.perf_counter()
            self._forward(batch_size, max_new_tokens)
            elapsed = time.perf_counter() - start
            latencies.append(round(elapsed * 1000.0, 3))
            tokens_per_second.append(round(max_new_tokens / elapsed, 2))
        return {
            "latencies_ms": latencies,
            "samples": iterations * batch_size,
            "tokens_per_second": tokens_per_second,
        }

    def _forward(self, batch_size: int, max_new_tokens: int = 32) -> None:
        import torch

        spec = self.MODELS[self.model_id]
        image = torch.randn(3, *spec["input_size"], device=self._device)
        inputs = self._processor(
            images=[image] * batch_size,
            text=["Describe this scene"] * batch_size,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
