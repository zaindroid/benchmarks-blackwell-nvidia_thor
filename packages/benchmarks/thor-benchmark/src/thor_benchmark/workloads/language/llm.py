"""Language model inference benchmark for Thor (prefill/decode).

Real inference requires ``thor-benchmark[language]`` + CUDA; otherwise
the workload runs in deterministic ``simulate`` mode. TensorRT-LLM
engine building is staged with the optimization tooling.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from thor_core.logging import get_logger

from thor_benchmark.workloads import Workload, WorkloadError

logger = get_logger(__name__)


class LLMBenchmark(Workload):
    """LLM inference benchmark for Thor."""

    TASK = "language"

    BENCHMARK_TYPES = ["prefill", "decode", "end_to_end"]

    MODELS: Dict[str, Dict[str, Any]] = {
        "meta-llama/Llama-3-8B": {
            "max_seq_len": 8192, "vocab_size": 128256,
            "source": "huggingface", "architecture": "transformer", "parameters": 8000000000,
        },
        "mistralai/Mistral-7B-v0.1": {
            "max_seq_len": 32768, "vocab_size": 32000,
            "source": "huggingface", "architecture": "transformer", "parameters": 7100000000,
        },
        "microsoft/Phi-3-mini-4k-instruct": {
            "max_seq_len": 4096, "vocab_size": 32064,
            "source": "huggingface", "architecture": "transformer", "parameters": 3800000000,
        },
        "Qwen/Qwen2-7B": {
            "max_seq_len": 32768, "vocab_size": 152064,
            "source": "huggingface", "architecture": "transformer", "parameters": 7600000000,
        },
        "google/gemma-7b": {
            "max_seq_len": 8192, "vocab_size": 256000,
            "source": "huggingface", "architecture": "transformer", "parameters": 8500000000,
        },
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self._tokenizer: Any = None
        self._model: Any = None

    def prepare_model(self, model_id: str, precision: str) -> None:
        self.model_id = self.resolve_model_id(model_id)
        self.precision = precision
        if self.simulate:
            logger.info("simulated llm benchmark", model=self.model_id,
                        precision=precision)
            return
        try:
            import torch  # noqa: F401
        except ImportError:
            raise WorkloadError(
                "LLM benchmarks require torch. Install with "
                "`pip install thor-benchmark[language]` or run with --simulate."
            ) from None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise WorkloadError(
                "transformers is required for LLM benchmarks. "
                "Install with `pip install thor-benchmark[language]`."
            ) from None

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.float16 if precision == "fp16" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self._device).eval()

        if precision in ("int4", "int8", "fp8"):
            logger.warning(
                "tensorrt-llm engine build not yet active; running with torch "
                "backend (bitsandbytes quantization is a follow-up)",
                precision=precision,
            )

    def run_inference(self, batch_size: int, iterations: int) -> Dict[str, Any]:
        """End-to-end: prompt prefill + autoregressive decode."""
        max_new_tokens = int(self.config.get("max_new_tokens") or 32)
        if self.simulate:
            # One synthetic latency per iteration, decode speed proportional to
            # tokens generated.
            time.sleep(iterations * self._sim_base_ms / 1000.0)
            latencies = self._simulate_latencies(iterations)
            tok_ps = [
                round(max_new_tokens / (ms / 1000.0), 2)
                for ms in latencies
            ]
            return {
                "latencies_ms": latencies,
                "samples": iterations * batch_size,
                "tokens_per_second": tok_ps,
            }

        latencies: List[float] = []
        tokens_per_second: List[float] = []
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

        prompt_len = int(self.config.get("prompt_tokens") or 128)
        input_ids = torch.randint(0, 1024, (batch_size, prompt_len), device=self._device)
        with torch.no_grad():
            self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )
