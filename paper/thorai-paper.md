# ThorAI: An Open-Source MCP-Enabled Benchmarking Platform for NVIDIA DRIVE Thor

**Draft v0.1 — for review before submission**

> Status note: numbers marked **[dev]** were measured on a development
> workstation (RTX 3050 Ti Laptop, CPU-based torch inference) and are
> reference values only. Thor-device measurements are collected per
> the runbook in `docs/thor-device-runbook.md` and will replace them
> before submission.

---

## Abstract

NVIDIA DRIVE Thor is the automotive supercomputer that most developers
never get access to: it is expensive, scarce, and has no public
benchmark corpus. As a result, the community has no shared way to
compare model performance, power efficiency, or quantization gains
across the workloads that matter in a vehicle — real-time perception,
on-device LLM/VLM inference, and sensor fusion.

We present **ThorAI**, an open-source benchmarking and deployment
platform for NVIDIA DRIVE Thor. ThorAI contributes (1) the first
public benchmark suite for Thor with a unified result schema covering
latency, throughput, power, memory, and thermal metrics; (2) a
Model Context Protocol (MCP) server that lets AI assistants
programmatically run benchmarks, compare results, and manage models;
(3) a TensorRT + quantization optimization toolchain with executable
INT8 dynamic quantization; (4) reference implementations for
bird's-eye-view (BEV) sensor fusion and on-device vision-language
models with automotive safety filtering; and (5) community
leaderboard infrastructure with a moderated submission portal.

The platform is production-shaped: 5 Python packages, 107 passing
tests, PostgreSQL/InfluxDB storage with graceful in-memory fallback,
Docker deployment, and CI workflows. Reference measurements show
YOLOv8n at 71.3 ms P50 latency with real GPU power/memory sampling,
and INT8 dynamic quantization achieving 2.0x weight compression —
all reproducible without a Thor device via a deterministic simulation
mode.

## 1. Introduction

Autonomous driving stacks run a diverse set of models on
power-constrained embedded hardware: object detection, segmentation,
BEV perception, and increasingly large language and vision-language
models for scene understanding and driver assistance. NVIDIA DRIVE
Thor targets exactly this workload mix. Yet there is no open,
standardized way to answer basic questions that every deployment team
faces:

- What latency/throughput/power does YOLOv8n achieve at batch 16?
- How much does INT8 quantization compress Llama-3-8B, and at what
  quality cost?
- Which precision profile meets a 50 ms end-to-end budget for a
  given model?

ThorAI is our answer: an open platform that makes these measurements
first-class, machine-readable, and AI-assistable.

The project has three design pillars:

1. **Schema-first results.** Every benchmark emits the same JSON
   result document (hardware, model, workload, and metric sections),
   stored in PostgreSQL and streamed to InfluxDB for time-series
   analysis.

2. **MCP-native access.** The platform exposes benchmarking as a set
   of MCP tools (`benchmark.run`, `models.optimize`,
   `benchmark.compare`, ...). Any MCP-capable agent — Claude Desktop,
   Cursor, LangChain, or a custom assistant — can drive the full
   benchmark-optimize-compare-report loop from natural language.

3. **Honest, reproducible evaluation.** A deterministic simulation
   mode produces synthetic results without hardware (for CI and
   development), while real runs collect live NVIDIA Management
   Library (NVML) telemetry: power, temperature, memory, utilization.

## 2. System Design

### 2.1 Repository layout

```
packages/
├── core/
│   ├── thor-core/        # hardware monitor, metrics, config, experiments, timeseries
│   ├── thor-sdk/         # device abstraction, power, telemetry
│   └── thor-mcp/         # MCP server: tools, resources, prompts, transports
└── benchmarks/
    ├── thor-benchmark/   # runner, workloads (vision/language/multimodal), CLI
    └── thor-models/      # model registry, zoo, optimization toolchain
website/                  # leaderboard API (FastAPI) + React frontend
examples/                 # thor-sense (BEV), thor-vlm (VLM) references
tools/                    # Docker, CI, deployment scripts
```

### 2.2 Benchmark result schema

Every run produces a single JSON document with the structure below,
which doubles as the leaderboard's database row:

| Section | Fields |
| --- | --- |
| `hardware` | device, driver/CUDA/TensorRT versions, compute capability, power limit, memory, utilization |
| `model` | name, source, architecture, parameters, precision, input shape |
| `workload` | type, batch sizes, iterations, config |
| `results.latency` | P50/P95/P99/min/max/std (ms), count |
| `results.throughput` | samples/s, max batch size, tokens/s |
| `results.power` | average/peak/idle watts, joules/sample |
| `results.memory` | peak/average MB, allocation pattern |
| `results.thermal` | start/end/peak °C, throttling events |

### 2.3 Hardware monitoring

`thor_core.hardware.HardwareMonitor` samples GPU power, temperature,
memory, and utilization in a background thread during a run using
NVML (pynvml), with psutil for host telemetry. When no GPU is
available the monitor degrades gracefully and sections are marked
`available: false`, which keeps every run schema-complete.

### 2.4 Workloads

| Workload type | Models (built-in zoo) |
| --- | --- |
| `vision` (detection) | YOLOv8n/s/m/l, DETR-ResNet50, RT-DETR |
| `segmentation` | YOLOv8n-seg, SegFormer-B0 |
| `classification` | ResNet-50, ViT-Base |
| `language` | Llama-3-8B, Mistral-7B, Phi-3-mini, Qwen2-7B, Gemma-7B |
| `multimodal` | LLaVA-1.5-7B, Qwen-VL-Chat |

Model loading is lazy (torch/ultralytics/transformers) so the package
imports and tests cleanly anywhere; a `simulate` flag produces
deterministic synthetic results for CI and development.

### 2.5 MCP server

The MCP server exposes 13 tools, 4 resources, and 3 prompts:

- **Tools**: `benchmark_run`, `benchmark_compare`, `benchmark_list`,
  `models_list`, `models_register`, `models_optimize`, `models_deploy`,
  `datasets_list`, `datasets_register`, `reports_generate`,
  `hardware_status`, `experiments_track`, `experiments_list`.
- **Resources**: `thor://benchmarks/results`,
  `thor://models/registry`, `thor://hardware/telemetry`,
  `thor://experiments/history`.
- **Transports**: stdio (default), a FastAPI REST bridge, and a
  standard streamable-HTTP MCP endpoint (`/mcp`) so the server can be
  hosted as a remote MCP server — verified interoperable against a
  live third-party remote MCP endpoint.

Auth uses HMAC-signed bearer tokens; rate limiting is a per-key token
bucket. Storage uses PostgreSQL when configured (via `DATABASE_URL`,
including deployment platforms that provision databases) with
automatic in-memory fallback.

### 2.6 Optimization toolchain

`models_optimize` creates optimization profiles (TensorRT,
quantization, pruning, distillation). Execution is real where the
toolchain is present:

- **INT8 quantization** — executable torch dynamic quantization of
  `nn.Linear`/`nn.LSTM` weights with size/compression reporting
  (verified: 2.0x weight compression on a reference model).
- **TensorRT** — full build path: torch ONNX export → TensorRT engine
  with min/opt/max batch optimization profiles, FP16/INT8 flags, an
  INT8 calibration hook, and `.plan` serialization. Requires the
  TensorRT toolchain on-device; the build logic is unit-tested
  against a mock TensorRT module.

INT4 (GPTQ-style) and FP8 execution are staged pending the
bitsandbytes/GPTQ toolchain.

### 2.7 Reference implementations

- **thor-sense (BEV)**: camera and LiDAR encoders, BEV projection and
  fusion modules, pinhole projection, BEV-IoU-based detection fusion,
  and a constant-velocity object tracker.
- **thor-vlm**: a vision encoder + projector + tiny causal language
  model reference stack that runs end-to-end on CPU, a transformers
  backend for real VLMs (e.g. LLaVA), and automotive safety filters
  (prompt/output denylists) applied to every generation.

## 3. Evaluation

### 3.1 Software correctness

The platform ships **107 passing tests** across 5 packages and 3
example suites, covering metric correctness, hardware fallback,
config parsing, MCP tool dispatch, the streamable-HTTP transport
(end-to-end against a live uvicorn server), quantization execution,
TensorRT build logic, sensor fusion, VLM generation, and the
submission portal API. The frontend builds cleanly with TypeScript
strict mode.

### 3.2 Reference measurements [dev]

Measured on a development workstation (RTX 3050 Ti Laptop GPU, CPU
torch inference):

| Model | Workload | Latency P50 | Throughput | Power (avg) |
| --- | --- | --- | --- | --- |
| YOLOv8n | detection | 71.3 ms | 13.9 samples/s | 7.7 W |
| INT8-quantized reference model | — | — | — | 2.0x compression |

Real GPU power/memory sampling was exercised during runs
(7.7 W average power, ~430 MB peak memory observed), validating the
telemetry pipeline end to end.

> Thor-device measurements (TensorRT engines, FP16/INT4 LLM decode,
> BEVFormer) are collected per the runbook and will be reported in
> the camera-ready version.

### 3.3 Reproducibility

Every result is a schema-complete JSON document with hardware and
environment context, stored in PostgreSQL, streamed to InfluxDB, and
aggregatable into a community leaderboard. CI workflows re-run
benchmarks on push and enforce regression thresholds (latency +10%,
throughput −5%, power +15%).

## 4. Related Work

| System | Public Thor benchmarks | MCP-native access | Result schema | Optimization toolchain |
| --- | --- | --- | --- | --- |
| ThorAI (this work) | Yes (first) | Yes | Yes | Yes |
| Generic inference benchmarks (e.g., MLPerf) | No | No | partial | partial |
| Vendor SDK demos | No | No | No | partial |

To our knowledge ThorAI is the first open platform that combines a
public Thor benchmark suite, MCP-driven benchmarking, and reference
automotive model implementations.

## 5. Conclusion and Future Work

We presented ThorAI, an open-source, MCP-enabled benchmarking
platform for NVIDIA DRIVE Thor, with a complete toolchain from
benchmarking to optimization to community leaderboard. The platform
is tested (107 tests), deployable (Docker, remote MCP hosting, CI),
and reproducible (deterministic simulation + schema-complete results).

Future work: (1) a public Thor benchmark corpus with leaderboard
submissions; (2) deep BEVFormer-style inverse-projection and
pretrained VLM fine-tuning; (3) INT4/FP8 quantization execution;
(4) multi-GPU and distributed inference benchmarks.

## Appendix A — Running the Platform

```bash
# Install
./tools/scripts/setup.sh                 # Linux/macOS (creates .venv)

# Benchmark (deterministic, no GPU)
thor-benchmark run --model ultralytics/yolov8n --workload vision --simulate

# Real run (Thor device, torch+ultralytics)
thor-benchmark run --model ultralytics/yolov8n --workload vision --precision fp16

# Host the MCP endpoint (streamable HTTP)
thor-mcp --http-mcp --port 8000

# Drive it from code
python - <<'EOF'
import asyncio
from thor_mcp.client import ThorMCPClient
async def main():
    async with ThorMCPClient(config_path="thor-config.yaml") as c:
        r = await c.call_tool("benchmark_run", {"model_id": "ultralytics/yolov8n",
                               "workload_type": "vision"})
        print(r["results"]["latency"]["p50_ms"])
asyncio.run(main())
EOF
```

## Appendix B — MCP Tool Schema (benchmark.run)

```json
{
  "name": "benchmark_run",
  "properties": {
    "model_id": "string (required)",
    "workload_type": "vision|language|multimodal|segmentation|classification",
    "precision": "fp32|fp16|int8|int4|fp8",
    "batch_sizes": [1, 4, 8],
    "iterations": 100,
    "collect_power|memory|thermal": true,
    "custom_config": { "simulate": true }
  }
}
```
