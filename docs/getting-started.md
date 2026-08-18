# Getting Started

## Prerequisites

For the **full** stack (real benchmarks on Thor):

- NVIDIA DRIVE Thor device with SDK installed
- Python 3.10+ (3.12/3.13 verified)
- Docker 24.0+ (for the compose stack)
- CUDA 12.0+, TensorRT 8.6+ (device side)
- PostgreSQL 16+, Redis 7+ (via Docker)

For **development / demos** (no Thor needed):

- Python 3.10+ only

## Installation

### 1. Clone

```bash
git clone https://github.com/yourusername/thor-ai-platform.git
cd thor-ai-platform
```

### 2. Install packages

```bash
# Linux/macOS
./tools/scripts/setup.sh
source .venv/bin/activate

# Windows (git-bash)
python -m venv .venv
source .venv/Scripts/activate
pip install -e packages/core/thor-core
pip install -e packages/core/thor-sdk
pip install -e packages/benchmarks/thor-benchmark
pip install -e packages/benchmarks/thor-models
pip install -e packages/core/thor-mcp
pip install pytest pytest-asyncio
```

Optional extras for real model inference:

```bash
pip install "thor-benchmark[vision]"     # YOLO / DETR / segmentation
pip install "thor-benchmark[language]"   # LLMs (transformers)
pip install "thor-benchmark[multimodal]" # VLMs
pip install "thor-models[tensorrt]"      # TensorRT toolchain
```

### 3. Database (optional — everything falls back to memory)

```bash
docker compose -f tools/docker/docker-compose.yml up -d postgres influxdb redis
```

The migration in `website/database/migrations/001_initial_schema.sql`
is applied automatically by the compose `postgres` service.

### 4. Configuration

```bash
cp thor-config.example.yaml thor-config.yaml   # edit for your device
cp .env.example .env
```

### 5. Verify

```bash
thor-benchmark hardware
thor-benchmark list-workloads
thor-mcp --stdio            # Ctrl+C to stop
```

## Run your first benchmark

```bash
# Synthetic (no GPU)
thor-benchmark run --model ultralytics/yolov8n --workload vision --simulate

# Real (on Thor, with the vision extra installed)
thor-benchmark run --model ultralytics/yolov8n --workload vision --precision fp16
```

Or from Python:

```python
from thor_benchmark import BenchmarkRunner

results = BenchmarkRunner().run(
    model_id="ultralytics/yolov8n",
    workload_type="vision",
    precision="fp16",
    batch_sizes=[1, 4, 8],
    iterations=100,
)
print(f"Latency P50: {results.results['latency']['p50_ms']:.2f} ms")
print(f"Throughput:  {results.results['throughput']['samples_per_second']:.0f} samples/s")
```

## Use it from an AI assistant

See [benchmarking-guide.md](benchmarking-guide.md) for MCP client configs
(Claude Desktop, Cursor) and example prompts.

## Project structure

See [architecture.md](architecture.md).
