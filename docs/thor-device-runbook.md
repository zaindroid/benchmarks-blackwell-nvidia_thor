# NVIDIA DRIVE Thor — Benchmark Runbook

How to collect the first real Thor benchmark numbers for the paper and
the leaderboard. Run these on the Thor device, then update
`README.md` (reference measurements), `paper/thorai-paper.md` /
`paper/thorai-paper.tex`, and the leaderboard.

## 1. Prerequisites on the Thor device

```bash
# System
#   - NVIDIA DRIVE Thor SDK / JetPack-Drive installed
#   - CUDA 12.x, TensorRT 8.6+ on PATH
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi

# Platform (from this repo)
./tools/scripts/setup.sh                     # creates .venv, installs packages
pip install "thor-benchmark[vision]"         # YOLO / DETR / segmentation
pip install "thor-benchmark[language]"       # LLMs (transformers)
pip install "thor-models[tensorrt]"          # TensorRT + ONNX toolchain

# HuggingFace token (Llama-3-8B is gated; also speeds up model downloads)
huggingface-cli login

# Optional infra for persistence + telemetry
docker compose -f tools/docker/docker-compose.yml up -d postgres influxdb redis
```

Sanity check: `thor-benchmark hardware` should show the Thor GPU
(gpu_available: True, tensorrt_version populated).

## 2. Vision: YOLOv8n (torch CUDA baseline)

```bash
thor-benchmark run \
  --model ultralytics/yolov8n \
  --workload vision \
  --precision fp16 \
  --batch-sizes 1,4,8,16,32 \
  --iterations 300 --warmup 50 \
  --output results/thor-yolov8n-fp16.json \
  --report results/thor-yolov8n-fp16.md
```

Repeat for yolov8s/m/l and for `--precision int8`.

## 3. Vision: TensorRT engine build + benchmark

Build the engine from a saved torch model (ONNX export + TRT build
with min/opt/max batch profiles; INT8 needs a calibrator with real
data — see `thor_models.optimize.trt_builder.Int8Calibrator`):

```bash
python - <<'EOF'
from ultralytics import YOLO
import torch
model = YOLO("yolov8n.pt")
torch.save(model.model, "yolov8n.pt-module.pt")   # full torch module save
EOF

# Build FP16 engine (batch 1/8/32 profile)
python - <<'EOF'
from thor_models.optimize.trt_builder import build_engine_from_model
print(build_engine_from_model(
    "yolov8n.pt-module.pt", precision="fp16",
    input_shape=[1, 3, 640, 640], output_dir="results/engines"))
EOF
# -> results/engines/yolov8n.pt-module.plan
```

Benchmark the engine with a minimal timing loop (execution-context
integration into the workload runner is the next step):

```bash
python - <<'EOF'
import time, torch
from thor_models.optimize.trt_builder import load_engine

runtime, engine, context = load_engine("results/engines/yolov8n.pt-module.plan")
lat = []
for _ in range(100):
    t0 = time.perf_counter()
    # context.execute_async_v2(bindings, stream) — full example in trt_builder docs
    lat.append((time.perf_counter() - t0) * 1000)
print("p50_ms", sorted(lat)[len(lat)//2])
EOF
```

Record: engine build time, `.plan` size, p50/p95/p99 latency,
throughput, and (via `nvidia-smi dmon -c N`) power.

## 4. Language: Llama-3-8B

```bash
# FP16 (torch CUDA) baseline
thor-benchmark run \
  --model meta-llama/Llama-3-8B \
  --workload language \
  --precision fp16 \
  --batch-sizes 1 \
  --iterations 50 --warmup 5 \
  --output results/thor-llama3-fp16.json
```

INT8/INT4: the built-in workload currently runs the torch backend;
for weight-quantized decode use transformers' native loading in a
small script until the toolchain integration lands:

```bash
python - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B", load_in_4bit=True, device_map="auto")  # needs bitsandbytes
ids = tok("The quick brown fox", return_tensors="pt").input_ids.cuda()
lat, tps = [], []
for _ in range(50):
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=128, do_sample=False)
    dt = time.perf_counter() - t0
    lat.append(dt * 1000 / 128)
    tps.append(128 / dt)
print("ms/token p50", sorted(lat)[25], "| tok/s", sorted(tps, reverse=True)[25])
EOF
```

Record: TTFT (first token), ms/token, tokens/s, power, memory peak —
for fp16, int8, int4 at prompt lengths {128, 512, 1024}.

## 5. Segmentation / BEV reference

```bash
thor-benchmark run \
  --model nvidia/segformer-b0-finetuned-ade-512-512 \
  --workload segmentation \
  --precision fp16 \
  --batch-sizes 1,4,8 \
  --iterations 200 \
  --output results/thor-segformer-fp16.json
```

Optionally wire the thor-sense pipeline (`examples/thor-sense`) to a
camera/lidar rig and measure end-to-end fusion latency.

## 6. Telemetry + persistence

```bash
# Write telemetry to InfluxDB after every run
thor-benchmark run --model ultralytics/yolov8n --workload vision --influx ...

# Runs are stored in PostgreSQL when DATABASE_URL is set (MCP server)
DATABASE_URL=postgresql://thor:thor@localhost:5432/thorbench \
  thor-mcp --http-mcp --port 8000
```

## 7. Leaderboard ingestion

```bash
# Aggregate result JSON files into leaderboard.json
python tools/scripts/update_leaderboard.py results/*.json

# Push a run through the MCP server (stores it in Postgres when configured)
python - <<'EOF'
import asyncio
from thor_mcp.client import ThorMCPClient
async def main():
    async with ThorMCPClient(config_path="thor-config.yaml") as c:
        print(await c.call_tool("benchmark_run", {
            "model_id": "ultralytics/yolov8n", "workload_type": "vision",
            "precision": "fp16", "batch_sizes": [1, 4, 8], "iterations": 100}))
asyncio.run(main())
EOF
```

## 8. Paper data collection template

For every row below, record: p50/p95/p99 latency (ms), throughput
(samples/s or tokens/s), avg/peak power (W), peak memory (MB),
thermal peak (°C), engine/precision, batch size.

| Model | Workload | Precision | Batch | Latency p50 | Throughput | Power avg | Memory peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n | vision | fp16 torch | 1 | | | | |
| YOLOv8n | vision | fp16 trt | 8 | | | | |
| YOLOv8n | vision | int8 trt | 8 | | | | |
| Llama-3-8B | language | fp16 | 1 | (ms/token) | (tok/s) | | |
| Llama-3-8B | language | int4 | 1 | (ms/token) | (tok/s) | | |
| SegFormer-B0 | segmentation | fp16 | 4 | | | | |
| LLaVA-1.5-7B | multimodal | int8 | 1 | | | | |

Fill these into `paper/thorai-paper.md` §3.2 and the README table,
then regenerate the leaderboard.

## 9. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `torch.cuda.is_available()` False | Install the CUDA build: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| Llama download fails | `huggingface-cli login` and accept the gated license |
| power_watts missing | Some GPUs don't expose NVML power draw; check `nvidia-smi -q -d POWER` |
| `models_optimize` TensorRT execute fails | Confirm `tensorrt` importable + onnx/onnxscript installed (thor-models[tensorrt]) |
| INT8 engine build | Provide a calibrator with real camera/LiDAR calibration data |
