#!/usr/bin/env bash
# Real benchmark runner for a DRIVE Thor device (or any NVIDIA GPU host).
#
# Runs real inference benchmarks (no simulation) with live power, memory
# and thermal sampling via NVML, writes result JSON + markdown report,
# and optionally submits the measured metrics to the public leaderboard
# for review.
#
# Usage:
#   ./tools/scripts/benchmark-device.sh                       # YOLOv8n baseline
#   ./tools/scripts/benchmark-device.sh --model meta-llama/Llama-3-8B --workload language --precision int8
#   ./tools/scripts/benchmark-device.sh --submit             # also POST to the leaderboard portal
#
# Requirements on the device: NVIDIA driver + CUDA toolchain, torch with
# CUDA support (pip install torch --index-url https://download.pytorch.org/whl/cu121).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="ultralytics/yolov8n"
WORKLOAD="vision"
PRECISION="fp16"
BATCH_SIZES="1,4,8"
ITERATIONS="300"
WARMUP="50"
SUBMIT=0
VENV_PY="python3"
[ -x .venv/bin/python ] && VENV_PY=".venv/bin/python"

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --submit) SUBMIT=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

OUT="results/thor-${MODEL##*/}-${PRECISION}"
mkdir -p results

echo "==> CUDA check"
$VENV_PY - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA is not available - install a CUDA torch build"
print("CUDA device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
EOF

echo "==> Installing model runtimes (extras)"
$VENV_PY -m pip install -q -e "packages/benchmarks/thor-benchmark[vision,language,multimodal]" 2>/dev/null || \
  $VENV_PY -m pip install -q -e "packages/benchmarks/thor-benchmark[vision,language]"

echo "==> Running real benchmark: $MODEL ($WORKLOAD, $PRECISION, batch $BATCH_SIZES)"
thor-benchmark run \
  --model "$MODEL" \
  --workload "$WORKLOAD" \
  --precision "$PRECISION" \
  --batch-sizes "$BATCH_SIZES" \
  --iterations "$ITERATIONS" \
  --warmup "$WARMUP" \
  --output "${OUT}.json" \
  --report "${OUT}.md"

echo "==> Result summary"
$VENV_PY - <<EOF
import json
d = json.load(open("${OUT}.json"))
r = d["results"]
print("run_id      :", d["run_id"])
print("hardware    :", d["hardware"].get("gpu_name") or d["hardware"].get("device"))
print("p50 latency :", r["latency"].get("p50_ms"), "ms")
print("p95 latency :", r["latency"].get("p95_ms"), "ms")
print("throughput  :", r["throughput"].get("samples_per_second"), "samples/s")
print("power avg   :", r["power"].get("average_watts"), "W")
print("memory peak :", r["memory"].get("peak_mb"), "MB")
print("thermal peak:", r["thermal"].get("peak_temp_c"), "C")
EOF

if [ "$SUBMIT" = "1" ]; then
  echo "==> Submitting measured metrics to the leaderboard portal"
  $VENV_PY - <<EOF
import json, urllib.request
d = json.load(open("${OUT}.json"))
r = d["results"]
payload = {
    "model_id": "${MODEL}",
    "architecture": d["model"].get("architecture"),
    "parameters": d["model"].get("parameters"),
    "source": "thor-device",
    "metrics": {
        "latency_p50_ms": r["latency"].get("p50_ms"),
        "latency_p95_ms": r["latency"].get("p95_ms"),
        "throughput": r["throughput"].get("samples_per_second"),
        "power_watts": r["power"].get("average_watts"),
        "memory_mb": r["memory"].get("peak_mb"),
    },
    "notes": "Measured on device: " + str(d["hardware"].get("gpu_name") or d["hardware"].get("device")),
}
req = urllib.request.Request(
    "https://thor-platform.zaindroid.me/api/submissions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=15) as resp:
    print("submission:", json.loads(resp.read())["submission"]["submission_id"], "- pending review")
EOF
fi

echo "==> Done: ${OUT}.json (results), ${OUT}.md (report)"
