#!/usr/bin/env bash
# Run a benchmark from the CLI.
# Usage: tools/scripts/benchmark.sh [--simulate] [--model MODEL] [--workload TYPE] [--precision P]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SIMULATE=""
MODEL="ultralytics/yolov8n"
WORKLOAD="vision"
PRECISION="fp16"
CONFIG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --simulate) SIMULATE="--simulate"; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ -n "$CONFIG" ]; then
  thor-benchmark run --config "$CONFIG" --simulate $SIMULATE
else
  thor-benchmark run \
    --model "$MODEL" \
    --workload "$WORKLOAD" \
    --precision "$PRECISION" \
    --output results/result.json \
    --report results/report.md \
    $SIMULATE
fi
