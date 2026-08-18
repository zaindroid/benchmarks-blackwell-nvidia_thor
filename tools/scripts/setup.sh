#!/usr/bin/env bash
# ThorAI Platform development setup (Linux/macOS; adapt for Windows).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv"

echo "==> Creating virtualenv at $VENV"
"$PYTHON" -m venv "$VENV"
source "$VENV/bin/activate"

echo "==> Installing packages (editable)"
pip install --upgrade pip
pip install -e packages/core/thor-core
pip install -e packages/core/thor-sdk
pip install -e packages/benchmarks/thor-benchmark
pip install -e packages/benchmarks/thor-models
pip install -e packages/core/thor-mcp

echo "==> Installing dev tooling"
pip install -e ".[dev]"

echo "==> Configuring environment"
[ -f .env.example ] && cp -n .env.example .env || true
[ -f thor-config.yaml ] || cp thor-config.example.yaml thor-config.yaml

echo "==> Done."
echo "Run: source .venv/bin/activate && thor-benchmark list-workloads"
echo "Optional infra: docker compose -f tools/docker/docker-compose.yml up -d postgres influxdb redis"
