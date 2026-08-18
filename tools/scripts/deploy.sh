#!/usr/bin/env bash
# Deploy the ThorAI platform stack with Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMPOSE="docker compose -f tools/docker/docker-compose.yml"

echo "==> Starting infrastructure (postgres, influxdb, redis)"
$COMPOSE up -d postgres influxdb redis
sleep 10

echo "==> Building images"
$COMPOSE build

echo "==> Starting services"
$COMPOSE up -d

echo "==> Status"
$COMPOSE ps
