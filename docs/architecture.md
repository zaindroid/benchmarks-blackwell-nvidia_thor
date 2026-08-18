# Architecture

## Overview

```
                        ┌────────────────────────────────────────────┐
                        │              AI Assistant                 │
                        │  (Claude Desktop / Cursor / LangChain /    │
                        │   AutoGPT / custom)                       │
                        └──────────────┬─────────────────────────────┘
                                       │ MCP (stdio) or REST (HTTP)
                        ┌──────────────▼─────────────────────────────┐
                        │              ThorMCP Server               │
                        │  tools/   (benchmark_run, models_optimize,│
                        │            reports_generate, ...)         │
                        │  resources/ (thor://benchmarks/results,   │
                        │            thor://models/registry, ...)   │
                        │  auth/     (HMAC bearer tokens)           │
                        │  rate_limit/ (token bucket)               │
                        └──────┬───────────────┬─────────────────────┘
                               │               │
                  ┌────────────▼──────┐  ┌─────▼──────────────────────┐
                  │   ThorBench       │  │     Storage layer          │
                  │  BenchmarkRunner  │  │  PostgreSQL (asyncpg)  ←──┼── fallback
                  │  workloads:       │  │  in-memory backend         │
                  │   vision (YOLO)   │  └────────────────────────────┘
                  │   language (LLM)  │
                  │   multimodal (VLM)│
                  │  metrics: latency │        ┌──────────────────────┐
                  │   throughput,     │        │  ThorCore / ThorSDK  │
                  │   power, memory,  │        │  HardwareMonitor     │
                  │   thermal         │        │  (pynvml + psutil)   │
                  └───────────────────┘        │  TelemetryCollector  │
                                               └──────────────────────┘
```

## Packages

### thor-core
Everything shared: `HardwareMonitor` (background sampler of GPU power/temp/
memory/utilization via pynvml, graceful degradation without a GPU),
`MetricCollector` + `percentile`/`summarize_latency` (schema-compatible
result shapes), structured logging (structlog), pydantic config
(`ThorConfig` mirrors `thor-config.yaml`), and `ExperimentTracker`
(in-memory/JSON by default, optional wandb/mlflow mirroring).

### thor-sdk
Device-oriented API: `ThorDevice` (local device; SSH transport scaffolded
behind the `ssh` extra), `PowerMonitor` and `TelemetryCollector` (psutil).

### thor-benchmark
`BenchmarkRunner` orchestrates a run:

1. resolves workload type → `Workload` subclass (vision/language/
   multimodal/segmentation/classification)
2. `prepare_model(model_id, precision)` — lazy imports of
   torch/ultralytics/transformers; raises helpful errors when extras are
   missing
3. for each batch size: warmup runs, then timed iterations while the
   `HardwareMonitor` samples power/memory/thermal
4. aggregates everything through `MetricCollector` into a
   `BenchmarkResult` matching the platform schema

`simulate=True` produces deterministic synthetic results without a GPU —
used by tests, CI and demos.

### thor-models
`ModelRegistry` (register/list/best-metrics, seeded from the built-in
zoo), plus `OptimizationProfile` plans for TensorRT/quantization/
pruning/distillation. Engine building is staged (needs the TensorRT
toolchain on the device).

### thor-mcp
`ThorMCPServer` registers tools, resources and prompts on the `mcp.Server`
instance; `dispatch` routes calls to handler functions in
`thor_mcp/tools/`. Handlers share a `ThorContext` (config, store,
registry, experiments, device, runner, limiter, auth). `--http` serves a
FastAPI JSON bridge (`thor_mcp/http.py`) over the same handlers.

## Data flow for a benchmark

```
client ──benchmark_run──▶ ThorMCPServer ──▶ BenchmarkRunner.run
                                              │
                                              ├─▶ workload.prepare_model
                                              ├─▶ HardwareMonitor.start
                                              ├─▶ per batch: warmup + timed runs
                                              ├─▶ HardwareMonitor.stop
                                              └─▶ MetricCollector → BenchmarkResult
                                                     │
                                                     ▼
                                              BenchmarkStore.save_run
                                              (PostgreSQL or memory)
                                                     │
                                                     ▼
                                    registry.update_best_metrics(model)
```

## Storage

`BenchmarkStore` (thor_mcp) tries PostgreSQL first and falls back to
in-memory per call. The leaderboard API (website/backend) reads the same
`benchmark_runs` table for rankings. InfluxDB is reserved for
time-series telemetry (writer is staged behind `thor-core[timeseries]`).

## Extending

- New workload → subclass `Workload` in `thor_benchmark/workloads/*` and
  register it in `create_workload`.
- New MCP tool → add a spec + handler in `thor_mcp/tools/`; it is
  auto-discovered by `tools/__init__.py`.
- New resource → add a resolver in `thor_mcp/resources/__init__.py`.
