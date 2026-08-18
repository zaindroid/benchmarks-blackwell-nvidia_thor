# ThorMCP API Reference

## Tools

### benchmark_run

Run a benchmark on NVIDIA Thor.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model_id` | string | — | Model identifier (e.g. `meta-llama/Llama-3-8B`, `ultralytics/yolov8n`) |
| `workload_type` | string | — | `vision` \| `language` \| `multimodal` \| `segmentation` \| `classification` |
| `precision` | string | `fp16` | `fp32` \| `fp16` \| `int8` \| `int4` \| `fp8` |
| `batch_sizes` | int[] | `[1,4,8]` | Batch sizes to test |
| `iterations` | int | `100` | Timed iterations per batch |
| `warmup_iterations` | int | `10` | Untimed warmup iterations |
| `collect_power` | bool | `true` | Sample GPU power |
| `collect_memory` | bool | `true` | Sample GPU memory |
| `collect_thermal` | bool | `true` | Sample GPU temperature |
| `custom_config` | object | `{}` | Workload config; `custom_config.simulate=true` = synthetic GPU-free run |

**Returns**

```json
{
  "status": "success",
  "run_id": "run-3f2a9c1b4d5e",
  "timestamp": "2026-08-18T...Z",
  "hardware": { "...": "..." },
  "model": { "name": "...", "precision": "fp16", "...": "..." },
  "workload": { "type": "vision", "batch_sizes": [1, 4, 8], "...": "..." },
  "results": {
    "latency": { "p50_ms": 3.2, "p95_ms": 4.1, "p99_ms": 5.0,
                 "min_ms": 2.9, "max_ms": 6.2, "std_ms": 0.4, "count": 300 },
    "throughput": { "samples_per_second": 833.3, "max_batch_size": 8,
                    "tokens_per_second": 0.0 },
    "power": { "average_watts": 250.0, "peak_watts": 275.0, "...": "..." },
    "memory": { "peak_mb": 2048.0, "...": "..." },
    "thermal": { "start_temp_c": 52.0, "...": "..." }
  },
  "simulated": false
}
```

### benchmark_compare

Compare benchmark results across runs.

**Parameters**: `benchmark_ids` (string[]), `metrics`
(`latency_p50`, `latency_p99`, `throughput`, `power_watts`, `memory_mb`,
`tokens_per_second`), `format` (`json` | `csv` | `markdown`).

**Returns**: `{ "comparison": [...], "metrics": [...] }`

### benchmark_list

List stored runs. Filters: `model_id`, `workload_type`, `limit`.

### models_list / models_register

Registry queries and registration (`models_register` requires `model_id`).

### models_optimize

Create an optimization profile.

**Parameters**: `model_id`, `optimization_type`
(`tensorrt` | `quantization` | `pruning` | `distillation`), `precision`,
`target_latency_ms`, `target_throughput`, `target_memory_mb`,
`enable_sparsity`, `execute` (default `false`).

**Returns**: `{ "profile_id": "opt-...", "status": "planned", "targets": {...}, "note": "..." }`

> Engine building (`execute=true`) requires the TensorRT toolchain on the
> device — staged for the optimization sprint.

### models_deploy

Create a deployment descriptor for an optimized model.

### datasets_list / datasets_register

Dataset registry. `datasets_register` requires `dataset_id`.

### reports_generate

Generate a report from a run. Parameters: `benchmark_id`, `format`
(`markdown` | `json`). Returns `{ "report_id", "content", ... }`.

### hardware_status

Current device status: GPU availability/utilization/temp/power, CPU,
driver/CUDA/TensorRT versions.

### experiments_track / experiments_list

Research experiment tracking. `experiments_track` requires `name`;
optional `hypothesis`, `config`, `results`, `metrics`, `tags`, `description`.

## Resources

| URI | Query params | Content |
| --- | --- | --- |
| `thor://benchmarks/results` | `model_id`, `workload_type`, `limit` | Stored runs |
| `thor://benchmarks/results/{run_id}` | — | Single run |
| `thor://models/registry` | `architecture`, `optimized` | Registered models |
| `thor://hardware/telemetry` | — | Device + host telemetry |
| `thor://experiments/history` | `status` | Tracked experiments |

## Prompts

| Name | Arguments | Use |
| --- | --- | --- |
| `benchmark-new-model` | `model_id`, `precision` | Benchmark a new model |
| `optimize-for-thor` | `model_id`, `optimization_type` | Optimize for Thor |
| `generate-report` | `run_id`, `format` | Generate a report |

## HTTP bridge (`--http`)

| Endpoint | Method | Auth |
| --- | --- | --- |
| `/health` | GET | — |
| `/auth/token` | POST | — |
| `/tools` | GET | Bearer |
| `/tools/{name}` | POST | Bearer |
| `/resources` | GET | Bearer |
| `/resources/{uri}` | GET | Bearer |

## Leaderboard API (website/backend)

| Endpoint | Description |
| --- | --- |
| `GET /api/leaderboard` | Ranked results (`metric`, `top_k`, `timeframe_days`) |
| `GET /api/models/{model_id}/history` | History for a model |
| `POST /api/compare` | Compare models |
| `GET /api/stats` | Platform statistics |
