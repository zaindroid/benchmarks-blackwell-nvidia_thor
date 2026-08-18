# Benchmarking Guide

## Workload types

| Workload | Models (built-in) | Real deps |
| --- | --- | --- |
| `vision` | ultralytics/yolov8n/s/m/l, hustvl/detr-resnet50, hustvl/rt-detr | `thor-benchmark[vision]` |
| `segmentation` | ultralytics/yolov8n-seg, nvidia/segformer-b0 | `thor-benchmark[vision]` |
| `classification` | timm/resnet50, google/vit-base | `thor-benchmark[vision]` |
| `language` | meta-llama/Llama-3-8B, mistralai/Mistral-7B, microsoft/Phi-3-mini, Qwen/Qwen2-7B, google/gemma-7b | `thor-benchmark[language]` |
| `multimodal` | llava-hf/llava-1.5-7b-hf, Qwen/Qwen-VL-Chat | `thor-benchmark[multimodal]` |

Short aliases work too: `yolov8n`, `llama-3-8b`, `mistral-7b`, ...

## CLI

```bash
thor-benchmark run \
  --model meta-llama/Llama-3-8B \
  --workload language \
  --precision int4 \
  --batch-sizes 1,4,8 \
  --iterations 100 \
  --warmup 10 \
  --output results.json \
  --report report.md
```

Config-file driven runs use `configs/*.yaml`:

```bash
thor-benchmark run --config packages/benchmarks/thor-benchmark/configs/llama-3-8b.yaml
```

`--simulate` runs deterministic synthetic benchmarks — useful for CI,
tests and demos without a Thor device.

## Metrics collected

| Section | Fields |
| --- | --- |
| `latency` | p50/p95/p99/min/max/std (ms) + count |
| `throughput` | samples_per_second, max_batch_size, tokens_per_second |
| `power` | average/peak/idle watts, joules_per_sample |
| `memory` | peak_mb, average_mb, allocation_pattern |
| `thermal` | start/end/peak temp °C, throttling_events |

Power/memory/thermal require a GPU with pynvml; on GPU-less machines
those sections are marked `"available": false`.

## Recommendations for reproducible results

1. **Warm up** — always keep a warmup (default 10 iterations) so clocks
   and caches are stable before timed iterations.
2. **Iterations** — use ≥ 100 iterations for stable percentiles; more for
   low-variance power readings.
3. **Pin the environment** — record driver/CUDA/TensorRT versions
   (already included in the `hardware` section of every result).
4. **Isolate the device** — close other GPU workloads; run multiple
   trials and take the best/median.
5. **Batch sweep** — benchmark the batch sizes you actually deploy with
   (e.g. `1,4,8,16,32` for real-time perception).

## Storing results

The MCP server stores every `benchmark_run` (PostgreSQL when configured,
in-memory otherwise). Query them:

```python
async with ThorMCPClient() as client:
    runs = await client.read_resource("thor://benchmarks/results")
    compare = await client.call_tool("benchmark_compare", {
        "benchmark_ids": [r["run_id"] for r in runs["runs"]],
        "metrics": ["latency_p50", "throughput", "power_watts"],
        "format": "markdown",
    })
```

## Regression checks

`tools/scripts/check_regression.py` compares baseline vs candidate
metrics and fails CI when latency/power regress beyond 10/15% or
throughput beyond 5%.

## Leaderboard

`tools/scripts/update_leaderboard.py results.json` aggregates result
files into `leaderboard.json`. The website backend
(`website/backend`, FastAPI) serves `/api/leaderboard`, `/api/compare`,
`/api/stats` from PostgreSQL, and the React frontend renders them.
