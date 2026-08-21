# Benchmarking Models with the ThorMCP Server

This guide explains how to benchmark AI models through the ThorMCP
server. The service is hosted, so no local installation is required.

**Endpoint:** `https://thor-platform.zaindroid.me/mcp`

---

## Quickstart

1. **Connect.** Add the endpoint to an MCP client (section 2) or use
   the Python client (section 3).
2. **Run a benchmark.** Request a benchmark of `ultralytics/yolov8n`
   at batch sizes 1, 4, 8, iterations 100. Results are returned
   immediately and stored automatically.
3. **Review results.** View the public leaderboard at
   <https://thor-platform.zaindroid.me> or query stored runs with
   `benchmark_list`.

---

## 1. Overview

ThorMCP exposes model benchmarking through the Model Context Protocol
(MCP). Clients that support MCP — Claude Desktop, Cursor, Codex,
opencode — and programs using the Python client can:

- run benchmarks (latency, throughput, power, memory, thermal)
- compare models and configurations
- query the model registry
- generate reports
- track experiments

Results are stored in PostgreSQL and published on the leaderboard at
<https://thor-platform.zaindroid.me>.

---

## 2. Connect from an MCP client

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "thor": {
      "url": "https://thor-platform.zaindroid.me/mcp"
    }
  }
}
```

### Cursor (`~/.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "thor": {
      "url": "https://thor-platform.zaindroid.me/mcp"
    }
  }
}
```

### Codex CLI (`~/.codex/config.toml`)

```toml
[mcp_servers.thor]
url = "https://thor-platform.zaindroid.me/mcp"
```

### opencode (`~/.config/opencode/config.json`)

```json
{
  "mcp": {
    "thor": {
      "type": "remote",
      "url": "https://thor-platform.zaindroid.me/mcp"
    }
  }
}
```

After restarting the client, the following requests are available:

> - "Run a simulated benchmark of ultralytics/yolov8n at batch sizes 1, 4, 8"
> - "List the models in the registry"
> - "Compare the last two benchmark runs"
> - "Generate a markdown report of the latest run"
> - "Report the current hardware status"

---

## 3. Python client

Install the client package:

```bash
pip install thor-mcp
```

Run a benchmark and compare two models:

```python
import asyncio
from thor_mcp.client import ThorMCPClient

URL = "https://thor-platform.zaindroid.me/mcp"

async def main():
    async with ThorMCPClient(url=URL) as mcp:
        # List available tools.
        tools = await mcp.list_tools()
        print(f"{len(tools)} tools available")

        # Benchmark a model. simulate=True returns deterministic results
        # without requiring a GPU.
        run1 = await mcp.call_tool("benchmark_run", {
            "model_id": "ultralytics/yolov8n",
            "workload_type": "vision",
            "precision": "fp16",
            "batch_sizes": [1, 4, 8],
            "iterations": 100,
            "custom_config": {"simulate": True},
        })
        run_id = run1["run_id"]
        print("run:", run_id)
        print("p50 latency:", run1["results"]["latency"]["p50_ms"], "ms")
        print("throughput:", run1["results"]["throughput"]["samples_per_second"], "samples/s")

        # Benchmark a second model and compare.
        run2 = await mcp.call_tool("benchmark_run", {
            "model_id": "ultralytics/yolov8s",
            "workload_type": "vision",
            "custom_config": {"simulate": True},
        })
        comparison = await mcp.call_tool("benchmark_compare", {
            "benchmark_ids": [run_id, run2["run_id"]],
        })
        print("comparison:", comparison["comparison"])

asyncio.run(main())
```

---

## 4. curl

Initialize a session:

```bash
curl -sS https://thor-platform.zaindroid.me/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl","version":"1.0"}}}'
```

The response includes a session id. Subsequent calls pass it back and
use `tools/call`:

```bash
curl -sS https://thor-platform.zaindroid.me/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: <session_id_from_initialize>" \
  -d '{"jsonrpc":"2.0","method":"tools/call","id":2,"params":{"name":"hardware_status","arguments":{}}}'
```

---

## 5. Tools

| Tool | Description |
| --- | --- |
| `benchmark_run` | Run a benchmark and store the result |
| `benchmark_compare` | Compare runs across models or configurations |
| `benchmark_list` | List stored runs, optionally filtered by model or workload |
| `benchmark_history` | Query runs over a trailing time window |
| `hardware_status` | Report current hardware (GPU, CUDA, TensorRT versions) |
| `models_list` / `models_register` | Query or extend the model registry |
| `models_optimize` | Create an optimization profile (TensorRT, INT8 quantization) |
| `models_deploy` | Create a deployment descriptor for an optimized model |
| `datasets_list` / `datasets_register` | Query or register datasets |
| `reports_generate` | Generate a report from a benchmark run |
| `experiments_track` / `experiments_list` | Track or query research experiments |

---

## 6. Result schema

Every benchmark returns the same schema, so any two runs can be
compared directly:

```json
{
  "run_id": "run-3f9c1a2b",
  "hardware": { "device": "...", "gpu_available": false },
  "model": { "name": "ultralytics/yolov8n", "precision": "fp16" },
  "workload": { "type": "vision", "batch_sizes": [1, 4, 8], "iterations": 100 },
  "results": {
    "latency":    { "p50_ms": 7.8, "p95_ms": 9.1, "p99_ms": 10.2, "count": 300 },
    "throughput": { "samples_per_second": 128.4 },
    "power":      { "average_watts": 0.0 },
    "memory":     { "peak_mb": 0.0 },
    "thermal":    { "peak_temp_c": null }
  },
  "simulated": true
}
```

Key metrics: p50/p95/p99 latency (ms), throughput (samples/s or
tokens/s), power (W), memory (MB), thermal (C).

---

## 7. Simulation and hardware runs

The hosted endpoint runs without a GPU, so real model inference
(torch, ultralytics, transformers, TensorRT) is not available there.

1. `custom_config: {"simulate": true}` produces deterministic results
   without a GPU. Use it to validate the workflow and the result
   schema. Any model id is accepted.
2. For hardware measurements, run the benchmark locally on a machine
   with the required runtime, or on a DRIVE Thor device. See the
   [Benchmarking Guide](benchmarking-guide.md) and the [Thor Device
   Runbook](thor-device-runbook.md). The result schema is identical,
   so local results can be stored in the same leaderboard.

---

## 8. Where results are stored

Every `benchmark_run` is stored automatically. Results can be viewed:

- **Web UI and leaderboard:** <https://thor-platform.zaindroid.me>
- **REST API:** `GET /api/leaderboard`, `GET /api/stats`
- **Through MCP:** `benchmark_list`, `benchmark_history`, `benchmark_compare`

---

## 9. Troubleshooting

| Symptom | Resolution |
| --- | --- |
| Connection failed | Verify the server: `curl https://thor-platform.zaindroid.me/health` should return `{"status":"ok"}` |
| Real run hangs or errors on the hosted endpoint | The hosted endpoint has no GPU; use `"custom_config": {"simulate": true}` or run locally (section 7) |
| Unknown model | In simulation mode any model id is accepted; for real runs the model must be registered with `models_register` and the runtime installed locally |
| Run not visible on the leaderboard | Use `benchmark_list` / `benchmark_history`; the leaderboard shows the best result per model |
