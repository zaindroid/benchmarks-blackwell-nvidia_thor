# Benchmarking Models with the ThorMCP Server

A simple, copy-paste guide to benchmarking AI models through the
ThorAI MCP server — no local install required.

**Live endpoint:** `https://thor-platform.zaindroid.me/mcp`

---

## 1. What this is

ThorMCP is an MCP server that exposes model benchmarking as
conversation. Any MCP-capable assistant (Claude Desktop, Cursor,
Codex, opencode, ...) — or your own Python — can:

- run a benchmark (latency, throughput, power, memory, thermal)
- compare models and configurations
- look up the model registry
- generate reports
- track experiments

Results are stored in PostgreSQL and shown on the public leaderboard
at <https://thor-platform.zaindroid.me>.

---

## 2. Option A — Use it from an AI assistant (easiest)

Add the server to your client, then just ask in plain language.

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

Restart your assistant, then try prompts like:

> - "Run a simulated benchmark of ultralytics/yolov8n at batch sizes 1, 4, 8"
> - "What models are in the registry?"
> - "Compare the last two benchmark runs"
> - "Generate a markdown report of the latest run"
> - "What hardware is this platform running on?"

---

## 3. Option B — Use it from Python

Install the client package:

```bash
pip install thor-mcp
```

Then run a benchmark:

```python
import asyncio
from thor_mcp.client import ThorMCPClient

URL = "https://thor-platform.zaindroid.me/mcp"

async def main():
    async with ThorMCPClient(url=URL) as mcp:
        # 1. What can I do?
        tools = await mcp.list_tools()
        print(f"{len(tools)} tools available")

        # 2. Benchmark a model (simulate=True is instant, no GPU needed)
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

        # 3. Benchmark a second model and compare
        run2 = await mcp.call_tool("benchmark_run", {
            "model_id": "ultralytics/yolov8s",
            "workload_type": "vision",
            "custom_config": {"simulate": True},
        })
        comparison = await mcp.call_tool("benchmark_compare", {
            "benchmark_ids": [run_id, run2["run_id"]],
        })
        print("comparison:", comparison["rows"])

asyncio.run(main())
```

---

## 4. Option C — Use it with curl

```bash
# initialize a session
curl -sS https://thor-platform.zaindroid.me/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl","version":"1.0"}}}'
```

The response includes a session id; subsequent calls pass it back and
use `tools/call`:

```bash
curl -sS https://thor-platform.zaindroid.me/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: <session_id_from_initialize>" \
  -d '{"jsonrpc":"2.0","method":"tools/call","id":2,"params":{"name":"hardware_status","arguments":{}}}'
```

---

## 5. The tools

| Tool | What it does |
| --- | --- |
| `benchmark_run` | **Run a benchmark** and store the result (core tool) |
| `benchmark_compare` | Compare runs across models / configs |
| `benchmark_list` | List stored runs (filter by model/workload) |
| `benchmark_history` | Query runs over a trailing time window |
| `hardware_status` | Current hardware (GPU, CUDA, TensorRT versions) |
| `models_list` / `models_register` | Browse / add models to the registry |
| `models_optimize` | Create an optimization profile (TensorRT, INT8 quantization) |
| `models_deploy` | Build a deployment descriptor for an optimized model |
| `datasets_list` / `datasets_register` | Browse / register datasets |
| `reports_generate` | Generate a markdown report from a run |
| `experiments_track` / `experiments_list` | Track / browse research experiments |

---

## 6. What a result looks like

Every benchmark returns the same schema — you can compare any two runs
directly:

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

Key metrics: **p50/p95/p99 latency** (ms), **throughput** (samples/s
or tokens/s), **power** (W), **memory** (MB), **thermal** (°C).

---

## 7. Simulated vs. real numbers (important)

The hosted endpoint runs in the cloud **without a GPU**, so real model
inference (torch/ultralytics/transformers/TensorRT) is not available
there. Two things to know:

1. **`custom_config: {"simulate": true}`** runs instantly and
   deterministically. Use it to explore the workflow, test your
   prompts, and validate the result schema. Any model id works.
2. **For real hardware numbers**, run the benchmark locally on your
   machine (or a DRIVE Thor device) — see the [Benchmarking
   Guide](benchmarking-guide.md) and the [Thor Device
   Runbook](thor-device-runbook.md). The result JSON is identical, so
   your real numbers drop straight into the same leaderboard schema.

---

## 8. Where results go

Every `benchmark_run` is stored automatically. View them:

- **Web UI + leaderboard:** <https://thor-platform.zaindroid.me>
- **REST API:** `GET /api/leaderboard`, `GET /api/stats`
- **Through MCP:** `benchmark_list`, `benchmark_history`, `benchmark_compare`

---

## 9. Troubleshooting

| Symptom | Fix |
| --- | --- |
| "Connection failed" | Check the server: `curl https://thor-platform.zaindroid.me/health` should return `{"status":"ok"}` |
| Real run hangs or errors on the hosted endpoint | It has no GPU — add `"custom_config": {"simulate": true}` or run locally (section 7) |
| "Unknown model" | In simulate mode any id works; for real runs the model must be registered (`models_register`) and the runtime installed locally |
| Can't see my run on the leaderboard | Use `benchmark_list` / `benchmark_history` — the leaderboard shows best-per-model |
