# ThorAI Platform

Open-source benchmarking and deployment platform for **NVIDIA DRIVE Thor**.

| Component | What it does |
| --- | --- |
| **ThorBench** | Automated benchmarking framework (latency / throughput / power / memory / thermal) |
| **ThorMCP** | MCP server so AI assistants can benchmark and optimize models on Thor |
| **ThorModels** | Model registry + zoo + optimization profiles |
| **ThorCore / ThorSDK** | Shared utilities, hardware monitoring, telemetry |
| **Leaderboard** | REST API + React frontend for community results |

## Repository layout

```
thor-ai-platform/
├── packages/
│   ├── core/
│   │   ├── thor-core/        # hardware, metrics, logging, config, experiments
│   │   ├── thor-sdk/         # device communication, power, telemetry
│   │   └── thor-mcp/         # MCP server (tools, resources, prompts) + REST bridge
│   └── benchmarks/
│       ├── thor-benchmark/   # runner, workloads (vision/language/multimodal), CLI
│       └── thor-models/      # registry, model zoo, optimization profiles
├── examples/                 # quickstart, MCP client, research automation, workflows
├── tools/
│   ├── docker/               # Dockerfiles + docker-compose (postgres, influxdb, redis)
│   ├── scripts/              # setup / benchmark / deploy / leaderboard / regression
│   └── ci/github-actions/    # benchmark + regression workflows (copy to .github/workflows)
├── website/                  # leaderboard backend (FastAPI) + frontend (React)
└── docs/                     # architecture, getting-started, benchmarking-guide, api-reference
```

## Quick start (no Thor hardware required)

```bash
# Linux/macOS — creates .venv, installs all packages in editable mode
./tools/scripts/setup.sh

# Windows
python -m venv .venv
source .venv/Scripts/activate        # (git-bash: source .venv/Scripts/activate)
pip install -e packages/core/thor-core
pip install -e packages/core/thor-sdk
pip install -e packages/benchmarks/thor-benchmark
pip install -e packages/benchmarks/thor-models
pip install -e packages/core/thor-mcp
pip install pytest
```

### Run a benchmark

```bash
# Deterministic synthetic run — no GPU needed
thor-benchmark run --model ultralytics/yolov8n --workload vision --precision fp16 --simulate

# From a config file
thor-benchmark run --config packages/benchmarks/thor-benchmark/configs/llama-3-8b.yaml --simulate

# Check hardware detection
thor-benchmark hardware
```

Programmatically:

```python
from thor_benchmark import BenchmarkRunner

runner = BenchmarkRunner()
results = runner.run(
    model_id="ultralytics/yolov8n",
    workload_type="vision",
    precision="fp16",
    batch_sizes=[1, 4, 8],
    iterations=100,
    simulate=True,          # remove on a real Thor device
)
print(results.results["latency"]["p50_ms"])
```

### Use the MCP server

```bash
# stdio mode (default) — works with Claude Desktop, Cursor, LangChain, ...
thor-mcp --stdio

# HTTP REST bridge
thor-mcp --http --port 3000
```

MCP client config for Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "thor": {
      "command": "thor-mcp",
      "args": ["--config", "/path/to/thor-config.yaml"]
    }
  }
}
```

Then ask your assistant:

- "Benchmark Llama-3-8B in int4 precision"
- "What models have been benchmarked on Thor?"
- "Generate a report of the last benchmark"

Programmatic client:

```python
import asyncio
from thor_mcp.client import ThorMCPClient

async def main():
    async with ThorMCPClient(config_path="thor-config.yaml") as client:
        status = await client.call_tool("hardware_status", {})
        result = await client.call_tool("benchmark_run", {
            "model_id": "ultralytics/yolov8n",
            "workload_type": "vision",
            "custom_config": {"simulate": True},
        })
        print(result["results"]["latency"]["p50_ms"])

asyncio.run(main())
```

### Tools exposed over MCP

| Tool | Purpose |
| --- | --- |
| `benchmark_run` | Run a benchmark (vision/language/multimodal/segmentation/classification) |
| `benchmark_compare` | Compare runs as JSON / CSV / markdown |
| `benchmark_list` | List stored runs |
| `models_list` / `models_register` | Registry management |
| `models_optimize` | Create an optimization profile (TensorRT/quantization/...) |
| `models_deploy` | Create a deployment descriptor |
| `datasets_list` / `datasets_register` | Dataset registry |
| `reports_generate` | Markdown/JSON report from a run |
| `hardware_status` | Device status and telemetry |
| `experiments_track` / `experiments_list` | Research experiment tracking |

### Resources

| URI | Content |
| --- | --- |
| `thor://benchmarks/results` | Stored benchmark runs (query: `?model_id=...&workload_type=...`) |
| `thor://models/registry` | Registered models |
| `thor://hardware/telemetry` | Device + host telemetry |
| `thor://experiments/history` | Tracked experiments |

## Infrastructure

```bash
# Full stack (postgres, influxdb, redis, mcp-server, frontend)
docker compose -f tools/docker/docker-compose.yml up -d

# Just the databases (for development)
docker compose -f tools/docker/docker-compose.yml up -d postgres influxdb redis
```

PostgreSQL schema: `website/database/migrations/001_initial_schema.sql` (auto-applied by docker-compose).
InfluxDB schema: `website/database/influxdb_schema.md`.

Storage **falls back to in-memory** when PostgreSQL is not reachable, so the MCP server and all tools work on a laptop without any database.

## Tests

```bash
pip install pytest pytest-asyncio
pytest packages/            # runs all unit tests (no GPU, no DB required)
```

## Environment

Copy `.env.example` to `.env` and `thor-config.example.yaml` to `thor-config.yaml`, then edit
device IPs, API keys and database credentials.

## Documented deviations from the original plan (MVP)

1. **MCP SDK API** — the plan's `server.run_stdio()` pseudocode was replaced with the current low-level `mcp>=1.0` API (`Server` + `stdio_server` + `InitializationOptions`), which is what real MCP clients speak.
2. **Auth** — HMAC-signed tokens (stdlib `hmac`) instead of JWT/passlib, avoiding Python 3.12+ bcrypt incompatibilities. Same bearer-token workflow.
3. **Rate limiting** — lightweight in-process token bucket instead of `slowapi`, so the same limiter works for stdio and HTTP modes.
4. **Heavy optional deps** — `asyncpg`, `influxdb-client`, `redis`, `wandb`, `mlflow` moved to optional extras; the MVP runs with zero infrastructure. Experiment tracking defaults to an in-memory/JSON store.
5. **Simulate mode** — deterministic synthetic benchmarks (`--simulate` / `custom_config.simulate=true`) so the platform is demonstrable and testable without a Thor device.
6. **HTTP mode** — a FastAPI JSON bridge over the same tool handlers (auth + rate limiting included). Full MCP-over-HTTP transport is a follow-up.
7. **Run ids** — `run-<hex>` text ids; the PostgreSQL migration uses `TEXT` primary keys to match (the plan's `UUID` column would reject these ids).
8. **Model ids** — zoo keys use full ids (`ultralytics/yolov8n`, `meta-llama/Llama-3-8B`) with short aliases (`yolov8n`, ...) resolved automatically.

## Roadmap (staged)

- [ ] Sprint 5: TensorRT engine building + quantization execution (`thor_models.optimize`)
- [ ] Time-series writer for InfluxDB (`thor-core[timeseries]`)
- [ ] MCP streamable-HTTP transport
- [ ] Full thor-sense (BEV/sensor fusion) and thor-vlm reference implementations
- [ ] Community leaderboard with model submission portal

## License

MIT — see [LICENSE](LICENSE).
