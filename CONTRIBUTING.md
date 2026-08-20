# Contributing

Thanks for your interest in ThorAI! This is an open benchmarking
platform for NVIDIA DRIVE Thor. All contributions — issues, docs,
benchmarks, reference implementations, and data — are welcome.

## Getting started

```bash
# Linux/macOS
./tools/scripts/setup.sh            # creates .venv, installs all packages
# or manually:
python -m venv .venv && source .venv/bin/activate
pip install -e "packages/core/thor-core[db]"
pip install -e packages/core/thor-sdk
pip install -e packages/benchmarks/thor-benchmark
pip install -e packages/benchmarks/thor-models
pip install -e "packages/core/thor-mcp[postgres]"
pip install -e "website/backend[postgres]"
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest packages/ -q
pytest examples/thor-sense/tests examples/thor-vlm/tests website/backend/tests -q
```

## Adding a benchmark

1. Add the model to the zoo in
   `packages/benchmarks/thor-models/src/thor_models/zoo/`.
2. Register a workload config under
   `packages/benchmarks/thor-benchmark/configs/` (see the YOLOv8/Llama
   examples).
3. Add a unit test asserting the result schema
   (`run_id`, `hardware`, `model`, `workload`, `results`).
4. Run `thor-benchmark run --config <your-config> --simulate` to
   verify.

## Reporting real Thor numbers

If you have access to a DRIVE Thor device, follow
`docs/thor-device-runbook.md` and submit your results to the
leaderboard (web UI or `POST /api/submissions`). Real hardware numbers
are the most valuable contribution this project can receive.

## Code style

- Python 3.10+, type hints, `ruff`-clean.
- Keep packages decoupled; optional heavy deps (torch, transformers,
  tensorrt) live behind extras so the base install stays light.
- Match the existing result JSON schema exactly — schema changes
  require a migration + leaderboard API update.

## License

MIT — see [LICENSE](LICENSE).
