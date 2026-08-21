# Public Launch Plan

Everything you need to go public. Steps, post drafts, and a
one-command GitHub push.

## Checklist

1. [ ] `./tools/scripts/push-to-github.sh` — pushes the repo to GitHub
2. [ ] Set the real repo URL (replace `yourusername/thor-ai-platform`):
   - README badge + docs links
   - `paper/thorai-paper.tex` author line
3. [ ] Post the drafts below (edit to taste)
4. [ ] Run `docs/thor-device-runbook.md` on real hardware, then update
   the README benchmark table + paper with Thor numbers

## Post drafts

### LinkedIn

> I built the first open-source platform for NVIDIA DRIVE Thor — the
> automotive supercomputer most developers never get access to.
>
> MCP server with 13 tools (AI assistants can benchmark models)
> Real BEV perception pipeline (multi-camera sensor fusion)
> Vision-language model with safety filters
> TensorRT + quantization optimization toolchain
> Community leaderboard with submission portal
>
> 107 tests passing.
>
> GitHub: <repo url> · Paper: <repo url>/paper/thorai-paper.md
>
> #NVIDIA #AutonomousDriving #EdgeAI #MCP #Robotics

### X / Twitter

> ThorAI: the first open-source benchmarking platform for NVIDIA DRIVE
> Thor.
>
> - 13 MCP tools — tell an AI assistant to benchmark/optimize models
> - YOLOv8n 71.3ms P50, real GPU power/memory sampling
> - INT8 quantization: 2x compression
> - BEV + VLM reference implementations
> - 107 tests passing
>
> <repo url>
>
> @NVIDIAEmbedded @NVIDIAAI

### Hacker News (Show HN)

> **Show HN: ThorAI — an MCP server for benchmarking automotive AI on
> NVIDIA DRIVE Thor**
>
> NVIDIA DRIVE Thor is the automotive supercomputer that almost nobody
> has access to — and there is no public benchmark corpus for it. I
> built an open platform to change that:
>
> - Benchmark suite: latency/throughput/power/memory/thermal with a
>   unified JSON schema (YOLOv8, Llama-3-8B, LLaVA, ...)
> - MCP server: 13 tools, so any AI assistant can run benchmarks,
>   compare results, and manage models from natural language
> - TensorRT + quantization toolchain (real INT8 dynamic quantization
>   execution, TensorRT builder with batch profiles)
> - Reference implementations: BEV sensor fusion (camera+LiDAR) and an
>   on-device VLM with automotive safety filters
> - Community leaderboard with a moderated submission portal
> - 107 tests, 5 packages, Docker + remote MCP hosting (streamable
>   HTTP), CI with regression thresholds
>
> Deterministic simulation mode means you can try everything without
> hardware. Feedback welcome — especially from anyone with Thor
> hardware who wants to contribute the first real numbers.

### Reddit

- **r/robotics**: "Open-source benchmarking platform for NVIDIA DRIVE
  Thor — BEV sensor fusion, MCP tools, community leaderboard"
- **r/LocalLLaMA**: "LLM/VLM benchmarking for automotive edge AI:
  Llama-3-8B int4/int8, on-device VLM with safety filters, TensorRT
  builder"
- **r/selfhosted**: "Self-host an MCP server that benchmarks AI models
  with GPU power/memory monitoring — Docker + streamable HTTP"
- **r/MachineLearning**: "ThorAI: first open benchmark suite + MCP
  platform for NVIDIA DRIVE Thor (paper draft included)"

## Key claims to verify before posting

- "107 tests passing" — `pytest packages/` + example suites
- YOLOv8n numbers are dev-workstation reference values (CPU torch),
  clearly labeled until real Thor numbers exist
