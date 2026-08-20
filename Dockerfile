# ThorAI platform app — MCP endpoint + leaderboard API + web UI.
#
# Multi-stage: builds the React frontend with Node, then the Python
# runtime with all packages. Served via `python platform_app.py`.
# No host port binding — the host routes to the container by name on the
# shared network, so EXPOSE is informational only.

# --- build metadata -------------------------------------------------------
# Stamp the real git sha + build time from the build context (.git clone)
# so GET /version reports actual values without external build args.
# Falls back to "dev" when .git is unavailable (e.g. archive builds).
FROM alpine/git AS buildmeta
COPY .git /repo/.git
RUN mkdir -p /out && \
    (git -C /repo rev-parse --short HEAD 2>/dev/null || echo dev) > /out/sha && \
    (date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo dev) > /out/built

# --- frontend --------------------------------------------------------------
FROM node:20-alpine AS frontend
WORKDIR /app
COPY website/frontend/package.json website/frontend/package-lock.json* ./
RUN npm install
COPY website/frontend/ .
RUN npm run build

# --- runtime ---------------------------------------------------------------
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl wget && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY packages /packages
COPY website/backend /website/backend
COPY pyproject.toml thor-config.yaml ./
COPY platform_app.py ./
COPY --from=frontend /app/dist /app/website/frontend/dist

# Build metadata (env overrides the stamped files; see thor_mcp.deploy).
ARG THOR_BUILD_SHA=dev
ARG THOR_BUILD_TIME=dev
ENV THOR_BUILD_SHA=${THOR_BUILD_SHA} \
    THOR_BUILD_TIME=${THOR_BUILD_TIME}
COPY --from=buildmeta /out/sha /app/.build_sha
COPY --from=buildmeta /out/built /app/.build_time

RUN pip install --no-cache-dir \
    -e "/packages/core/thor-core[db]" \
    -e /packages/core/thor-sdk \
    -e /packages/benchmarks/thor-benchmark \
    -e /packages/benchmarks/thor-models \
    -e "/packages/core/thor-mcp[postgres]" \
    -e "/website/backend[postgres]"

# Fail loudly in production if required secrets are missing.
ENV APP_ENV=production
EXPOSE 8000
CMD ["python", "platform_app.py"]
