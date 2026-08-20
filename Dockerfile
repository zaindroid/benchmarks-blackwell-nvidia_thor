# ThorAI platform app — MCP endpoint + leaderboard API + web UI.
#
# Multi-stage: builds the React frontend with Node, then the Python
# runtime with all packages. Served via `python platform_app.py`.
# No host port binding — the host routes to the container by name on the
# shared network, so EXPOSE is informational only.

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

# Build metadata. SOURCE_COMMIT is auto-injected by Coolify as a build arg
# (application setting include_source_commit_in_build=true) -- no .git copy
# needed, and no dependency on Coolify's git-import step actually leaving
# .git in the build context (it doesn't).
#
# build_info() (thor_mcp.deploy) resolves THOR_BUILD_SHA/THOR_BUILD_TIME
# env vars first, then these stamped files, then "dev". THOR_BUILD_SHA is
# set as an env var here (real value when SOURCE_COMMIT is known, else the
# harmless "dev" literal). THOR_BUILD_TIME is deliberately NOT set as an
# env var -- there's no real build-arg source for it, and setting it to a
# literal "dev" env var would permanently shadow the real timestamp
# stamped into .build_time below. Runtime env var overrides still work for
# both if anyone ever wants to pass one explicitly.
ARG SOURCE_COMMIT=dev
ARG THOR_BUILD_SHA=${SOURCE_COMMIT}
ENV THOR_BUILD_SHA=${THOR_BUILD_SHA}
RUN echo "${THOR_BUILD_SHA}" > /app/.build_sha && \
    date -u +%Y-%m-%dT%H:%M:%SZ > /app/.build_time

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
