# ThorAI platform app — MCP endpoint + leaderboard API + web UI.
#
# Multi-stage: builds the React frontend with Node, then the Python
# runtime with all packages. Served via `python platform_app.py`.
# No host port binding — the host routes to the container by name on the
# shared network, so EXPOSE is informational only.

FROM node:20-alpine AS frontend
WORKDIR /app
COPY website/frontend/package.json website/frontend/package-lock.json* ./
RUN npm install
COPY website/frontend/ .
RUN npm run build

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl wget && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY packages /packages
COPY website/backend /website/backend
COPY pyproject.toml thor-config.yaml ./
COPY platform_app.py ./
COPY --from=frontend /app/dist /app/website/frontend/dist

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
