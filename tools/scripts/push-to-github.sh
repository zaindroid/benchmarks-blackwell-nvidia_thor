#!/usr/bin/env bash
# Push the ThorAI Platform repository to GitHub.
#
# Usage:
#   REPO=yourusername/thor-ai-platform ./tools/scripts/push-to-github.sh
#   (or set REPO inside the file; defaults to zaindroid/thor-ai-platform)
#
# Creates the GitHub repo with `gh` when available, adds the remote,
# and pushes main. Run `gh auth login` first if gh is installed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REPO="${REPO:-zaindroid/thor-ai-platform}"
REMOTE="origin"

if [ -z "$(git remote get-url $REMOTE 2>/dev/null || true)" ]; then
  echo "==> Creating GitHub repo: $REPO"
  if command -v gh >/dev/null 2>&1; then
    gh repo create "$REPO" --public --source=. --remote=$REMOTE --push \
      --description "Open-source benchmarking and deployment platform for NVIDIA DRIVE Thor (MCP server, BEV/VLM references, optimization toolchain)" \
      || true
  else
    echo "gh not installed — create the repo on github.com manually, then:"
    echo "  git remote add $REMOTE https://github.com/$REPO.git"
    echo "  git push -u $REMOTE main"
    exit 1
  fi
fi

echo "==> Pushing main to $REMOTE ($REPO)"
git push -u $REMOTE main

echo "==> Done: https://github.com/$REPO"
echo "Next: follow docs/launch-plan.md and docs/hosting-zorc.md"
