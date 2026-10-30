#!/usr/bin/env bash
# API for LAN / Tailscale — bind all interfaces, trust proxy headers.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${API_PORT:-8000}" \
  --forwarded-allow-ips '*' \
  "$@"
