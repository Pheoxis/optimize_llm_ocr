#!/usr/bin/env bash
# RQ worker — runs OCR (GPU). API only enqueues when USE_BACKGROUND_WORKER=true.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export DOCOCR_WORKER=1
export USE_BACKGROUND_WORKER=true

QUEUE="${RQ_QUEUE_NAME:-default}"
REDIS="${REDIS_URL:-redis://127.0.0.1:6379/0}"

source .venv/bin/activate
exec rq worker --url "$REDIS" "$QUEUE"
