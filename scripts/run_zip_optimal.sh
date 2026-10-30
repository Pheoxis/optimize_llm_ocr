#!/usr/bin/env bash
# Bulk ZIP benchmark: smart line-level OCR, both GPUs, no full-page VLM on PDF crops.
set -euo pipefail
cd "$(dirname "$0")/.."
export USE_BACKGROUND_WORKER=false
export OCR_ENGINE=smart
export OCR_LINE_LEVEL_ENABLED=true
export PDF_SMART_ESCALATE=false
export FAST_USE_CUDA=true
export OCR_BULK_GPU_IDS=0,1
export OCR_PAGE_PARALLEL_ENABLED=true
export OCR_PAGE_PARALLEL_MIN_PAGES=2
export OCR_PARALLEL_ENABLED=true
export OCR_PARALLEL_MIN_JOBS=4
export ZIP_PARALLEL_WORKERS=0   # auto: 1 worker / GPU (2 total when OCR_BULK_GPU_IDS=0,1)
export ZIP_ORDER_IMAGES_FIRST=true

ZIP="${1:-samples/openpdf-multireceipt-1k.zip}"
exec .venv/bin/python3 scripts/run_sample.py "$ZIP" --no-vlm
