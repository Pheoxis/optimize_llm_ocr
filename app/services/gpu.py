"""CUDA GPU discovery and assignment for OCR (single-GPU fast path vs bulk)."""

from __future__ import annotations

import logging
import subprocess
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_counter = 0
_primary_gpu_cache: int | None = None
_bulk_gpu_ids_cache: list[int] | None = None


def detect_cuda_device_count() -> int:
    """Return the number of visible CUDA devices (0 if none)."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch CUDA probe failed: %s", exc)

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"],
            text=True,
            timeout=5,
        )
        return sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))
    except Exception as exc:  # noqa: BLE001
        logger.debug("nvidia-smi probe failed: %s", exc)
    return 0


def _parse_gpu_ids(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def get_primary_ocr_gpu_id() -> int:
    """GPU for single photos / small jobs (keep away from Ollama if possible)."""
    global _primary_gpu_cache
    if _primary_gpu_cache is not None:
        return _primary_gpu_cache
    from app.core.config import settings

    _primary_gpu_cache = settings.image_ocr_gpu
    return _primary_gpu_cache


def get_bulk_ocr_gpu_ids() -> list[int]:
    """GPU pool for large PDF / ZIP batch OCR (may use both cards)."""
    global _bulk_gpu_ids_cache
    if _bulk_gpu_ids_cache is not None:
        return _bulk_gpu_ids_cache

    from app.core.config import settings

    if settings.ocr_bulk_gpu_ids.strip():
        ids = _parse_gpu_ids(settings.ocr_bulk_gpu_ids)
    else:
        n = detect_cuda_device_count()
        ids = list(range(n)) if n > 0 else [get_primary_ocr_gpu_id()]

    if not ids:
        ids = [get_primary_ocr_gpu_id()]
    _bulk_gpu_ids_cache = ids
    return ids


def get_ocr_gpu_ids() -> list[int]:
    """Backward-compatible alias — returns bulk pool."""
    return get_bulk_ocr_gpu_ids()


def invalidate_gpu_cache() -> None:
    """Clear cached GPU lists (tests / hot reload)."""
    global _primary_gpu_cache, _bulk_gpu_ids_cache
    _primary_gpu_cache = None
    _bulk_gpu_ids_cache = None


def allocate_gpu(gpu_ids: list[int] | None = None) -> int:
    """Round-robin pick the next GPU from ``gpu_ids`` (default: bulk pool)."""
    global _counter
    ids = gpu_ids if gpu_ids is not None else get_bulk_ocr_gpu_ids()
    with _lock:
        gpu_id = ids[_counter % len(ids)]
        _counter += 1
    return gpu_id


def parallel_worker_count(gpu_ids: list[int] | None = None) -> int:
    """Concurrent OCR workers (one slot per GPU in the pool)."""
    from app.core.config import settings

    ids = gpu_ids if gpu_ids is not None else get_bulk_ocr_gpu_ids()
    if not settings.ocr_parallel_enabled or len(ids) < 2:
        return 1
    if settings.ocr_parallel_workers > 0:
        return min(settings.ocr_parallel_workers, len(ids), 32)
    return len(ids)


def gpu_semaphores(gpu_ids: list[int] | None = None) -> dict[int, threading.Semaphore]:
    """One in-flight OCR job per GPU."""
    ids = gpu_ids if gpu_ids is not None else get_bulk_ocr_gpu_ids()
    return {gid: threading.Semaphore(1) for gid in ids}


def should_use_page_parallel(page_count: int) -> bool:
    """Run PDF pages on dedicated GPUs in parallel (page N → GPU N mod pool)."""
    from app.core.config import settings

    if not settings.ocr_page_parallel_enabled:
        return False
    if not settings.ocr_parallel_enabled:
        return False
    if len(get_bulk_ocr_gpu_ids()) < 2:
        return False
    return page_count >= settings.ocr_page_parallel_min_pages


def should_use_parallel_bulk(
    kind: object,
    *,
    page_count: int,
    ocr_job_count: int,
    file_bytes: int,
) -> bool:
    """
    Region-level round-robin across GPUs (fallback when page-parallel is off).

    Page-parallel mode handles multi-page PDFs; this path helps dense single pages
    with many segmented regions.
    """
    from app.core.config import settings
    from app.core.constants import SourceKind

    if not settings.ocr_parallel_enabled:
        return False
    if kind not in (SourceKind.PDF, SourceKind.ZIP):
        return False
    if len(get_bulk_ocr_gpu_ids()) < 2:
        return False

    if kind == SourceKind.PDF and should_use_page_parallel(page_count):
        return False

    if kind == SourceKind.ZIP:
        return True

    if page_count == 1 and ocr_job_count >= 2:
        return True
    if file_bytes >= settings.ocr_parallel_min_bytes:
        return True
    if page_count >= settings.ocr_parallel_min_pages:
        return True
    if ocr_job_count >= settings.ocr_parallel_min_jobs:
        return True
    return False
