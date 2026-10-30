"""Shared GPU slots for OCR — one in-flight job per card across ZIP/PDF/image workers."""

from __future__ import annotations

import logging
import threading

from app.services.gpu import get_bulk_ocr_gpu_ids

logger = logging.getLogger(__name__)

_pool_lock = threading.Lock()
_semaphores: dict[int, threading.Semaphore] = {}


def ocr_gpu_semaphores(gpu_ids: list[int] | None = None) -> dict[int, threading.Semaphore]:
    """Process-wide semaphores: ZIP workers + PDF page workers share the same pool."""
    ids = gpu_ids if gpu_ids is not None else get_bulk_ocr_gpu_ids()
    with _pool_lock:
        for gid in ids:
            if gid not in _semaphores:
                _semaphores[gid] = threading.Semaphore(1)
    return {gid: _semaphores[gid] for gid in ids}


def gpu_for_page(page_idx: int, gpu_ids: list[int]) -> int:
    """Pin page ``page_idx`` (1-based) to a stable GPU in the pool."""
    if not gpu_ids:
        return 0
    return gpu_ids[(page_idx - 1) % len(gpu_ids)]


def reset_ocr_gpu_pool() -> None:
    """Clear semaphores (tests only)."""
    global _semaphores
    with _pool_lock:
        _semaphores = {}
