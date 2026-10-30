"""ZIP OCR in isolated child processes (own RapidOCR / ONNX / CUDA per process)."""

from __future__ import annotations

import atexit
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

# Set once per worker process via ``init_zip_process_worker`` (Manager proxies).
_gpu_locks: dict[int, Any] | None = None
_worker_stats: Any | None = None
_max_docs_per_worker: int = 0
_worker_started_at: float = 0.0
_worker_docs_processed: int = 0


@dataclass(frozen=True)
class WorkerExitRecord:
    """Metrics emitted when an OCR child process exits."""

    pid: int
    docs_processed: int
    lifetime_sec: float
    exit_reason: str  # "recycle" | "pool_shutdown"


def init_zip_process_worker(
    gpu_locks: dict[int, Any] | None,
    worker_stats: Any | None,
    max_docs_per_worker: int,
) -> None:
    """
    Pool initializer: GPU locks + per-process metrics (no shared OCR engines).

    Registers ``atexit`` to log lifetime and append stats for parent aggregation.
    """
    global _gpu_locks, _worker_stats, _max_docs_per_worker
    global _worker_started_at, _worker_docs_processed

    _gpu_locks = gpu_locks
    _worker_stats = worker_stats
    _max_docs_per_worker = max(0, int(max_docs_per_worker))
    _worker_started_at = time.monotonic()
    _worker_docs_processed = 0

    atexit.register(_record_worker_exit)


def _record_worker_exit() -> None:
    """Log and publish worker lifetime metrics (recycle vs pool shutdown)."""
    global _worker_docs_processed, _worker_started_at

    lifetime = time.monotonic() - _worker_started_at
    docs = _worker_docs_processed
    if (
        _max_docs_per_worker > 0
        and docs >= _max_docs_per_worker
    ):
        reason = "recycle"
    else:
        reason = "pool_shutdown"

    record = WorkerExitRecord(
        pid=os.getpid(),
        docs_processed=docs,
        lifetime_sec=round(lifetime, 3),
        exit_reason=reason,
    )
    logger.info(
        "ZIP OCR worker exit pid=%s reason=%s docs=%d lifetime=%.1fs "
        "(max_docs_per_worker=%s)",
        record.pid,
        record.exit_reason,
        record.docs_processed,
        record.lifetime_sec,
        _max_docs_per_worker or "off",
    )

    if _worker_stats is not None:
        try:
            _worker_stats.append(asdict(record))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not append worker exit stats: %s", exc)


def _bump_worker_docs() -> None:
    global _worker_docs_processed
    _worker_docs_processed += 1


@dataclass(frozen=True)
class ZipWorkItem:
    """Picklable ZIP member job for a child process."""

    job_id: str
    archive_path: str
    extract_root: str
    member_path: str
    rel_path: str
    kind: str
    size_bytes: int
    work_dir: str
    result_dir: str
    pinned_gpu: int | None
    allow_vision_escalation: bool | None
    # Tier assigned by AdaptiveZipQueue ("fast"|"medium"|"escalation"|None).
    # None = legacy path (no adaptive scheduling).
    tier: str | None = None


@dataclass(frozen=True)
class ZipWorkResult:
    ok: bool
    json_path: str | None = None
    txt_path: str | None = None
    error: str | None = None
    error_type: str | None = None


def process_zip_work_item(item: ZipWorkItem) -> ZipWorkResult:
    """Run OCR for one ZIP member in the current process (isolated engines)."""
    from app.core.constants import SourceKind
    from app.services.pipeline import _process_zip_member
    from app.services.zip_queue import ZipMember

    member = ZipMember(
        path=Path(item.member_path),
        rel_path=item.rel_path,
        kind=SourceKind(item.kind),
        size_bytes=item.size_bytes,
    )
    job_id = UUID(item.job_id)
    archive_path = Path(item.archive_path)
    extract_root = Path(item.extract_root)
    work_dir = Path(item.work_dir)
    result_dir = Path(item.result_dir)

    from app.services.zip_streaming import (
        ensure_zip_member_extracted,
        release_zip_member_temp,
    )

    lock = None
    if item.pinned_gpu is not None and _gpu_locks:
        lock = _gpu_locks.get(item.pinned_gpu)

    try:
        ensure_zip_member_extracted(archive_path, extract_root, member)
        if lock is not None:
            lock.acquire()
        try:
            json_path, txt_path = _process_zip_member(
                job_id,
                member,
                work_dir,
                result_dir,
                allow_vision_escalation=item.allow_vision_escalation,
                pinned_gpu=item.pinned_gpu,
            )
        finally:
            if lock is not None:
                lock.release()
        result = ZipWorkResult(
            ok=True,
            json_path=str(json_path),
            txt_path=str(txt_path),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "ZIP worker failed for %s (gpu=%s): %s",
            item.rel_path,
            item.pinned_gpu,
            exc,
        )
        result = ZipWorkResult(
            ok=False,
            error=str(exc),
            error_type=type(exc).__name__,
        )
    finally:
        release_zip_member_temp(extract_root, work_dir, member)
        _bump_worker_docs()

    return result
