"""ZIP member scheduling: order + bounded parallel processing (process pool)."""

from __future__ import annotations

import json
import logging
import multiprocessing
import threading
from concurrent.futures import (
    ProcessPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from dataclasses import asdict, dataclass, field
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from uuid import UUID

from app.core.config import settings
from app.core.constants import SourceKind
from app.services.gpu import get_bulk_ocr_gpu_ids, parallel_worker_count

if TYPE_CHECKING:
    from app.services.zip_worker import ZipWorkResult

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"})

R = TypeVar("R")

_progress_lock = threading.Lock()


@dataclass(frozen=True)
class ZipPoolRecycleMetrics:
    """Aggregated OCR worker recycle stats for one ZIP process-pool run."""

    max_docs_per_worker: int
    pool_workers: int
    members_total: int
    worker_exit_count: int
    worker_restart_count: int
    total_docs_processed: int
    worker_exits: list[dict] = field(default_factory=list)

    @property
    def avg_worker_lifetime_sec(self) -> float:
        if not self.worker_exits:
            return 0.0
        return sum(e["lifetime_sec"] for e in self.worker_exits) / len(self.worker_exits)

    @property
    def avg_docs_per_worker_exit(self) -> float:
        if not self.worker_exits:
            return 0.0
        return self.total_docs_processed / len(self.worker_exits)


def summarize_worker_recycle_metrics(
    worker_exits: list[dict],
    *,
    max_docs_per_worker: int,
    pool_workers: int,
    members_total: int,
) -> ZipPoolRecycleMetrics:
    """Build recycle summary from child ``WorkerExitRecord`` dicts."""
    restarts = sum(
        1 for e in worker_exits if e.get("exit_reason") == "recycle"
    )
    total_docs = sum(int(e.get("docs_processed", 0)) for e in worker_exits)
    return ZipPoolRecycleMetrics(
        max_docs_per_worker=max_docs_per_worker,
        pool_workers=pool_workers,
        members_total=members_total,
        worker_exit_count=len(worker_exits),
        worker_restart_count=restarts,
        total_docs_processed=total_docs,
        worker_exits=worker_exits,
    )


def _pool_max_tasks_per_child() -> int | None:
    """``ProcessPoolExecutor`` recycle limit (``None`` = disabled)."""
    limit = settings.max_docs_per_worker
    return limit if limit > 0 else None


def _log_recycle_metrics(metrics: ZipPoolRecycleMetrics) -> None:
    logger.info(
        "ZIP worker recycle summary: restarts=%d exits=%d members=%d "
        "docs_logged=%d avg_lifetime=%.1fs avg_docs/exit=%.1f "
        "(max_docs_per_worker=%s, pool_workers=%d)",
        metrics.worker_restart_count,
        metrics.worker_exit_count,
        metrics.members_total,
        metrics.total_docs_processed,
        metrics.avg_worker_lifetime_sec,
        metrics.avg_docs_per_worker_exit,
        metrics.max_docs_per_worker or "off",
        metrics.pool_workers,
    )


def write_worker_recycle_report(
    result_dir: Path,
    metrics: ZipPoolRecycleMetrics,
) -> Path:
    """Persist recycle metrics next to ZIP job results."""
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "worker_recycle.json"
    payload = {
        **asdict(metrics),
        "avg_worker_lifetime_sec": round(metrics.avg_worker_lifetime_sec, 3),
        "avg_docs_per_worker_exit": round(metrics.avg_docs_per_worker_exit, 3),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True)
class ZipMember:
    """One document extracted from a ZIP archive."""

    path: Path
    rel_path: str  # posix path relative to extract root (used as result subfolder)
    kind: SourceKind
    size_bytes: int


def _member_kind(path: Path) -> SourceKind:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return SourceKind.PDF
    if suffix in IMAGE_EXTENSIONS:
        return SourceKind.IMAGE
    raise ValueError(f"Unsupported ZIP member: {suffix}")


def _sort_zip_members(members: list[ZipMember]) -> list[ZipMember]:
    """Images first (fast progress), then PDFs by size ascending."""
    if not settings.zip_order_images_first:
        return sorted(members, key=lambda m: (m.kind != SourceKind.PDF, m.size_bytes))

    images = sorted(
        (m for m in members if m.kind == SourceKind.IMAGE),
        key=lambda m: m.size_bytes,
    )
    pdfs = sorted(
        (m for m in members if m.kind == SourceKind.PDF),
        key=lambda m: m.size_bytes,
    )
    return [*images, *pdfs]


def plan_zip_members(files: list[Path], extract_root: Path) -> list[ZipMember]:
    """Build processing plan from already-extracted files on disk."""
    root = extract_root.resolve()
    members: list[ZipMember] = []
    for path in files:
        resolved = path.resolve()
        try:
            rel = resolved.relative_to(root).as_posix()
        except ValueError:
            rel = resolved.name
        kind = _member_kind(resolved)
        members.append(
            ZipMember(
                path=resolved,
                rel_path=rel,
                kind=kind,
                size_bytes=resolved.stat().st_size,
            )
        )
    return _sort_zip_members(members)


def resolve_zip_parallel_workers(members: list[ZipMember]) -> int:
    """
    Auto: up to one ZIP worker per bulk GPU (images + PDFs).

    With ``ProcessPoolExecutor``, each worker is a separate process with its own
    ONNX/RapidOCR stack. Cross-process ``Manager`` locks enforce at most one
    in-flight OCR job per GPU when pinning is enabled.
    """
    if settings.zip_parallel_workers > 0:
        cap = settings.zip_parallel_workers
    else:
        cap = parallel_worker_count(get_bulk_ocr_gpu_ids())

    if (
        settings.zip_pdf_max_parallel > 0
        and members
        and all(m.kind == SourceKind.PDF for m in members)
    ):
        cap = min(cap, settings.zip_pdf_max_parallel)

    return max(1, min(cap, len(members)))


def build_zip_work_item(
    job_id: UUID,
    member: ZipMember,
    *,
    archive_path: Path,
    extract_root: Path,
    work_dir: Path,
    result_dir: Path,
    pinned_gpu: int | None,
    allow_vision_escalation: bool | None,
) -> Any:
    """Build a picklable work item for ``process_zip_work_item``."""
    from app.services.zip_worker import ZipWorkItem

    return ZipWorkItem(
        job_id=str(job_id),
        archive_path=str(archive_path),
        extract_root=str(extract_root),
        member_path=str(member.path),
        rel_path=member.rel_path,
        kind=member.kind.value,
        size_bytes=member.size_bytes,
        work_dir=str(work_dir),
        result_dir=str(result_dir),
        pinned_gpu=pinned_gpu,
        allow_vision_escalation=allow_vision_escalation,
    )


def _create_gpu_locks(
    manager: SyncManager,
    gpu_ids: list[int],
) -> dict[int, Any]:
    return {gid: manager.Lock() for gid in gpu_ids}


def _shutdown_pool(
    pool: ProcessPoolExecutor,
    *,
    cancel_pending: bool,
) -> None:
    """Graceful shutdown with a bounded wait for running workers."""
    grace = settings.zip_pool_shutdown_grace_sec
    try:
        pool.shutdown(wait=True, cancel_futures=cancel_pending)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ZIP process pool shutdown(wait=True) failed: %s", exc)
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception as exc2:  # noqa: BLE001
            logger.warning("ZIP process pool forced shutdown failed: %s", exc2)
            return
        try:
            pool.shutdown(wait=True, cancel_futures=False)
        except Exception as exc3:  # noqa: BLE001
            logger.warning(
                "ZIP process pool final shutdown failed after %.0fs grace: %s",
                grace,
                exc3,
            )


class ZipProcessingQueue:
    """FIFO queue with optional parallel process workers for ZIP members."""

    def __init__(
        self,
        members: list[ZipMember],
        *,
        max_workers: int | None = None,
    ) -> None:
        self.members = members
        self.max_workers = max_workers or resolve_zip_parallel_workers(members)

    def run(
        self,
        processor: Callable[[ZipMember], R],
        *,
        on_complete: Callable[[ZipMember, R], None] | None = None,
    ) -> list[R]:
        """
        Process all members in the parent process (single worker only).

        For ``max_workers > 1``, use ``run_in_process_pool`` instead.
        """
        if not self.members:
            return []

        if self.max_workers > 1:
            raise RuntimeError(
                "ZIP parallel OCR requires run_in_process_pool() "
                f"(max_workers={self.max_workers})",
            )

        logger.info(
            "ZIP queue: %d files, 1 worker in-process (images first=%s)",
            len(self.members),
            settings.zip_order_images_first,
        )

        results: list[R] = []
        for member in self.members:
            out = processor(member)
            if on_complete:
                with _progress_lock:
                    on_complete(member, out)
            results.append(out)
        return results

    def run_in_process_pool(
        self,
        *,
        job_id: UUID,
        archive_path: Path,
        extract_root: Path,
        work_dir: Path,
        result_dir: Path,
        pinned_for: Callable[[ZipMember], int | None],
        allow_vision_escalation: bool | None,
        gpu_ids: list[int],
        on_complete: Callable[[ZipMember, ZipWorkResult], None],
    ) -> ZipPoolRecycleMetrics:
        """
        Process members in child processes (spawn context, isolated OCR stacks).

        Workers are recycled after ``max_docs_per_worker`` documents (native/VRAM
        fragmentation). ``on_complete`` runs in the parent under a lock (safe for
        ``job_store``). Returns aggregated recycle metrics.
        """
        if not self.members:
            return ZipPoolRecycleMetrics(
                max_docs_per_worker=settings.max_docs_per_worker,
                pool_workers=self.max_workers,
                members_total=0,
                worker_exit_count=0,
                worker_restart_count=0,
                total_docs_processed=0,
            )

        from app.services.zip_worker import (
            ZipWorkResult,
            init_zip_process_worker,
            process_zip_work_item,
        )

        work_items = [
            build_zip_work_item(
                job_id,
                member,
                archive_path=archive_path,
                extract_root=extract_root,
                work_dir=work_dir,
                result_dir=result_dir,
                pinned_gpu=pinned_for(member),
                allow_vision_escalation=allow_vision_escalation,
            )
            for member in self.members
        ]
        item_by_member = {m.rel_path: m for m in self.members}

        max_docs = settings.max_docs_per_worker
        max_tasks = _pool_max_tasks_per_child()
        logger.info(
            "ZIP process pool: %d files, %d workers, spawn, gpu_locks=%s, "
            "max_docs_per_worker=%s",
            len(self.members),
            self.max_workers,
            bool(gpu_ids),
            max_docs if max_docs > 0 else "off",
        )

        mp_ctx = multiprocessing.get_context("spawn")
        manager: SyncManager | None = mp_ctx.Manager()
        worker_stats = manager.list()
        gpu_locks: dict[int, Any] | None = None
        if gpu_ids:
            gpu_locks = _create_gpu_locks(manager, gpu_ids)

        member_timeout = settings.zip_member_timeout_sec
        pool: ProcessPoolExecutor | None = None
        recycle_metrics = ZipPoolRecycleMetrics(
            max_docs_per_worker=max_docs,
            pool_workers=self.max_workers,
            members_total=len(self.members),
            worker_exit_count=0,
            worker_restart_count=0,
            total_docs_processed=0,
        )

        try:
            pool_kwargs: dict[str, Any] = {
                "max_workers": self.max_workers,
                "mp_context": mp_ctx,
                "initializer": init_zip_process_worker,
                "initargs": (gpu_locks, worker_stats, max_docs),
            }
            if max_tasks is not None:
                pool_kwargs["max_tasks_per_child"] = max_tasks
            pool = ProcessPoolExecutor(**pool_kwargs)
            futures = {
                pool.submit(process_zip_work_item, item): item
                for item in work_items
            }
            for fut in as_completed(futures):
                item = futures[fut]
                member = item_by_member[item.rel_path]
                try:
                    result: ZipWorkResult = fut.result(timeout=member_timeout)
                except FuturesTimeoutError:
                    logger.error(
                        "ZIP member %s timed out after %.0fs",
                        item.rel_path,
                        member_timeout,
                    )
                    fut.cancel()
                    result = ZipWorkResult(
                        ok=False,
                        error=f"OCR timed out after {member_timeout:.0f}s",
                        error_type="TimeoutError",
                    )
                with _progress_lock:
                    on_complete(member, result)
        finally:
            if pool is not None:
                _shutdown_pool(pool, cancel_pending=False)
            recycle_metrics = summarize_worker_recycle_metrics(
                list(worker_stats),
                max_docs_per_worker=max_docs,
                pool_workers=self.max_workers,
                members_total=len(self.members),
            )
            _log_recycle_metrics(recycle_metrics)
            try:
                write_worker_recycle_report(result_dir, recycle_metrics)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not write worker_recycle.json: %s", exc)
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("ZIP Manager shutdown: %s", exc)

        return recycle_metrics
