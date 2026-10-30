"""
Adaptive ZIP queue scheduling.

Problem
-------
A single FIFO queue means one dense 10-page PDF can block 50 easy JPG
receipts from completing quickly. Mixed ZIP workloads suffer high average
latency and poor progress reporting.

Solution
--------
Three parallel process pools, each sized for its workload:

  FAST        → 1-page PDFs, small JPGs (<400 KB)
  MEDIUM      → multi-page PDFs (2-4 pages), larger images
  ESCALATION  → dense scans, large PDFs (>4 pages), complex photos

Each pool has its own ``ProcessPoolExecutor``. All three run concurrently
inside threads, so FAST files are never blocked by ESCALATION work.

Preserved invariants
--------------------
* Checkpoint (``ZipCheckpointStore``) — ``on_complete`` is still called for
  every member exactly once, in whatever order each pool completes.
* Retry logic — unchanged; lives in the caller's ``on_complete`` callback.
* Result ordering — the caller appends to ``result_files`` as results arrive;
  final order is arrival order (same as the original FIFO pool).
* ``ZipWorkItem`` pickling — tier is passed as a string field; existing
  workers ignore it gracefully (default ``None``).

Usage
-----
    from app.services.zip_adaptive_queue import AdaptiveZipQueue

    queue = AdaptiveZipQueue(members)
    metrics = queue.run_in_process_pool(
        job_id=...,
        archive_path=...,
        ...
        on_complete=_on_pool_result,
    )
    print(queue.tier_summary())
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from app.core.config import settings
from app.core.constants import SourceKind
from app.services.gpu import get_bulk_ocr_gpu_ids, parallel_worker_count
from app.services.zip_queue import (
    ZipMember,
    ZipPoolRecycleMetrics,
    _create_gpu_locks,
    _pool_max_tasks_per_child,
    _shutdown_pool,
    build_zip_work_item,
    summarize_worker_recycle_metrics,
    write_worker_recycle_report,
)

logger = logging.getLogger(__name__)


# ── tier enum ─────────────────────────────────────────────────────────────────


class ZipTier(IntEnum):
    FAST = 0        # easy JPG, 1-page small PDF
    MEDIUM = 1      # multi-page PDF, larger image
    ESCALATION = 2  # complex scan, premium VLM needed


_TIER_NAME: dict[ZipTier, str] = {
    ZipTier.FAST: "fast",
    ZipTier.MEDIUM: "medium",
    ZipTier.ESCALATION: "escalation",
}

_TIER_FROM_NAME: dict[str, ZipTier] = {v: k for k, v in _TIER_NAME.items()}


# ── classified member ─────────────────────────────────────────────────────────


@dataclass
class ClassifiedMember:
    """``ZipMember`` augmented with its assigned tier and classification reason."""
    member: ZipMember
    tier: ZipTier
    reason: str
    difficulty_score: float = 0.0  # from probe (0 when probe disabled)


# ── per-tier metrics ──────────────────────────────────────────────────────────


@dataclass
class TierMetrics:
    tier: str
    total: int = 0
    completed: int = 0
    failed: int = 0
    timed_out: int = 0
    elapsed_sec: float = 0.0
    worker_restarts: int = 0

    @property
    def throughput(self) -> float:
        if self.elapsed_sec <= 0:
            return 0.0
        return self.completed / self.elapsed_sec

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "timed_out": self.timed_out,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "throughput_per_sec": round(self.throughput, 3),
            "worker_restarts": self.worker_restarts,
        }


@dataclass
class AdaptiveZipMetrics:
    """Aggregated metrics for all three tiers of one ZIP job."""
    fast: TierMetrics = field(default_factory=lambda: TierMetrics("fast"))
    medium: TierMetrics = field(default_factory=lambda: TierMetrics("medium"))
    escalation: TierMetrics = field(default_factory=lambda: TierMetrics("escalation"))
    classify_ms: float = 0.0
    total_elapsed_sec: float = 0.0
    gpu_scheduler: dict[str, Any] = field(default_factory=dict)

    def tier(self, t: ZipTier) -> TierMetrics:
        return [self.fast, self.medium, self.escalation][int(t)]

    def to_dict(self) -> dict[str, Any]:
        out = {
            "fast": self.fast.to_dict(),
            "medium": self.medium.to_dict(),
            "escalation": self.escalation.to_dict(),
            "classify_ms": round(self.classify_ms, 2),
            "total_elapsed_sec": round(self.total_elapsed_sec, 2),
        }
        if self.gpu_scheduler:
            out["gpu_scheduler"] = self.gpu_scheduler
        return out

    def log_summary(self) -> None:
        logger.info(
            "AdaptiveZipQueue │ classify=%.0fms  total=%.1fs  │  "
            "FAST(%d) %.0f/s  MEDIUM(%d) %.0f/s  ESCALATION(%d) %.0f/s  │  "
            "failed=%d timed_out=%d restarts=%d",
            self.classify_ms,
            self.total_elapsed_sec,
            self.fast.total, self.fast.throughput,
            self.medium.total, self.medium.throughput,
            self.escalation.total, self.escalation.throughput,
            self.fast.failed + self.medium.failed + self.escalation.failed,
            self.fast.timed_out + self.medium.timed_out + self.escalation.timed_out,
            self.fast.worker_restarts + self.medium.worker_restarts + self.escalation.worker_restarts,
        )


# ── PDF page count probe ──────────────────────────────────────────────────────


def _pdf_page_count_fast(path: Path) -> int:
    """Return page count without rendering (fast fitz metadata probe)."""
    try:
        import fitz
        doc = fitz.open(path)
        n = doc.page_count
        doc.close()
        return n
    except Exception:  # noqa: BLE001
        return 1


# ── member classification ─────────────────────────────────────────────────────


def _classify_member_heuristic(member: ZipMember) -> ClassifiedMember:
    """Fast size/kind/page-count heuristic — no image rendering needed."""
    kind = member.kind
    size = member.size_bytes

    if kind == SourceKind.IMAGE:
        if size < settings.zip_aq_fast_max_image_bytes:
            return ClassifiedMember(
                member=member, tier=ZipTier.FAST,
                reason=f"image<{settings.zip_aq_fast_max_image_bytes // 1024}KB",
            )
        if size < settings.zip_aq_medium_max_image_bytes:
            return ClassifiedMember(
                member=member, tier=ZipTier.MEDIUM,
                reason=f"image<{settings.zip_aq_medium_max_image_bytes // 1024}KB",
            )
        return ClassifiedMember(
            member=member, tier=ZipTier.ESCALATION,
            reason=f"large_image({size // 1024}KB)",
        )

    if kind == SourceKind.PDF:
        pages = _pdf_page_count_fast(member.path)
        if pages <= settings.zip_aq_fast_max_pdf_pages and size < 300_000:
            return ClassifiedMember(
                member=member, tier=ZipTier.FAST,
                reason=f"pdf_1page_{size // 1024}KB",
            )
        if pages <= settings.zip_aq_medium_max_pdf_pages:
            return ClassifiedMember(
                member=member, tier=ZipTier.MEDIUM,
                reason=f"pdf_{pages}pages",
            )
        return ClassifiedMember(
            member=member, tier=ZipTier.ESCALATION,
            reason=f"pdf_{pages}pages",
        )

    # Unknown kind → MEDIUM as safe default
    return ClassifiedMember(
        member=member, tier=ZipTier.MEDIUM, reason="unknown_kind",
    )


def _classify_member_with_probe(member: ZipMember) -> ClassifiedMember:
    """
    Classify using ``ocr_difficulty_analyzer`` (adds ~12–18 ms per member).

    Maps 3-tier DifficultyDecision → ZipTier:
      EASY   → FAST
      MEDIUM → MEDIUM
      HARD   → ESCALATION
    """
    try:
        from app.services.ocr_difficulty_analyzer import Tier, classify_zip_member
        dd = classify_zip_member(member)
        _zip_tier_map = {
            Tier.EASY: ZipTier.FAST,
            Tier.MEDIUM: ZipTier.MEDIUM,
            Tier.HARD: ZipTier.ESCALATION,
        }
        tier = _zip_tier_map[dd.tier]
        reason = (
            f"probe:{dd.tier.value}"
            f" conf={dd.confidence:.2f}"
            f" {dd.routing_time_ms:.0f}ms"
        )
        # Extract numeric score from first reason (e.g. "score=42.3→medium")
        score = 0.0
        for r in dd.reasons:
            if r.startswith("score="):
                try:
                    score = float(r.split("=")[1].split("→")[0])
                except (IndexError, ValueError):
                    pass
                break
        return ClassifiedMember(
            member=member, tier=tier, reason=reason, difficulty_score=score,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Difficulty probe failed for %s: %s", member.path.name, exc)

    # Fallback to heuristic
    return _classify_member_heuristic(member)


def classify_members(
    members: list[ZipMember],
    *,
    use_probe: bool | None = None,
) -> list[ClassifiedMember]:
    """
    Classify every ZIP member into a tier.

    Parameters
    ----------
    members : list[ZipMember]
    use_probe : bool | None
        Override ``settings.zip_aq_use_difficulty_probe``.
    """
    probe = use_probe if use_probe is not None else settings.zip_aq_use_difficulty_probe
    classify_fn = _classify_member_with_probe if probe else _classify_member_heuristic

    classified = [classify_fn(m) for m in members]

    # Log tier distribution
    dist = {t.name: 0 for t in ZipTier}
    for c in classified:
        dist[c.tier.name] += 1
    logger.info(
        "ZIP member classification: total=%d  FAST=%d MEDIUM=%d ESCALATION=%d",
        len(classified), dist["FAST"], dist["MEDIUM"], dist["ESCALATION"],
    )
    return classified


# ── tier-specific work item builder ──────────────────────────────────────────


def _build_tier_work_item(
    job_id: UUID,
    cm: ClassifiedMember,
    *,
    archive_path: Path,
    extract_root: Path,
    work_dir: Path,
    result_dir: Path,
    pinned_gpu: int | None,
    allow_vision_escalation: bool | None,
) -> Any:
    """Build a work item with the tier string embedded."""
    from app.services.zip_worker import ZipWorkItem

    member = cm.member
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
        tier=_TIER_NAME[cm.tier],
    )


# ── per-tier pool runner ──────────────────────────────────────────────────────


def _tier_workers(tier: ZipTier) -> int:
    """Resolve worker count for a tier (respects GPU count for default 0)."""
    if tier == ZipTier.FAST:
        n = settings.zip_aq_fast_workers
        max_n = max(n, 3) if n > 0 else parallel_worker_count(get_bulk_ocr_gpu_ids()) + 1
    elif tier == ZipTier.MEDIUM:
        n = settings.zip_aq_medium_workers
        max_n = max(n, 2) if n > 0 else parallel_worker_count(get_bulk_ocr_gpu_ids())
    else:
        n = settings.zip_aq_escalation_workers
        max_n = max(n, 2) if n > 0 else parallel_worker_count(get_bulk_ocr_gpu_ids())

    if n <= 0:
        n = parallel_worker_count(get_bulk_ocr_gpu_ids())

    # Dynamic GPU scheduler: adjust workers based on queue load
    if settings.gpu_scheduler_enabled and settings.dynamic_gpu_scheduler_enabled:
        try:
            from app.services.dynamic_gpu_scheduler import get_dynamic_scheduler

            sched = get_dynamic_scheduler()
            return sched.adaptive_zip_workers(
                _TIER_NAME[tier],
                default_workers=n,
                max_workers=max_n,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Adaptive ZIP worker allocation skipped: %s", exc)

    return n


def _tier_timeout(tier: ZipTier) -> float:
    """Per-member timeout seconds for a tier (0 → global default)."""
    if tier == ZipTier.FAST:
        t = settings.zip_aq_fast_timeout_sec
    elif tier == ZipTier.MEDIUM:
        t = settings.zip_aq_medium_timeout_sec
    else:
        t = settings.zip_aq_escalation_timeout_sec
    return t if t > 0 else settings.zip_member_timeout_sec


def _run_tier_pool(
    tier: ZipTier,
    items_and_members: list[tuple[Any, ZipMember]],
    *,
    gpu_ids: list[int],
    manager_gpu_locks: dict[int, Any] | None,
    manager_worker_stats: Any,
    max_tasks: int | None,
    on_complete: Callable[[ZipMember, Any], None],
    progress_lock: threading.Lock,
    tier_metrics: TierMetrics,
    result_dir: Path,
) -> ZipPoolRecycleMetrics:
    """Run one tier's process pool. Blocks until all items complete."""
    from app.services.zip_worker import (
        ZipWorkResult,
        init_zip_process_worker,
        process_zip_work_item,
    )

    n_workers = min(_tier_workers(tier), len(items_and_members))
    timeout = _tier_timeout(tier)
    max_docs = settings.max_docs_per_worker

    mp_ctx = multiprocessing.get_context("spawn")

    pool_kwargs: dict[str, Any] = {
        "max_workers": n_workers,
        "mp_context": mp_ctx,
        "initializer": init_zip_process_worker,
        "initargs": (manager_gpu_locks, manager_worker_stats, max_docs),
    }
    if max_tasks is not None:
        pool_kwargs["max_tasks_per_child"] = max_tasks

    logger.info(
        "ZIP tier[%s]: %d members, %d workers, timeout=%.0fs",
        _TIER_NAME[tier], len(items_and_members), n_workers, timeout,
    )

    t0 = time.monotonic()
    item_by_rel = {item.rel_path: member for item, member in items_and_members}

    pool: ProcessPoolExecutor | None = None
    try:
        pool = ProcessPoolExecutor(**pool_kwargs)
        futures: dict[Future, Any] = {
            pool.submit(process_zip_work_item, item): item
            for item, _member in items_and_members
        }

        for fut in as_completed(futures):
            item = futures[fut]
            member = item_by_rel[item.rel_path]
            try:
                result: ZipWorkResult = fut.result(timeout=timeout)
            except FuturesTimeoutError:
                logger.error(
                    "ZIP tier[%s] member %s timed out after %.0fs",
                    _TIER_NAME[tier], item.rel_path, timeout,
                )
                fut.cancel()
                tier_metrics.timed_out += 1
                result = ZipWorkResult(
                    ok=False,
                    error=f"OCR timed out after {timeout:.0f}s",
                    error_type="TimeoutError",
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "ZIP tier[%s] member %s raised: %s",
                    _TIER_NAME[tier], item.rel_path, exc,
                )
                tier_metrics.failed += 1
                result = ZipWorkResult(
                    ok=False, error=str(exc), error_type=type(exc).__name__,
                )

            if result.ok:
                tier_metrics.completed += 1
            else:
                tier_metrics.failed += 1

            with progress_lock:
                on_complete(member, result)

    finally:
        if pool is not None:
            _shutdown_pool(pool, cancel_pending=False)

    tier_metrics.elapsed_sec = time.monotonic() - t0
    recycle = summarize_worker_recycle_metrics(
        list(manager_worker_stats) if manager_worker_stats is not None else [],
        max_docs_per_worker=max_docs,
        pool_workers=n_workers,
        members_total=len(items_and_members),
    )
    tier_metrics.worker_restarts = recycle.worker_restart_count
    return recycle


# ── adaptive queue ────────────────────────────────────────────────────────────


class AdaptiveZipQueue:
    """
    Three-tier ZIP processing queue.

    FAST, MEDIUM, and ESCALATION pools run concurrently so that simple
    documents are never blocked by complex ones.

    Parameters
    ----------
    members : list[ZipMember]
        All pending members (already checkpoint-filtered by the caller).
    classified : list[ClassifiedMember] | None
        Pre-classified members (skips re-classification if provided).
    """

    def __init__(
        self,
        members: list[ZipMember],
        classified: list[ClassifiedMember] | None = None,
    ) -> None:
        self.members = members
        self._classified = classified

    # ── public properties ───────────────────────────────────────────────────

    @property
    def classified(self) -> list[ClassifiedMember]:
        if self._classified is None:
            self._classified = classify_members(self.members)
        return self._classified

    @property
    def max_workers(self) -> int:
        """Total workers across all tiers (for compatibility logging)."""
        return (
            _tier_workers(ZipTier.FAST)
            + _tier_workers(ZipTier.MEDIUM)
            + _tier_workers(ZipTier.ESCALATION)
        )

    def tier_summary(self) -> str:
        dist = {t: 0 for t in ZipTier}
        for c in self.classified:
            dist[c.tier] += 1
        return (
            f"FAST={dist[ZipTier.FAST]} "
            f"MEDIUM={dist[ZipTier.MEDIUM]} "
            f"ESCALATION={dist[ZipTier.ESCALATION]}"
        )

    # ── main entry point ────────────────────────────────────────────────────

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
        on_complete: Callable[[ZipMember, Any], None],
    ) -> ZipPoolRecycleMetrics:
        """
        Run all three pools concurrently.

        Returns a ``ZipPoolRecycleMetrics`` instance compatible with the
        original ``ZipProcessingQueue.run_in_process_pool`` signature.
        Detailed per-tier metrics are stored in ``self.metrics``.
        """
        if not self.members:
            self.metrics = AdaptiveZipMetrics()
            return ZipPoolRecycleMetrics(
                max_docs_per_worker=settings.max_docs_per_worker,
                pool_workers=self.max_workers,
                members_total=0,
                worker_exit_count=0,
                worker_restart_count=0,
                total_docs_processed=0,
            )

        t0 = time.monotonic()
        classify_t0 = time.perf_counter()
        by_tier: dict[ZipTier, list[ClassifiedMember]] = {
            ZipTier.FAST: [],
            ZipTier.MEDIUM: [],
            ZipTier.ESCALATION: [],
        }
        for cm in self.classified:
            by_tier[cm.tier].append(cm)
        classify_ms = (time.perf_counter() - classify_t0) * 1000.0

        aq_metrics = AdaptiveZipMetrics(classify_ms=classify_ms)
        for tier, cms in by_tier.items():
            aq_metrics.tier(tier).total = len(cms)

        # Shared multiprocessing infrastructure (single manager for all pools)
        mp_ctx = multiprocessing.get_context("spawn")
        manager = mp_ctx.Manager()
        worker_stats = manager.list()
        manager_gpu_locks: dict[int, Any] | None = None
        if gpu_ids:
            manager_gpu_locks = _create_gpu_locks(manager, gpu_ids)
        max_tasks = _pool_max_tasks_per_child()

        progress_lock = threading.Lock()
        all_recycles: list[ZipPoolRecycleMetrics] = []

        def _run_one_tier(tier: ZipTier) -> None:
            cms = by_tier[tier]
            if not cms:
                return

            items_and_members = [
                (
                    _build_tier_work_item(
                        job_id, cm,
                        archive_path=archive_path,
                        extract_root=extract_root,
                        work_dir=work_dir,
                        result_dir=result_dir,
                        pinned_gpu=pinned_for(cm.member),
                        allow_vision_escalation=allow_vision_escalation,
                    ),
                    cm.member,
                )
                for cm in cms
            ]

            recycle = _run_tier_pool(
                tier,
                items_and_members,
                gpu_ids=gpu_ids,
                manager_gpu_locks=manager_gpu_locks,
                manager_worker_stats=worker_stats,
                max_tasks=max_tasks,
                on_complete=on_complete,
                progress_lock=progress_lock,
                tier_metrics=aq_metrics.tier(tier),
                result_dir=result_dir,
            )
            with progress_lock:
                all_recycles.append(recycle)

        # Launch all non-empty tiers concurrently
        active_tiers = [t for t in ZipTier if by_tier[t]]
        threads = [
            threading.Thread(
                target=_run_one_tier,
                args=(tier,),
                name=f"zip_tier_{_TIER_NAME[tier]}",
                daemon=True,
            )
            for tier in active_tiers
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        aq_metrics.total_elapsed_sec = time.monotonic() - t0

        if settings.gpu_scheduler_enabled:
            try:
                from app.services.gpu_scheduler import scheduler_metrics
                aq_metrics.gpu_scheduler = scheduler_metrics()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not attach GPU scheduler metrics: %s", exc)

        aq_metrics.log_summary()
        self.metrics = aq_metrics

        # Persist per-tier summary alongside the standard worker recycle report
        _write_tier_report(result_dir, aq_metrics)

        try:
            write_worker_recycle_report(
                result_dir,
                _merge_recycle_metrics(all_recycles, len(self.members)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not write worker_recycle.json: %s", exc)

        try:
            manager.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ZIP adaptive manager shutdown: %s", exc)

        return _merge_recycle_metrics(all_recycles, len(self.members))


# ── helpers ───────────────────────────────────────────────────────────────────


def _merge_recycle_metrics(
    parts: list[ZipPoolRecycleMetrics],
    members_total: int,
) -> ZipPoolRecycleMetrics:
    """Combine per-tier recycle metrics into one summary."""
    max_docs = settings.max_docs_per_worker
    total_workers = sum(p.pool_workers for p in parts)
    exits = []
    for p in parts:
        exits.extend(p.worker_exits)
    total_docs = sum(p.total_docs_processed for p in parts)
    restarts = sum(p.worker_restart_count for p in parts)
    return ZipPoolRecycleMetrics(
        max_docs_per_worker=max_docs,
        pool_workers=total_workers,
        members_total=members_total,
        worker_exit_count=len(exits),
        worker_restart_count=restarts,
        total_docs_processed=total_docs,
        worker_exits=exits,
    )


def _write_tier_report(result_dir: Path, metrics: AdaptiveZipMetrics) -> None:
    """Write tier distribution + timing to ``zip_tier_report.json``."""
    from app.services.zip_checkpoint import atomic_write_json
    try:
        path = result_dir / "zip_tier_report.json"
        atomic_write_json(path, metrics.to_dict())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not write zip_tier_report.json: %s", exc)
