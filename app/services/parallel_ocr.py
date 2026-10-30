"""Run independent OCR jobs — sequential, region-parallel, or page-parallel."""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from app.services.gpu import (
    allocate_gpu,
    get_primary_ocr_gpu_id,
    parallel_worker_count,
)
from app.services.gpu_pool import gpu_for_page, ocr_gpu_semaphores
from app.services.oom import with_oom_retry

logger = logging.getLogger(__name__)

_PAGE_KEY_RE = re.compile(r"^p(\d+):")


@dataclass(frozen=True)
class OcrJob:
    """Single OCR unit (one image path)."""

    key: str
    path: Path
    lang: str | None = None
    allow_vision_escalation: bool | None = None
    gpu_id: int | None = None
    line_level: bool | None = None
    profile_name: str | None = None  # active pipeline profile per job
    priority: int | None = None  # TaskPriority value; None → NORMAL


@dataclass
class OcrJobResult:
    key: str
    text: str
    confidence: float | None
    meta: dict


def page_idx_from_key(key: str) -> int | None:
    m = _PAGE_KEY_RE.match(key)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _run_one(
    job: OcrJob,
    semaphores: dict[int, threading.Semaphore],
    gpu_ids: list[int],
) -> OcrJobResult:
    from app.core.config import settings
    from app.services.ocr import ocr_image

    gpu_id = job.gpu_id if job.gpu_id is not None else allocate_gpu(gpu_ids)
    sem = semaphores.get(gpu_id)
    if sem is None:
        sem = threading.Semaphore(1)
        semaphores[gpu_id] = sem

    from app.services.pipeline_profile import PROFILES, use_profile

    active_profile = PROFILES.get(job.profile_name) if job.profile_name else None

    def _call() -> tuple[str, float | None, dict]:
        # Re-acquire the semaphore on each retry: any held GPU memory is
        # released between attempts (caller logs OOM + empty_cache).
        with sem, use_profile(active_profile):
            return ocr_image(
                job.path,
                lang=job.lang,
                allow_vision_escalation=job.allow_vision_escalation,
                gpu_id=gpu_id,
                line_level=job.line_level,
            )

    # ── GPU scheduler: route through primary queue when enabled ───────────
    # This ensures primary OCR work runs on the primary GPU (GPU0) and its
    # thread pool is managed independently from escalation (GPU1/VLM).
    # We still hold the semaphore inside the submitted callable so GPU
    # concurrency limits are respected.
    if settings.gpu_scheduler_enabled:
        try:
            from app.services.gpu_scheduler import get_scheduler
            from app.services.dynamic_gpu_scheduler import TaskPriority

            scheduler = get_scheduler()
            priority = (
                TaskPriority(job.priority)
                if job.priority is not None
                else TaskPriority.NORMAL
            )
            # Pin job to primary GPU (override round-robin allocation).
            primary_gpu = scheduler.primary.gpu_id
            if gpu_id != primary_gpu:
                # Keep user-specified pinning (page-parallel, member-pin, etc.)
                pass
            else:
                gpu_id = primary_gpu

            if settings.ocr_oom_retry:
                future = scheduler.submit_primary(
                    with_oom_retry,
                    _call,
                    gpu_id=gpu_id,
                    max_retries=settings.ocr_oom_retry_max,
                    label=f"ocr_job[{job.key}]",
                    priority=priority,
                )
            else:
                future = scheduler.submit_primary(_call, priority=priority)
            text, confidence, meta = future.result()
            meta = {**meta, "gpu_id": gpu_id, "scheduled": True, "priority": int(priority)}
            return OcrJobResult(key=job.key, text=text, confidence=confidence, meta=meta)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GpuScheduler primary submit failed for job %s, falling back: %s",
                job.key, exc,
            )
            # Fall through to direct execution below

    # ── direct execution (scheduler disabled or fallback) ─────────────────
    if settings.ocr_oom_retry:
        text, confidence, meta = with_oom_retry(
            _call,
            gpu_id=gpu_id,
            max_retries=settings.ocr_oom_retry_max,
            label=f"ocr_job[{job.key}]",
        )
    else:
        text, confidence, meta = _call()
    meta = {**meta, "gpu_id": gpu_id}
    return OcrJobResult(key=job.key, text=text, confidence=confidence, meta=meta)


def _run_sequential(
    jobs: list[OcrJob],
    gpu_ids: list[int],
) -> dict[str, OcrJobResult]:
    primary = gpu_ids[0]
    pinned = [
        replace(j, gpu_id=j.gpu_id if j.gpu_id is not None else primary)
        for j in jobs
    ]
    semaphores = ocr_gpu_semaphores(gpu_ids)
    return {j.key: _run_one(j, semaphores, gpu_ids) for j in pinned}


def _run_region_parallel(
    jobs: list[OcrJob],
    gpu_ids: list[int],
) -> dict[str, OcrJobResult]:
    workers = parallel_worker_count(gpu_ids)
    logger.info(
        "Parallel OCR (regions): %d jobs on GPUs %s (%d workers)",
        len(jobs), gpu_ids, workers,
    )
    semaphores = ocr_gpu_semaphores(gpu_ids)
    results: dict[str, OcrJobResult] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one, job, semaphores, gpu_ids): job for job in jobs
        }
        for fut in as_completed(futures):
            res = fut.result()
            results[res.key] = res

    return results


def _run_page_parallel(
    jobs: list[OcrJob],
    gpu_ids: list[int],
) -> dict[str, OcrJobResult]:
    """Each page pinned to one GPU; pages run concurrently (shared semaphores)."""
    by_page: dict[int, list[OcrJob]] = {}
    orphan: list[OcrJob] = []
    for job in jobs:
        page_idx = page_idx_from_key(job.key)
        if page_idx is None:
            orphan.append(job)
        else:
            by_page.setdefault(page_idx, []).append(job)

    semaphores = ocr_gpu_semaphores(gpu_ids)
    results: dict[str, OcrJobResult] = {}

    def _process_page(page_idx: int, page_jobs: list[OcrJob]) -> None:
        gid = gpu_for_page(page_idx, gpu_ids)
        for job in page_jobs:
            pinned = replace(job, gpu_id=gid)
            res = _run_one(pinned, semaphores, gpu_ids)
            results[res.key] = res

    page_workers = min(len(by_page), len(gpu_ids), parallel_worker_count(gpu_ids))
    logger.info(
        "Parallel OCR (pages): %d jobs, %d pages, GPUs %s (%d page workers)",
        len(jobs), len(by_page), gpu_ids, page_workers,
    )

    if by_page:
        with ThreadPoolExecutor(max_workers=max(1, page_workers)) as pool:
            futures = [
                pool.submit(_process_page, page_idx, page_jobs)
                for page_idx, page_jobs in sorted(by_page.items())
            ]
            for fut in futures:
                fut.result()

    for job in orphan:
        res = _run_one(job, semaphores, gpu_ids)
        results[res.key] = res

    return results


def run_ocr_jobs(
    jobs: list[OcrJob],
    *,
    parallel: bool = False,
    page_parallel: bool = False,
    gpu_ids: list[int] | None = None,
) -> dict[str, OcrJobResult]:
    """Execute OCR jobs with optional multi-GPU scheduling."""
    if not jobs:
        return {}

    if gpu_ids is None:
        gpu_ids = [get_primary_ocr_gpu_id()]

    if page_parallel and len(gpu_ids) >= 2:
        return _run_page_parallel(jobs, gpu_ids)

    if not parallel or len(gpu_ids) < 2:
        return _run_sequential(jobs, gpu_ids)

    return _run_region_parallel(jobs, gpu_ids)
