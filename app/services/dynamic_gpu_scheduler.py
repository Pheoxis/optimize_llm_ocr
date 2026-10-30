"""
Dynamic GPU scheduler — VRAM-aware, priority-based, throughput-optimised.

Extends the base :class:`GpuScheduler` with:

  GPU0 (primary)
    * RapidOCR, easy/medium documents
    * CRITICAL / HIGH priority — never blocked by escalation
    * Throughput-optimised worker allocation

  GPU1 (escalation)
    * MiniCPM-V crop escalation, hard documents, retries
    * Throttled when primary queue is saturated or VRAM is low
    * LOW priority — may wait longer

Features
--------
  * VRAM-aware scheduling       — defer escalation when free VRAM < threshold
  * Dynamic queue balancing     — boost primary workers when escalation idle
  * Escalation throttling       — pause VLM when primary OCR backlog grows
  * Adaptive worker allocation  — ZIP tier worker counts adjust to GPU load
  * Queue priorities            — CRITICAL > HIGH > NORMAL > LOW > ESCALATION

Metrics (beyond base scheduler)
-------------------------------
  * queue_latency_ms            — time from submit to worker start (per priority)
  * gpu_utilization_pct         — estimated from in_flight / workers
  * ocr_throughput_rps          — primary queue completions / second
  * escalation_queue_depth      — current escalation backlog
  * vram_free_mb / vram_used_pct — per-GPU memory snapshot
  * escalation_throttled_count  — VLM calls rejected due to throttling

Usage
-----
    from app.services.dynamic_gpu_scheduler import (
        TaskPriority,
        get_dynamic_scheduler,
    )

    sched = get_dynamic_scheduler()
    future = sched.submit_primary(fn, image, priority=TaskPriority.HIGH)
    result = sched.submit_escalation(vlm_fn, crop, priority=TaskPriority.ESCALATION)
    print(sched.metrics())
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, TypeVar

from app.services.gpu_scheduler import GpuScheduler, QueueFullError, QueueSnapshot

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── priority enum ─────────────────────────────────────────────────────────────


class TaskPriority(IntEnum):
    """
    Task priority for GPU scheduling.

    Lower value = higher priority (processed sooner, less throttling).
    """
    CRITICAL = 0     # ZIP FAST, single easy JPG — must complete immediately
    HIGH = 1         # easy/medium docs, throughput path
    NORMAL = 2       # standard OCR jobs
    LOW = 3          # hard docs, quality retries
    ESCALATION = 4   # MiniCPM / VLM crop calls (GPU1)


_PRIORITY_NAMES = {
    TaskPriority.CRITICAL: "critical",
    TaskPriority.HIGH: "high",
    TaskPriority.NORMAL: "normal",
    TaskPriority.LOW: "low",
    TaskPriority.ESCALATION: "escalation",
}


def priority_from_zip_tier(tier: str) -> TaskPriority:
    """Map ZIP adaptive tier string → TaskPriority."""
    mapping = {
        "fast": TaskPriority.CRITICAL,
        "medium": TaskPriority.HIGH,
        "escalation": TaskPriority.LOW,
    }
    return mapping.get((tier or "").casefold(), TaskPriority.NORMAL)


def priority_from_difficulty(tier: str) -> TaskPriority:
    """Map OCR difficulty tier (easy/medium/hard) → TaskPriority."""
    mapping = {
        "easy": TaskPriority.HIGH,
        "medium": TaskPriority.NORMAL,
        "hard": TaskPriority.LOW,
    }
    return mapping.get((tier or "").casefold(), TaskPriority.NORMAL)


# ── VRAM monitor ──────────────────────────────────────────────────────────────


@dataclass
class VramSnapshot:
    gpu_id: int
    total_mb: float = 0.0
    free_mb: float = 0.0
    used_mb: float = 0.0
    used_pct: float = 0.0
    available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_id": self.gpu_id,
            "total_mb": round(self.total_mb, 1),
            "free_mb": round(self.free_mb, 1),
            "used_mb": round(self.used_mb, 1),
            "used_pct": round(self.used_pct, 1),
            "available": self.available,
        }


def probe_vram(gpu_id: int) -> VramSnapshot:
    """Return current VRAM usage for ``gpu_id`` (best-effort)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return VramSnapshot(gpu_id=gpu_id)
        if gpu_id >= torch.cuda.device_count():
            return VramSnapshot(gpu_id=gpu_id)
        free_b, total_b = torch.cuda.mem_get_info(gpu_id)
        total_mb = total_b / (1024 ** 2)
        free_mb = free_b / (1024 ** 2)
        used_mb = total_mb - free_mb
        used_pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0
        return VramSnapshot(
            gpu_id=gpu_id,
            total_mb=total_mb,
            free_mb=free_mb,
            used_mb=used_mb,
            used_pct=used_pct,
            available=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("VRAM probe failed for gpu%d: %s", gpu_id, exc)
        return VramSnapshot(gpu_id=gpu_id)


# ── latency tracker ───────────────────────────────────────────────────────────


class _LatencyTracker:
    """Rolling queue-wait latency per priority band."""

    _WINDOW = 128

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, deque[float]] = {
            name: deque(maxlen=self._WINDOW) for name in _PRIORITY_NAMES.values()
        }

    def record(self, priority: TaskPriority, wait_ms: float) -> None:
        name = _PRIORITY_NAMES.get(priority, "normal")
        with self._lock:
            self._samples[name].append(wait_ms)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        with self._lock:
            for name, samples in self._samples.items():
                if not samples:
                    out[name] = {"mean_ms": 0.0, "p95_ms": 0.0, "count": 0.0}
                    continue
                sl = sorted(samples)
                out[name] = {
                    "mean_ms": round(sum(sl) / len(sl), 2),
                    "p95_ms": round(sl[max(0, int(len(sl) * 0.95))], 2),
                    "count": float(len(sl)),
                }
        return out


# ── dynamic scheduler ───────────────────────────────────────────────────────


@dataclass
class DynamicSchedulerMetrics:
    """Enriched metrics snapshot."""

    primary: dict[str, Any]
    escalation: dict[str, Any]
    vram_primary: dict[str, Any]
    vram_escalation: dict[str, Any]
    queue_latency_ms: dict[str, dict[str, float]]
    gpu_utilization_pct: dict[str, float]
    ocr_throughput_rps: float
    escalation_queue_depth: int
    escalation_throttled: int
    dynamic_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "dynamic_enabled": self.dynamic_enabled,
            "primary": self.primary,
            "escalation": self.escalation,
            "vram": {
                "primary": self.vram_primary,
                "escalation": self.vram_escalation,
            },
            "queue_latency_ms": self.queue_latency_ms,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "ocr_throughput_rps": round(self.ocr_throughput_rps, 3),
            "escalation_queue_depth": self.escalation_queue_depth,
            "escalation_throttled": self.escalation_throttled,
        }


class DynamicGpuScheduler:
    """
    VRAM-aware wrapper around :class:`GpuScheduler`.

    Presents the same public API as ``GpuScheduler`` plus priority-aware
    ``submit_primary`` / ``submit_escalation`` and enriched metrics.
    """

    def __init__(self, base: GpuScheduler) -> None:
        self._base = base
        self._latency = _LatencyTracker()
        self._escalation_throttled = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._metrics_thread: threading.Thread | None = None
        self._metrics_interval = 0.0

    # ── passthrough properties ────────────────────────────────────────────────

    @property
    def primary(self):
        return self._base.primary

    @property
    def escalation(self):
        return self._base.escalation

    # ── VRAM / throttling ─────────────────────────────────────────────────────

    def _vram_ok_for_escalation(self) -> bool:
        from app.core.config import settings

        snap = probe_vram(self._base.escalation.gpu_id)
        if not snap.available:
            return True   # cannot probe — allow
        min_free = float(settings.dynamic_gpu_escalation_min_free_mb)
        max_used = float(settings.dynamic_gpu_escalation_max_used_pct)
        if snap.free_mb < min_free:
            return False
        if snap.used_pct > max_used:
            return False
        return True

    def _should_throttle_escalation(self, priority: TaskPriority) -> bool:
        """True when escalation should be deferred to protect primary throughput."""
        from app.core.config import settings

        if not settings.dynamic_gpu_scheduler_enabled:
            return False

        # Hard-doc retries (LOW) bypass throttling — quality path
        if priority <= TaskPriority.LOW:
            return False

        primary_snap = self._base.primary.snapshot()
        max_queued = int(settings.dynamic_gpu_primary_busy_queue)
        max_inflight_ratio = float(settings.dynamic_gpu_primary_busy_inflight_ratio)

        if primary_snap.queued >= max_queued:
            return True
        if (
            primary_snap.workers > 0
            and primary_snap.in_flight >= primary_snap.workers * max_inflight_ratio
            and primary_snap.queued > 0
        ):
            return True
        if not self._vram_ok_for_escalation():
            return True
        return False

    def _wait_for_primary_vram(self, priority: TaskPriority) -> float:
        """
        Optionally brief-wait for primary GPU VRAM to free up.

        Returns wait time in milliseconds actually spent waiting.
        CRITICAL / HIGH tasks skip waiting entirely.
        """
        from app.core.config import settings

        if not settings.dynamic_gpu_scheduler_enabled:
            return 0.0

        if priority <= TaskPriority.HIGH:
            return 0.0

        min_free = float(settings.dynamic_gpu_primary_min_free_mb)
        max_wait_ms = float(settings.dynamic_gpu_primary_vram_wait_ms)
        if max_wait_ms <= 0:
            return 0.0

        t0 = time.perf_counter()
        deadline = t0 + max_wait_ms / 1000.0
        while time.monotonic() < deadline:
            snap = probe_vram(self._base.primary.gpu_id)
            if not snap.available or snap.free_mb >= min_free:
                break
            time.sleep(0.05)
        return (time.monotonic() - t0) * 1000.0

    # ── public submit API ─────────────────────────────────────────────────────

    def submit_primary(
        self,
        fn: Callable[..., T],
        *args: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs: Any,
    ):
        """Submit primary OCR task with priority-aware VRAM waiting."""
        t0 = time.perf_counter()
        wait_ms = self._wait_for_primary_vram(priority)
        self._latency.record(priority, wait_ms)
        queue_wait_ms = (time.perf_counter() - t0) * 1000.0
        self._latency.record(priority, queue_wait_ms)
        return self._base.submit_primary(fn, *args, **kwargs)

    def submit_escalation(
        self,
        fn: Callable[..., T],
        *args: Any,
        timeout_ms: float | None = None,
        fallback: T = None,  # type: ignore[assignment]
        priority: TaskPriority = TaskPriority.ESCALATION,
        **kwargs: Any,
    ) -> T:
        """Submit VLM escalation with throttling and VRAM checks."""
        if self._should_throttle_escalation(priority):
            with self._lock:
                self._escalation_throttled += 1
            logger.debug(
                "DynamicGpuScheduler: escalation throttled (primary busy / VRAM low)"
            )
            return fallback

        t0 = time.perf_counter()
        result = self._base.submit_escalation(
            fn, *args,
            timeout_ms=timeout_ms,
            fallback=fallback,
            **kwargs,
        )
        self._latency.record(TaskPriority.ESCALATION, (time.perf_counter() - t0) * 1000.0)
        return result

    # ── metrics ───────────────────────────────────────────────────────────────

    def _gpu_utilization(self, snap: QueueSnapshot) -> float:
        if snap.workers <= 0:
            return 0.0
        return min(100.0, (snap.in_flight / snap.workers) * 100.0)

    def metrics(self) -> dict[str, Any]:
        """Return enriched metrics dict (compatible with base scheduler)."""
        p_snap = self._base.primary.snapshot()
        e_snap = self._base.escalation.snapshot()
        vram_p = probe_vram(self._base.primary.gpu_id)
        vram_e = probe_vram(self._base.escalation.gpu_id)

        with self._lock:
            throttled = self._escalation_throttled

        enriched = DynamicSchedulerMetrics(
            primary=p_snap.to_dict(),
            escalation=e_snap.to_dict(),
            vram_primary=vram_p.to_dict(),
            vram_escalation=vram_e.to_dict(),
            queue_latency_ms=self._latency.summary(),
            gpu_utilization_pct={
                "primary": round(self._gpu_utilization(p_snap), 1),
                "escalation": round(self._gpu_utilization(e_snap), 1),
            },
            ocr_throughput_rps=p_snap.throughput_rps,
            escalation_queue_depth=e_snap.queued + e_snap.in_flight,
            escalation_throttled=throttled,
        )
        return enriched.to_dict()

    def log_metrics(self) -> None:
        m = self.metrics()
        p, e = m["primary"], m["escalation"]
        vp = m["vram"]["primary"]
        ve = m["vram"]["escalation"]
        util = m["gpu_utilization_pct"]
        logger.info(
            "DynamicGpuScheduler │ "
            "primary(gpu%d) q=%d fly=%d util=%.0f%% vram_free=%.0fMB tput=%.2f/s │ "
            "escalation(gpu%d) depth=%d throttled=%d util=%.0f%% vram_free=%.0fMB │ "
            "latency_critical=%.0fms latency_high=%.0fms",
            p["gpu_id"], p["queued"], p["in_flight"], util["primary"],
            vp.get("free_mb", 0), m["ocr_throughput_rps"],
            e["gpu_id"], m["escalation_queue_depth"], m["escalation_throttled"],
            util["escalation"], ve.get("free_mb", 0),
            m["queue_latency_ms"].get("critical", {}).get("mean_ms", 0),
            m["queue_latency_ms"].get("high", {}).get("mean_ms", 0),
        )

    def shutdown(self, wait: bool = True) -> None:
        self._stop_event.set()
        if self._metrics_thread and self._metrics_thread.is_alive():
            self._metrics_thread.join(timeout=5.0)
        self._base.shutdown(wait=wait)

    def _start_dynamic_metrics_logger(self, interval_sec: float) -> None:
        """Start background metrics logging for the dynamic wrapper."""
        if interval_sec <= 0:
            return
        self._metrics_interval = interval_sec
        self._stop_event = threading.Event()
        self._metrics_thread: threading.Thread | None = None

        def _loop() -> None:
            while not self._stop_event.wait(timeout=self._metrics_interval):
                try:
                    self.log_metrics()
                except Exception:  # noqa: BLE001
                    pass

        self._metrics_thread = threading.Thread(
            target=_loop,
            name="dynamic_gpu_scheduler_metrics",
            daemon=True,
        )
        self._metrics_thread.start()

    # ── adaptive ZIP worker allocation ────────────────────────────────────────

    def adaptive_zip_workers(
        self,
        tier: str,
        *,
        default_workers: int,
        max_workers: int,
    ) -> int:
        """
        Dynamically adjust ZIP tier worker count based on GPU load.

        FAST tier gets extra workers when escalation queue is idle.
        ESCALATION tier workers reduced when primary is saturated.
        """
        from app.core.config import settings

        if not settings.dynamic_gpu_scheduler_enabled:
            return default_workers

        p = self._base.primary.snapshot()
        e = self._base.escalation.snapshot()
        tier_l = tier.casefold()

        if tier_l == "fast":
            # Boost FAST pool when escalation is idle — small files finish faster
            if e.queued == 0 and e.in_flight == 0 and p.queued < 2:
                return min(max_workers, default_workers + 1)
            return default_workers

        if tier_l == "escalation":
            # Reduce hard-doc pool when primary OCR is backlogged
            if p.queued >= settings.dynamic_gpu_primary_busy_queue:
                return max(1, default_workers - 1)
            return default_workers

        # MEDIUM: unchanged unless primary completely idle
        if p.in_flight == 0 and p.queued == 0:
            return min(max_workers, default_workers + 1)
        return default_workers


# ── factory ───────────────────────────────────────────────────────────────────

def get_dynamic_scheduler() -> DynamicGpuScheduler:
    """Return the dynamic scheduler (same singleton as ``get_scheduler()`` when enabled)."""
    from app.services.gpu_scheduler import get_scheduler

    sched = get_scheduler()
    if isinstance(sched, DynamicGpuScheduler):
        return sched
    return DynamicGpuScheduler(sched)


def reset_dynamic_scheduler() -> None:
    """Reset dynamic scheduler state (tests — base reset via ``reset_scheduler``)."""
    pass


def scheduler_metrics_enriched() -> dict[str, Any]:
    """Safe metrics accessor — works when scheduler disabled."""
    from app.services.gpu_scheduler import scheduler_metrics

    return scheduler_metrics()
