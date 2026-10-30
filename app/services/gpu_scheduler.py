"""
GPU-specialised OCR scheduler.

Problem
-------
When crop-level VLM escalation (MiniCPM-V on GPU1) and primary OCR
(RapidOCR on GPU0) share the same thread pool, VLM latency spikes can
starve incoming OCR requests and inflate ZIP throughput latency.

Solution
--------
Two dedicated queues with separate thread pools and GPU affinity:

  PrimaryQueue   → GPU0: RapidOCR, batch OCR, ZIP member throughput
  EscalationQueue → GPU1: MiniCPM-V / VLM crop calls, premium Tesseract

Primary OCR threads never block on escalation queue depth.
If the escalation queue is saturated and the wait exceeds the configured
timeout, the caller receives the RapidOCR baseline immediately.

Architecture
------------

  ┌─────────────────────────────────────────────────────┐
  │                    GpuScheduler                     │
  │                                                     │
  │  PrimaryQueue (GPU0)    EscalationQueue (GPU1)      │
  │  ┌──────────────┐       ┌──────────────────────┐   │
  │  │ ThreadPool   │       │ BoundedThreadPool    │   │
  │  │ workers=2    │       │ workers=1            │   │
  │  │ semaphore    │       │ depth cap=32         │   │
  │  └──────────────┘       └──────────────────────┘   │
  │      ↑  submit_primary()      ↑ submit_escalation() │
  │      │                        │ (with timeout+fb)   │
  └─────────────────────────────────────────────────────┘

Usage
-----
  from app.services.gpu_scheduler import get_scheduler

  scheduler = get_scheduler()

  # Primary OCR task (blocks until done, uses GPU0 slot):
  future = scheduler.submit_primary(rapid_ocr_fn, image)
  result = future.result()

  # VLM escalation task (timeout + fallback):
  text = scheduler.submit_escalation(
      vlm_fn, crop_image,
      timeout_ms=settings.gpu_scheduler_escalation_timeout_ms,
      fallback="",
  )

  # Metrics snapshot:
  stats = scheduler.metrics()
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── per-queue metrics ─────────────────────────────────────────────────────────


@dataclass
class QueueSnapshot:
    """Point-in-time metrics for one GPU queue."""

    name: str
    gpu_id: int
    workers: int
    queued: int          # tasks waiting for a free worker
    in_flight: int       # tasks currently executing
    completed: int       # total completed since start
    rejected: int        # tasks dropped due to depth cap / timeout
    mean_latency_ms: float   # rolling mean over last N completions
    p95_latency_ms: float
    max_latency_ms: float
    throughput_rps: float    # completions / second over recent window

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gpu_id": self.gpu_id,
            "workers": self.workers,
            "queued": self.queued,
            "in_flight": self.in_flight,
            "completed": self.completed,
            "rejected": self.rejected,
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "throughput_rps": round(self.throughput_rps, 3),
        }


class _MetricsTracker:
    """Thread-safe rolling statistics (last 256 samples)."""

    _WINDOW = 256

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=self._WINDOW)
        self._completed = 0
        self._rejected = 0
        self._in_flight = 0
        self._queued = 0
        # Throughput window: (timestamp, count) pairs in the last 60 s
        self._ts_window: deque[tuple[float, int]] = deque(maxlen=3600)

    def task_queued(self) -> None:
        with self._lock:
            self._queued += 1

    def task_dequeued(self) -> None:
        with self._lock:
            self._queued = max(0, self._queued - 1)
            self._in_flight += 1

    def task_done(self, elapsed_ms: float) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._completed += 1
            self._latencies.append(elapsed_ms)
            self._ts_window.append((time.monotonic(), 1))

    def task_rejected(self) -> None:
        with self._lock:
            self._queued = max(0, self._queued - 1)
            self._rejected += 1

    def snapshot(self, name: str, gpu_id: int, workers: int) -> QueueSnapshot:
        with self._lock:
            lats = list(self._latencies)
            completed = self._completed
            rejected = self._rejected
            in_flight = self._in_flight
            queued = self._queued

            if lats:
                sl = sorted(lats)
                mean_lat = sum(sl) / len(sl)
                p95_lat = sl[max(0, int(len(sl) * 0.95))]
                max_lat = sl[-1]
            else:
                mean_lat = p95_lat = max_lat = 0.0

            # Throughput: count events in the last 60 s
            now = time.monotonic()
            recent = [c for ts, c in self._ts_window if now - ts <= 60.0]
            throughput = sum(recent) / 60.0 if recent else 0.0

        return QueueSnapshot(
            name=name,
            gpu_id=gpu_id,
            workers=workers,
            queued=queued,
            in_flight=in_flight,
            completed=completed,
            rejected=rejected,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            max_latency_ms=max_lat,
            throughput_rps=throughput,
        )


# ── GPU queue ─────────────────────────────────────────────────────────────────


class GpuQueue:
    """
    Thread-pool-backed task queue pinned to a specific GPU.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "primary", "escalation").
    gpu_id : int
        CUDA device index set via ``CUDA_VISIBLE_DEVICES`` on the worker thread.
    workers : int
        Maximum concurrent tasks.
    max_queued : int
        Maximum number of tasks waiting in the queue (0 = unlimited).
        When exceeded, ``submit`` raises ``QueueFullError``.
    """

    def __init__(
        self,
        name: str,
        gpu_id: int,
        workers: int,
        max_queued: int = 0,
    ) -> None:
        self.name = name
        self.gpu_id = gpu_id
        self.workers = workers
        self.max_queued = max_queued
        self._metrics = _MetricsTracker()
        self._pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=f"gpu{gpu_id}_{name}",
        )
        self._shutdown = False

    # ── submit ─────────────────────────────────────────────────────────────

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        """
        Submit a task to this queue.

        Raises
        ------
        QueueFullError
            When ``max_queued`` is set and the queue is at capacity.
        RuntimeError
            When the queue has been shut down.
        """
        if self._shutdown:
            raise RuntimeError(f"GpuQueue[{self.name}] is shut down")

        if self.max_queued > 0:
            snapshot = self._metrics.snapshot(self.name, self.gpu_id, self.workers)
            if snapshot.queued >= self.max_queued:
                self._metrics.task_rejected()
                raise QueueFullError(
                    f"GpuQueue[{self.name}] depth {snapshot.queued} >= {self.max_queued}"
                )

        self._metrics.task_queued()

        def _wrapped() -> T:
            self._metrics.task_dequeued()
            t0 = time.perf_counter()
            # Pin worker thread to the correct CUDA device
            _set_cuda_device(self.gpu_id)
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._metrics.task_done(elapsed_ms)

        return self._pool.submit(_wrapped)

    def submit_with_timeout(
        self,
        fn: Callable[..., T],
        *args: Any,
        timeout_ms: float,
        fallback: T,
        **kwargs: Any,
    ) -> T:
        """
        Submit a task and wait up to ``timeout_ms`` milliseconds.

        Returns ``fallback`` if the queue is full, the task times out, or the
        task raises an exception.  Never blocks the caller longer than the timeout.
        """
        timeout_sec = timeout_ms / 1000.0 if timeout_ms > 0 else None
        try:
            future = self.submit(fn, *args, **kwargs)
        except (QueueFullError, RuntimeError) as exc:
            logger.debug("GpuQueue[%s] submit rejected: %s", self.name, exc)
            return fallback

        try:
            return future.result(timeout=timeout_sec)
        except TimeoutError:
            future.cancel()
            logger.debug(
                "GpuQueue[%s] task timed out after %.0f ms — using fallback",
                self.name, timeout_ms,
            )
            return fallback
        except Exception as exc:  # noqa: BLE001
            logger.warning("GpuQueue[%s] task raised: %s", self.name, exc)
            return fallback

    # ── introspection ──────────────────────────────────────────────────────

    def snapshot(self) -> QueueSnapshot:
        return self._metrics.snapshot(self.name, self.gpu_id, self.workers)

    def shutdown(self, wait: bool = True) -> None:
        self._shutdown = True
        self._pool.shutdown(wait=wait)

    def __repr__(self) -> str:
        s = self.snapshot()
        return (
            f"GpuQueue({self.name!r}, gpu={self.gpu_id}, "
            f"workers={self.workers}, queued={s.queued}, in_flight={s.in_flight})"
        )


class QueueFullError(RuntimeError):
    """Raised when a GpuQueue's depth cap is exceeded."""


# ── scheduler ─────────────────────────────────────────────────────────────────


class GpuScheduler:
    """
    Process-wide GPU scheduler.

    Manages a ``primary`` queue (GPU0 → RapidOCR / ZIP throughput) and an
    ``escalation`` queue (GPU1 → VLM crops / premium Tesseract) so that
    escalation latency spikes cannot block primary OCR throughput.

    The scheduler is a singleton; obtain the instance via ``get_scheduler()``.
    """

    def __init__(
        self,
        *,
        primary_gpu: int,
        escalation_gpu: int,
        primary_workers: int,
        escalation_workers: int,
        escalation_max_queued: int,
        metrics_interval_sec: float,
        start_metrics_logger: bool = True,
    ) -> None:
        self.primary = GpuQueue(
            name="primary",
            gpu_id=primary_gpu,
            workers=primary_workers,
            max_queued=0,  # primary never drops
        )
        self.escalation = GpuQueue(
            name="escalation",
            gpu_id=escalation_gpu,
            workers=escalation_workers,
            max_queued=escalation_max_queued,
        )
        self._metrics_interval = metrics_interval_sec
        self._metrics_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        if start_metrics_logger and metrics_interval_sec > 0:
            self._start_metrics_logger()

    # ── public API ──────────────────────────────────────────────────────────

    def submit_primary(
        self,
        fn: Callable[..., T],
        *args: Any,
        priority: Any = None,  # ignored — used by DynamicGpuScheduler
        **kwargs: Any,
    ) -> Future[T]:
        """Submit a primary OCR task (non-blocking, returns Future)."""
        return self.primary.submit(fn, *args, **kwargs)

    def submit_escalation(
        self,
        fn: Callable[..., T],
        *args: Any,
        timeout_ms: float | None = None,
        fallback: T = None,  # type: ignore[assignment]
        priority: Any = None,  # ignored — used by DynamicGpuScheduler
        **kwargs: Any,
    ) -> T:
        """
        Submit a VLM escalation task and wait for the result.

        Returns ``fallback`` if the queue is saturated or the timeout expires.
        Timeout defaults to ``settings.gpu_scheduler_escalation_timeout_ms``.
        """
        from app.core.config import settings

        ms = timeout_ms if timeout_ms is not None else settings.gpu_scheduler_escalation_timeout_ms
        return self.escalation.submit_with_timeout(
            fn, *args, timeout_ms=ms, fallback=fallback, **kwargs,
        )

    def metrics(self) -> dict[str, Any]:
        """Return a snapshot of both queues as a JSON-serialisable dict."""
        return {
            "primary": self.primary.snapshot().to_dict(),
            "escalation": self.escalation.snapshot().to_dict(),
        }

    def log_metrics(self) -> None:
        """Emit current queue metrics to the application log."""
        m = self.metrics()
        p, e = m["primary"], m["escalation"]
        logger.info(
            "GpuScheduler │ primary(gpu%d)  queued=%d in_flight=%d "
            "done=%d p95=%.0fms tput=%.2f/s  │  "
            "escalation(gpu%d) queued=%d in_flight=%d "
            "done=%d rejected=%d p95=%.0fms tput=%.2f/s",
            p["gpu_id"], p["queued"], p["in_flight"],
            p["completed"], p["p95_latency_ms"], p["throughput_rps"],
            e["gpu_id"], e["queued"], e["in_flight"],
            e["completed"], e["rejected"], e["p95_latency_ms"], e["throughput_rps"],
        )

    def shutdown(self, wait: bool = True) -> None:
        self._stop_event.set()
        if self._metrics_thread and self._metrics_thread.is_alive():
            self._metrics_thread.join(timeout=5.0)
        self.primary.shutdown(wait=wait)
        self.escalation.shutdown(wait=wait)

    # ── background metrics logger ───────────────────────────────────────────

    def _start_metrics_logger(self) -> None:
        def _loop() -> None:
            while not self._stop_event.wait(timeout=self._metrics_interval):
                try:
                    self.log_metrics()
                except Exception:  # noqa: BLE001
                    pass

        self._metrics_thread = threading.Thread(
            target=_loop,
            name="gpu_scheduler_metrics",
            daemon=True,
        )
        self._metrics_thread.start()


# ── CUDA device pinning ───────────────────────────────────────────────────────


def _set_cuda_device(gpu_id: int) -> None:
    """
    Pin the current thread's CUDA context to ``gpu_id``.

    Uses ``torch.cuda.set_device`` when torch is available; otherwise sets
    ``CUDA_VISIBLE_DEVICES`` for sub-processes spawned from this thread.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            return
    except Exception:  # noqa: BLE001
        pass

    # Fallback: set env var so any cuda call from this thread's descendants
    # sees only the correct device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# ── singleton ─────────────────────────────────────────────────────────────────


_scheduler: GpuScheduler | None = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> GpuScheduler:
    """
    Return the process-wide GpuScheduler singleton.

    Thread-safe; initialises from ``settings`` on first call.
    Raises ``RuntimeError`` if ``settings.gpu_scheduler_enabled`` is False
    (callers should check before calling).
    """
    global _scheduler

    if _scheduler is not None:
        return _scheduler

    with _scheduler_lock:
        if _scheduler is not None:
            return _scheduler

        from app.core.config import settings

        if not settings.gpu_scheduler_enabled:
            raise RuntimeError(
                "get_scheduler() called but gpu_scheduler_enabled=False"
            )

        use_dynamic = settings.dynamic_gpu_scheduler_enabled
        base = GpuScheduler(
            primary_gpu=settings.gpu_scheduler_primary_gpu,
            escalation_gpu=settings.gpu_scheduler_escalation_gpu,
            primary_workers=settings.gpu_scheduler_primary_workers,
            escalation_workers=settings.gpu_scheduler_escalation_workers,
            escalation_max_queued=settings.gpu_scheduler_escalation_queue_depth,
            metrics_interval_sec=settings.gpu_scheduler_metrics_interval_sec,
            start_metrics_logger=not use_dynamic,
        )
        if use_dynamic:
            from app.services.dynamic_gpu_scheduler import DynamicGpuScheduler

            _scheduler = DynamicGpuScheduler(base)
            _scheduler._start_dynamic_metrics_logger(
                settings.gpu_scheduler_metrics_interval_sec,
            )
        else:
            _scheduler = base
        logger.info(
            "GpuScheduler initialised: primary=gpu%d×%d  escalation=gpu%d×%d  "
            "esc_depth=%d  esc_timeout=%.0f ms  dynamic=%s",
            settings.gpu_scheduler_primary_gpu,
            settings.gpu_scheduler_primary_workers,
            settings.gpu_scheduler_escalation_gpu,
            settings.gpu_scheduler_escalation_workers,
            settings.gpu_scheduler_escalation_queue_depth,
            settings.gpu_scheduler_escalation_timeout_ms,
            use_dynamic,
        )
        return _scheduler


def reset_scheduler() -> None:
    """Tear down and reset the singleton (tests / hot reload)."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
            _scheduler = None
    try:
        from app.services.dynamic_gpu_scheduler import reset_dynamic_scheduler
        reset_dynamic_scheduler()
    except ImportError:
        pass


def scheduler_metrics() -> dict[str, Any]:
    """
    Return queue metrics without raising when the scheduler is disabled.

    Always safe to call — returns ``{"enabled": False}`` when the scheduler
    is not active.
    """
    from app.core.config import settings

    if not settings.gpu_scheduler_enabled:
        return {"enabled": False}

    try:
        sched = get_scheduler()
        return {"enabled": True, **sched.metrics()}
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "error": str(exc)}
