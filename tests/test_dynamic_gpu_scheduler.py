"""Dynamic GPU scheduler — priority, throttling, adaptive workers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.dynamic_gpu_scheduler import (
    DynamicGpuScheduler,
    TaskPriority,
    priority_from_difficulty,
    priority_from_zip_tier,
    probe_vram,
)
from app.services.gpu_scheduler import GpuScheduler, reset_scheduler


@pytest.fixture(autouse=True)
def _reset_schedulers() -> None:
    reset_scheduler()
    yield
    reset_scheduler()


def _make_base(**overrides) -> GpuScheduler:
    defaults = dict(
        primary_gpu=0,
        escalation_gpu=1,
        primary_workers=2,
        escalation_workers=1,
        escalation_max_queued=8,
        metrics_interval_sec=0,
        start_metrics_logger=False,
    )
    defaults.update(overrides)
    return GpuScheduler(**defaults)


def test_priority_mapping() -> None:
    assert priority_from_zip_tier("fast") == TaskPriority.CRITICAL
    assert priority_from_zip_tier("medium") == TaskPriority.HIGH
    assert priority_from_zip_tier("escalation") == TaskPriority.LOW
    assert priority_from_difficulty("easy") == TaskPriority.HIGH
    assert priority_from_difficulty("hard") == TaskPriority.LOW


def test_probe_vram_no_cuda() -> None:
    snap = probe_vram(0)
    assert snap.gpu_id == 0
    # CI may or may not have CUDA — just ensure structure is valid
    assert snap.total_mb >= 0
    assert snap.free_mb >= 0


def test_submit_primary_delegates_to_base() -> None:
    base = _make_base()
    dyn = DynamicGpuScheduler(base)
    called: list[str] = []

    def _fn(x: int) -> int:
        called.append("run")
        return x * 2

    with patch.object(base, "submit_primary", wraps=base.submit_primary) as mock_submit:
        fut = dyn.submit_primary(_fn, 3, priority=TaskPriority.HIGH)
        assert fut.result() == 6
        mock_submit.assert_called_once()


def test_escalation_throttled_when_primary_busy() -> None:
    base = _make_base()
    dyn = DynamicGpuScheduler(base)

    busy_snap = MagicMock()
    busy_snap.queued = 10
    busy_snap.in_flight = 2
    busy_snap.workers = 2

    with (
        patch("app.core.config.settings") as mock_settings,
        patch.object(base.primary, "snapshot", return_value=busy_snap),
        patch.object(base, "submit_escalation") as mock_esc,
    ):
        mock_settings.dynamic_gpu_scheduler_enabled = True
        mock_settings.dynamic_gpu_primary_busy_queue = 4
        mock_settings.dynamic_gpu_primary_busy_inflight_ratio = 0.75

        result = dyn.submit_escalation(
            lambda: "vlm",
            fallback="fallback",
            priority=TaskPriority.ESCALATION,
        )
        assert result == "fallback"
        mock_esc.assert_not_called()
        metrics = dyn.metrics()
        assert metrics["escalation_throttled"] >= 1


def test_escalation_not_throttled_for_hard_docs() -> None:
    base = _make_base()
    dyn = DynamicGpuScheduler(base)

    busy_snap = MagicMock()
    busy_snap.queued = 10
    busy_snap.in_flight = 2
    busy_snap.workers = 2

    with (
        patch("app.core.config.settings") as mock_settings,
        patch.object(base.primary, "snapshot", return_value=busy_snap),
        patch.object(base, "submit_escalation", return_value="vlm_ok") as mock_esc,
    ):
        mock_settings.dynamic_gpu_scheduler_enabled = True
        mock_settings.dynamic_gpu_primary_busy_queue = 4
        mock_settings.dynamic_gpu_primary_busy_inflight_ratio = 0.75

        result = dyn.submit_escalation(
            lambda: "vlm",
            fallback="fallback",
            priority=TaskPriority.LOW,
        )
        assert result == "vlm_ok"
        mock_esc.assert_called_once()


def test_adaptive_zip_workers_boost_fast_when_idle() -> None:
    base = _make_base()
    dyn = DynamicGpuScheduler(base)

    idle_primary = MagicMock(queued=0, in_flight=0)
    idle_esc = MagicMock(queued=0, in_flight=0)

    with (
        patch("app.core.config.settings") as mock_settings,
        patch.object(base.primary, "snapshot", return_value=idle_primary),
        patch.object(base.escalation, "snapshot", return_value=idle_esc),
    ):
        mock_settings.dynamic_gpu_scheduler_enabled = True
        mock_settings.dynamic_gpu_primary_busy_queue = 4

        workers = dyn.adaptive_zip_workers(
            "fast", default_workers=2, max_workers=4,
        )
        assert workers == 3


def test_metrics_include_enriched_fields() -> None:
    base = _make_base()
    dyn = DynamicGpuScheduler(base)
    m = dyn.metrics()
    assert "queue_latency_ms" in m
    assert "gpu_utilization_pct" in m
    assert "ocr_throughput_rps" in m
    assert "escalation_queue_depth" in m
    assert "vram" in m
    assert "primary" in m["vram"]
    assert "escalation" in m["vram"]
