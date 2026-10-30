"""ZIP OCR worker recycle metrics and pool configuration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.core.config import settings
from app.core.constants import SourceKind
from app.services.zip_queue import (
    ZipMember,
    ZipProcessingQueue,
    _pool_max_tasks_per_child,
    summarize_worker_recycle_metrics,
)
from app.services.zip_worker import WorkerExitRecord


@pytest.fixture(autouse=True)
def _restore_settings() -> None:
    saved = (
        settings.zip_parallel_workers,
        settings.max_docs_per_worker,
    )
    yield
    settings.zip_parallel_workers, settings.max_docs_per_worker = saved


def test_pool_max_tasks_per_child_disabled_when_zero() -> None:
    settings.max_docs_per_worker = 0
    assert _pool_max_tasks_per_child() is None


def test_pool_max_tasks_per_child_uses_setting() -> None:
    settings.max_docs_per_worker = 200
    assert _pool_max_tasks_per_child() == 200


def test_summarize_worker_recycle_metrics() -> None:
    exits = [
        {
            "pid": 100,
            "docs_processed": 200,
            "lifetime_sec": 120.5,
            "exit_reason": "recycle",
        },
        {
            "pid": 101,
            "docs_processed": 50,
            "lifetime_sec": 30.0,
            "exit_reason": "pool_shutdown",
        },
    ]
    metrics = summarize_worker_recycle_metrics(
        exits,
        max_docs_per_worker=200,
        pool_workers=2,
        members_total=250,
    )
    assert metrics.worker_restart_count == 1
    assert metrics.worker_exit_count == 2
    assert metrics.total_docs_processed == 250
    assert metrics.avg_worker_lifetime_sec == pytest.approx(75.25)
    assert metrics.avg_docs_per_worker_exit == pytest.approx(125.0)


def test_process_pool_passes_max_tasks_per_child() -> None:
    from app.services.zip_worker import ZipWorkItem, ZipWorkResult

    settings.max_docs_per_worker = 200
    settings.zip_parallel_workers = 2
    members = [
        ZipMember(Path("a.pdf"), "a.pdf", SourceKind.PDF, 1),
    ]
    queue = ZipProcessingQueue(members, max_workers=2)
    mock_pool_cls = MagicMock()
    mock_executor = MagicMock()
    mock_pool_cls.return_value = mock_executor
    mock_executor.shutdown = MagicMock()
    fut = MagicMock()
    fut.result.return_value = ZipWorkResult(ok=True, json_path="/a.json", txt_path="/a.txt")
    mock_executor.submit.return_value = fut

    with (
        patch("app.services.zip_queue.ProcessPoolExecutor", mock_pool_cls),
        patch("app.services.zip_queue.as_completed", lambda fs: fs),
        patch("app.services.zip_queue.multiprocessing.get_context") as mock_ctx,
    ):
        manager = MagicMock()
        stats_list = MagicMock()
        stats_list.__iter__ = lambda self: iter([])
        manager.list.return_value = stats_list
        ctx = MagicMock()
        ctx.Manager.return_value = manager
        mock_ctx.return_value = ctx

        queue.run_in_process_pool(
            job_id=uuid4(),
            archive_path=Path("/tmp/archive.zip"),
            extract_root=Path("/tmp/work/extracted"),
            work_dir=Path("/tmp/work"),
            result_dir=Path("/tmp/results"),
            pinned_for=lambda _m: None,
            allow_vision_escalation=None,
            gpu_ids=[],
            on_complete=lambda _m, _r: None,
        )

    call_kw = mock_pool_cls.call_args.kwargs
    assert call_kw["max_tasks_per_child"] == 200
    assert call_kw["initargs"][2] == 200


def test_worker_exit_record_recycle_reason() -> None:
    record = WorkerExitRecord(
        pid=1,
        docs_processed=200,
        lifetime_sec=10.0,
        exit_reason="recycle",
    )
    assert record.exit_reason == "recycle"
