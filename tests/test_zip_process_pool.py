"""ZIP process-pool scheduling (no CUDA required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.core.config import settings
from app.core.constants import SourceKind
from app.services.zip_queue import ZipMember, ZipProcessingQueue
from app.services.zip_worker import ZipWorkItem, ZipWorkResult


@pytest.fixture(autouse=True)
def _restore_settings() -> None:
    saved = settings.zip_parallel_workers
    yield
    settings.zip_parallel_workers = saved


def _pdf(name: str) -> ZipMember:
    return ZipMember(Path(name), name, SourceKind.PDF, 100)


def test_run_rejects_parallel_without_process_pool() -> None:
    queue = ZipProcessingQueue([_pdf("a.pdf"), _pdf("b.pdf")], max_workers=2)
    with pytest.raises(RuntimeError, match="run_in_process_pool"):
        queue.run(lambda m: m)


def test_run_in_process_pool_invokes_worker_and_on_complete() -> None:
    members = [_pdf("a.pdf"), _pdf("b.pdf")]
    queue = ZipProcessingQueue(members, max_workers=2)
    job_id = uuid4()
    work_dir = Path("/tmp/work")
    result_dir = Path("/tmp/results")
    completed: list[str] = []

    def fake_worker(item: ZipWorkItem) -> ZipWorkResult:
        return ZipWorkResult(
            ok=True,
            json_path=f"{result_dir}/{item.rel_path}/result.json",
            txt_path=f"{result_dir}/{item.rel_path}/result.txt",
        )

    mock_pool_cls = MagicMock()
    mock_executor = MagicMock()
    mock_pool_cls.return_value = mock_executor
    mock_executor.shutdown = MagicMock()

    futures: list[tuple[MagicMock, ZipWorkItem]] = []

    def submit(fn, item):
        fut = MagicMock()
        fut.result.return_value = fake_worker(item)
        futures.append((fut, item))
        return fut

    mock_executor.submit.side_effect = submit

    def fake_as_completed(fs):
        for fut, _item in futures:
            yield fut

    with (
        patch(
            "app.services.zip_queue.ProcessPoolExecutor",
            mock_pool_cls,
        ),
        patch("app.services.zip_queue.as_completed", fake_as_completed),
        patch(
            "app.services.zip_queue.multiprocessing.get_context",
        ) as mock_ctx,
    ):
        mock_manager = MagicMock()
        mock_ctx.return_value = _spawn_context_mock(mock_manager)

        queue.run_in_process_pool(
            job_id=job_id,
            archive_path=Path("/tmp/archive.zip"),
            extract_root=work_dir / "extracted",
            work_dir=work_dir,
            result_dir=result_dir,
            pinned_for=lambda _m: 0,
            allow_vision_escalation=None,
            gpu_ids=[0],
            on_complete=lambda m, r: completed.append(m.rel_path),
        )

    assert sorted(completed) == ["a.pdf", "b.pdf"]
    mock_pool_cls.assert_called_once()
    call_kw = mock_pool_cls.call_args.kwargs
    assert call_kw["max_workers"] == 2
    assert call_kw["mp_context"] is mock_ctx.return_value
    mock_executor.shutdown.assert_called()


def _spawn_context_mock(mock_manager: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.Manager.return_value = mock_manager
    lock = MagicMock()
    mock_manager.Lock.return_value = lock
    return ctx


def test_run_in_process_pool_member_timeout_records_failure() -> None:
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    members = [_pdf("slow.pdf")]
    queue = ZipProcessingQueue(members, max_workers=1)
    outcomes: list[ZipWorkResult] = []

    mock_pool_cls = MagicMock()
    mock_executor = MagicMock()
    mock_pool_cls.return_value = mock_executor
    mock_executor.shutdown = MagicMock()

    fut = MagicMock()
    fut.result.side_effect = FuturesTimeoutError()
    mock_executor.submit.return_value = fut

    with (
        patch("app.services.zip_queue.ProcessPoolExecutor", mock_pool_cls),
        patch("app.services.zip_queue.as_completed", lambda fs: fs),
        patch(
            "app.services.zip_queue.multiprocessing.get_context",
        ) as mock_ctx,
    ):
        mock_ctx.return_value = _spawn_context_mock(MagicMock())
        settings.zip_member_timeout_sec = 1.0
        queue.run_in_process_pool(
            job_id=uuid4(),
            archive_path=Path("/tmp/archive.zip"),
            extract_root=Path("/tmp/work/extracted"),
            work_dir=Path("/tmp/work"),
            result_dir=Path("/tmp/results"),
            pinned_for=lambda _m: None,
            allow_vision_escalation=None,
            gpu_ids=[],
            on_complete=lambda _m, r: outcomes.append(r),
        )

    assert len(outcomes) == 1
    assert not outcomes[0].ok
    assert outcomes[0].error_type == "TimeoutError"
    fut.cancel.assert_called_once()
