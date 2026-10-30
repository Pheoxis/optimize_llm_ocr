"""ZIP checkpoint: atomic writes, resume skip, crash-safe metadata."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from PIL import Image

from app.core.config import settings
from app.core.constants import JobStatus, SourceKind
from app.models.schemas import ProcessingResult
from app.services import pipeline
from app.services.jobs import job_store
from app.services.zip_checkpoint import (
    ZipCheckpointStore,
    atomic_write_json,
    checkpoint_path,
    load_checkpoint,
    member_has_results,
)
from app.services.zip_queue import ZipMember
from app.services.pipeline import resume_zip_checkpoint


def _pdf_member(name: str) -> ZipMember:
    return ZipMember(Path(name), name, SourceKind.PDF, 10)


def test_atomic_write_json_creates_valid_file(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.json"
    atomic_write_json(target, {"version": 1, "ok": True})
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8"))["ok"] is True
    assert not list(tmp_path.glob(".checkpoint.json.*.tmp"))


def test_checkpoint_record_and_flush(tmp_path: Path) -> None:
    job_id = uuid4()
    result_dir = tmp_path / str(job_id)
    store = ZipCheckpointStore(
        job_id, result_dir, total_files=2, source_archive="t.zip",
    )
    member = _pdf_member("a.pdf")
    result_dir.mkdir(parents=True)
    out = result_dir / "a.pdf"
    out.mkdir()
    (out / "result.json").write_text("{}", encoding="utf-8")
    (out / "result.txt").write_text("x", encoding="utf-8")

    store.record_completed(
        member,
        json_path=out / "result.json",
        txt_path=out / "result.txt",
        retry_count=0,
    )

    loaded = load_checkpoint(result_dir)
    assert loaded is not None
    assert loaded["completed_members"] == ["a.pdf"]
    assert loaded["completed"][0]["processed_at"]
    assert loaded["completed"][0]["retry_count"] == 0


def test_should_skip_only_with_results(tmp_path: Path) -> None:
    job_id = uuid4()
    result_dir = tmp_path / str(job_id)
    store = ZipCheckpointStore(job_id, result_dir, total_files=1)
    member = _pdf_member("a.pdf")

    store._data["completed_members"] = ["a.pdf"]
    assert not store.should_skip_member(member)

    out = result_dir / "a.pdf"
    out.mkdir(parents=True)
    (out / "result.json").write_text("{}", encoding="utf-8")
    (out / "result.txt").write_text("x", encoding="utf-8")
    assert member_has_results(result_dir, "a.pdf")
    assert store.should_skip_member(member)


def test_zip_job_writes_checkpoint_per_member(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "storage" / "uploads")
    monkeypatch.setattr(settings, "results_dir", tmp_path / "storage" / "results")
    monkeypatch.setattr(settings, "temp_dir", tmp_path / "storage" / "temp")
    monkeypatch.setattr(settings, "project_root", tmp_path)
    monkeypatch.setattr(settings, "zip_parallel_workers", 1)
    settings.ensure_dirs()

    img_dir = tmp_path / "src"
    img_dir.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (4, 4), color=(255, 255, 255)).save(img_dir / name)
    zip_path = tmp_path / "two.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in img_dir.iterdir():
            zf.write(p, arcname=p.name)

    job = job_store.create()
    job_id = job.job_id
    seen: list[str] = []

    def fake_process_single_image(
        _job_id, input_path, *, source_name, update_job_progress,
        allow_vision_escalation=None, gpu_id=None,
    ):
        from datetime import UTC, datetime

        seen.append(Path(source_name).name)
        return ProcessingResult(
            source=source_name,
            source_kind=SourceKind.IMAGE,
            processed_at=datetime.now(UTC),
            page_count=1,
            language_detected=["pol"],
            text="ok",
            confidence=0.9,
            engine="fake",
            warnings=[],
            metadata={},
        )

    with (
        patch.object(pipeline, "_process_single_image", side_effect=fake_process_single_image),
        patch.object(pipeline, "get_bulk_ocr_gpu_ids", return_value=[0]),
        patch.object(pipeline, "allocate_gpu", return_value=0),
    ):
        pipeline.run_job(job_id, zip_path, source_name="two.zip")

    ckpt_path = checkpoint_path(settings.results_dir / str(job_id))
    assert ckpt_path.is_file()
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert set(ckpt["completed_members"]) == {"a.png", "b.png"}
    assert ckpt["status"] == "finished"
    assert len(seen) == 2


def test_resume_skips_checkpointed_members(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "storage" / "uploads")
    monkeypatch.setattr(settings, "results_dir", tmp_path / "storage" / "results")
    monkeypatch.setattr(settings, "temp_dir", tmp_path / "storage" / "temp")
    monkeypatch.setattr(settings, "project_root", tmp_path)
    monkeypatch.setattr(settings, "zip_parallel_workers", 1)
    settings.ensure_dirs()

    img_dir = tmp_path / "src"
    img_dir.mkdir()
    for name in ("done.png", "pending.png"):
        Image.new("RGB", (4, 4), color=(255, 255, 255)).save(img_dir / name)
    zip_path = tmp_path / "pair.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in img_dir.iterdir():
            zf.write(p, arcname=p.name)

    job = job_store.create()
    job_id = job.job_id
    import shutil

    shutil.copy(zip_path, settings.uploads_dir / f"{job_id}_pair.zip")
    result_dir = settings.results_dir / str(job_id)
    result_dir.mkdir(parents=True)
    done_dir = result_dir / "done.png"
    done_dir.mkdir()
    (done_dir / "result.json").write_text("{}", encoding="utf-8")
    (done_dir / "result.txt").write_text("done", encoding="utf-8")

    store = ZipCheckpointStore(job_id, result_dir, total_files=2)
    store.record_completed(
        ZipMember(done_dir / "x", "done.png", SourceKind.IMAGE, 1),
        json_path=done_dir / "result.json",
        txt_path=done_dir / "result.txt",
    )
    store._data["status"] = "in_progress"
    atomic_write_json(checkpoint_path(result_dir), store._data)

    processed: list[str] = []

    def fake_process_single_image(
        _job_id, input_path, *, source_name, update_job_progress,
        allow_vision_escalation=None, gpu_id=None,
    ):
        from datetime import UTC, datetime

        processed.append(Path(source_name).name)
        return ProcessingResult(
            source=source_name,
            source_kind=SourceKind.IMAGE,
            processed_at=datetime.now(UTC),
            page_count=1,
            language_detected=["pol"],
            text="ok",
            confidence=0.9,
            engine="fake",
            warnings=[],
            metadata={},
        )

    with (
        patch.object(pipeline, "_process_single_image", side_effect=fake_process_single_image),
        patch.object(pipeline, "get_bulk_ocr_gpu_ids", return_value=[0]),
        patch.object(pipeline, "allocate_gpu", return_value=0),
    ):
        resume_zip_checkpoint(job_id)

    assert processed == ["pending.png"]
    refreshed = job_store.get(job_id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.DONE
