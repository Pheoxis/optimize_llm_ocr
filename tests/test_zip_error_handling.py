"""ZIP job: one failing member must not kill the whole job (per-file errors)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from PIL import Image

from app.core.config import settings
from app.core.constants import JobStatus
from app.models.schemas import ProcessingResult
from app.services import pipeline
from app.services.jobs import job_store


@pytest.fixture
def tiny_zip(tmp_path: Path) -> Path:
    """ZIP with two 1x1 PNGs (kinds → images, no OCR engine needed)."""
    img_dir = tmp_path / "src"
    img_dir.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (4, 4), color=(255, 255, 255)).save(img_dir / name)

    zip_path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in img_dir.iterdir():
            zf.write(p, arcname=p.name)
    return zip_path


def _fake_result(name: str) -> ProcessingResult:
    from datetime import UTC, datetime

    from app.core.constants import SourceKind

    return ProcessingResult(
        source=name,
        source_kind=SourceKind.IMAGE,
        processed_at=datetime.now(UTC),
        page_count=1,
        language_detected=["pol"],
        text=f"OCR for {name}",
        confidence=0.99,
        engine="fake",
        warnings=[],
        metadata={},
    )


def test_zip_partial_failure_keeps_other_members(
    tiny_zip: Path, tmp_path: Path, monkeypatch
) -> None:
    """One member raises → job is DONE with errors.json + partial results."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "storage" / "uploads")
    monkeypatch.setattr(settings, "results_dir", tmp_path / "storage" / "results")
    monkeypatch.setattr(settings, "temp_dir", tmp_path / "storage" / "temp")
    monkeypatch.setattr(settings, "project_root", tmp_path)
    monkeypatch.setattr(settings, "zip_parallel_workers", 1)
    settings.ensure_dirs()

    job = job_store.create()
    job_id = job.job_id

    call_count = {"n": 0}

    def fake_process_single_image(
        _job_id, input_path, *, source_name, update_job_progress,
        allow_vision_escalation=None, gpu_id=None,
    ):
        call_count["n"] += 1
        if Path(source_name).name == "b.png":
            raise RuntimeError("simulated OCR failure")
        return _fake_result(source_name)

    with (
        patch.object(pipeline, "_process_single_image", side_effect=fake_process_single_image),
        patch.object(pipeline, "get_bulk_ocr_gpu_ids", return_value=[0]),
        patch.object(pipeline, "allocate_gpu", return_value=0),
    ):
        pipeline.run_job(job_id, tiny_zip, source_name="tiny.zip")

    refreshed = job_store.get(job_id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.DONE
    assert refreshed.error is not None
    assert "1/2" in refreshed.error
    assert call_count["n"] == 2

    errors_path = settings.results_dir / str(job_id) / "errors.json"
    assert errors_path.is_file()
    payload = json.loads(errors_path.read_text(encoding="utf-8"))
    assert payload["total_files"] == 2
    assert payload["succeeded"] == 1
    assert payload["failed"] == 1
    assert payload["failures"][0]["member"].endswith("b.png")
    assert "simulated" in payload["failures"][0]["error"]

    a_result = settings.results_dir / str(job_id) / "a.png" / "result.json"
    assert a_result.is_file()


def test_zip_all_failed_marks_job_failed(
    tiny_zip: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "storage" / "uploads")
    monkeypatch.setattr(settings, "results_dir", tmp_path / "storage" / "results")
    monkeypatch.setattr(settings, "temp_dir", tmp_path / "storage" / "temp")
    monkeypatch.setattr(settings, "project_root", tmp_path)
    monkeypatch.setattr(settings, "zip_parallel_workers", 1)
    settings.ensure_dirs()

    job = job_store.create()
    job_id = job.job_id

    def boom(*_args, **_kwargs):
        raise RuntimeError("everything broke")

    with (
        patch.object(pipeline, "_process_single_image", side_effect=boom),
        patch.object(pipeline, "get_bulk_ocr_gpu_ids", return_value=[0]),
        patch.object(pipeline, "allocate_gpu", return_value=0),
    ):
        pipeline.run_job(job_id, tiny_zip, source_name="tiny.zip")

    refreshed = job_store.get(job_id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.FAILED
    assert "All 2" in (refreshed.error or "")
