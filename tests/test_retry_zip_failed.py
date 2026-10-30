"""Retry only failed ZIP members from errors.json."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from PIL import Image

from app.core.config import settings
from app.core.constants import JobStatus
from app.models.schemas import ProcessingResult
from app.services import pipeline
from app.services.jobs import job_store
from app.services.pipeline import retry_zip_failed


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


def _make_three_png_zip(tmp_path: Path) -> Path:
    img_dir = tmp_path / "src"
    img_dir.mkdir()
    for name in ("ok.png", "fail1.png", "fail2.png"):
        Image.new("RGB", (4, 4), color=(255, 255, 255)).save(img_dir / name)
    zip_path = tmp_path / "three.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in sorted(img_dir.iterdir()):
            zf.write(p, arcname=p.name)
    return zip_path


def test_retry_zip_failed_only_missing_members(
    tmp_path: Path, monkeypatch,
) -> None:
    """Initial run: 1 ok, 2 failed → retry processes only the 2 failed."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "storage" / "uploads")
    monkeypatch.setattr(settings, "results_dir", tmp_path / "storage" / "results")
    monkeypatch.setattr(settings, "temp_dir", tmp_path / "storage" / "temp")
    monkeypatch.setattr(settings, "project_root", tmp_path)
    monkeypatch.setattr(settings, "zip_parallel_workers", 1)
    settings.ensure_dirs()

    zip_path = _make_three_png_zip(tmp_path)
    job = job_store.create()
    job_id = job.job_id
    shutil.copy(zip_path, settings.uploads_dir / f"{job_id}_three.zip")

    fail_names = {"fail1.png", "fail2.png"}
    phase = {"run": "initial"}

    def fake_process_single_image(
        _job_id, input_path, *, source_name, update_job_progress,
        allow_vision_escalation=None, gpu_id=None,
    ):
        name = Path(source_name).name
        if phase["run"] == "initial" and name in fail_names:
            raise RuntimeError(f"simulated failure for {name}")
        return _fake_result(source_name)

    with (
        patch.object(pipeline, "_process_single_image", side_effect=fake_process_single_image),
        patch.object(pipeline, "get_bulk_ocr_gpu_ids", return_value=[0]),
        patch.object(pipeline, "allocate_gpu", return_value=0),
    ):
        pipeline.run_job(job_id, zip_path, source_name="three.zip")

    errors_path = settings.results_dir / str(job_id) / "errors.json"
    assert errors_path.is_file()
    err = json.loads(errors_path.read_text(encoding="utf-8"))
    assert err["succeeded"] == 1
    assert err["failed"] == 2
    assert (settings.results_dir / str(job_id) / "ok.png" / "result.json").is_file()
    assert not (settings.results_dir / str(job_id) / "fail1.png" / "result.json").is_file()

    phase["run"] = "retry"
    retried_members: list[str] = []

    def track_retry(
        _job_id, input_path, *, source_name, update_job_progress,
        allow_vision_escalation=None, gpu_id=None,
    ):
        retried_members.append(Path(source_name).name)
        return _fake_result(source_name)

    with (
        patch.object(pipeline, "_process_single_image", side_effect=track_retry),
        patch.object(pipeline, "get_bulk_ocr_gpu_ids", return_value=[0]),
        patch.object(pipeline, "allocate_gpu", return_value=0),
    ):
        summary = retry_zip_failed(job_id)

    assert summary["retried"] == 2
    assert summary["succeeded"] == 2
    assert summary["still_failed"] == 0
    assert summary["members_ok"] == 3
    assert set(retried_members) == fail_names
    assert not errors_path.is_file()
    assert (settings.results_dir / str(job_id) / "fail1.png" / "result.json").is_file()
    assert (settings.results_dir / str(job_id) / "fail2.png" / "result.json").is_file()

    refreshed = job_store.get(job_id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.DONE
    assert refreshed.error is None
