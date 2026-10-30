"""ZIP missing-member detection and selective recovery."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from PIL import Image

from app.core.config import settings
from app.services.ingest import extract_zip_members, list_zip_documents
from app.services.zip_checkpoint import checkpoint_path, member_has_results
from app.services.zip_missing import (
    REPORT_FILENAME,
    analyze_zip_missing,
    process_zip_missing,
)
from app.services.pipeline import find_job_zip_upload


def _make_zip(tmp_path: Path, names: list[str]) -> Path:
    img_dir = tmp_path / "src"
    img_dir.mkdir()
    for name in names:
        Image.new("RGB", (4, 4), color=(255, 255, 255)).save(img_dir / name)
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in img_dir.iterdir():
            zf.write(p, arcname=p.name)
    return zip_path


def test_list_zip_documents_without_extract(tmp_path: Path) -> None:
    zip_path = _make_zip(tmp_path, ["a.png", "b.png"])
    entries = list_zip_documents(zip_path)
    assert {name for name, _ in entries} == {"a.png", "b.png"}


def test_analyze_zip_missing_detects_gap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "results_dir", tmp_path / "results")
    settings.uploads_dir.mkdir(parents=True)

    job_id = uuid4()
    zip_path = _make_zip(tmp_path, ["done.png", "missing.png"])
    (settings.uploads_dir / f"{job_id}_test.zip").write_bytes(zip_path.read_bytes())

    result_dir = settings.results_dir / str(job_id)
    done_dir = result_dir / "done.png"
    done_dir.mkdir(parents=True)
    (done_dir / "result.json").write_text("{}", encoding="utf-8")
    (done_dir / "result.txt").write_text("ok", encoding="utf-8")

    report = analyze_zip_missing(job_id)
    assert report.total_in_archive == 2
    assert report.complete_count == 1
    assert report.missing_count == 1
    assert report.missing_members == ["missing.png"]


def test_analyze_flags_checkpoint_stale(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "results_dir", tmp_path / "results")
    settings.uploads_dir.mkdir(parents=True)

    job_id = uuid4()
    zip_path = _make_zip(tmp_path, ["ghost.png"])
    (settings.uploads_dir / f"{job_id}_test.zip").write_bytes(zip_path.read_bytes())

    result_dir = settings.results_dir / str(job_id)
    result_dir.mkdir(parents=True)
    checkpoint_path(result_dir).write_text(
        json.dumps({
            "version": 1,
            "job_id": str(job_id),
            "completed_members": ["ghost.png"],
            "failed_members": [],
            "completed": [],
            "failed": [],
        }),
        encoding="utf-8",
    )

    report = analyze_zip_missing(job_id)
    assert "ghost.png" in report.checkpoint_stale
    assert "ghost.png" in report.missing_members


def test_selective_extract_only_missing(tmp_path: Path) -> None:
    zip_path = _make_zip(tmp_path, ["a.png", "b.png"])
    dest = tmp_path / "out"
    extracted = extract_zip_members(zip_path, dest, {"b.png"})
    assert len(extracted) == 1
    assert extracted[0].name == "b.png"
    assert not (dest / "a.png").exists()


def test_process_zip_missing_dry_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "results_dir", tmp_path / "results")
    monkeypatch.setattr(settings, "temp_dir", tmp_path / "temp")
    settings.uploads_dir.mkdir(parents=True)

    job_id = uuid4()
    zip_path = _make_zip(tmp_path, ["only.png"])
    (settings.uploads_dir / f"{job_id}_t.zip").write_bytes(zip_path.read_bytes())

    summary = process_zip_missing(job_id, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["missing_count"] == 1
    report_file = settings.results_dir / str(job_id) / REPORT_FILENAME
    assert report_file.is_file()


def test_process_zip_missing_runs_only_missing(
    tmp_path: Path, monkeypatch,
) -> None:
    from datetime import UTC, datetime

    from app.core.constants import SourceKind
    from app.models.schemas import ProcessingResult
    from app.services import pipeline

    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "results_dir", tmp_path / "results")
    monkeypatch.setattr(settings, "temp_dir", tmp_path / "temp")
    monkeypatch.setattr(settings, "project_root", tmp_path)
    monkeypatch.setattr(settings, "zip_parallel_workers", 1)
    settings.uploads_dir.mkdir(parents=True)

    job_id = uuid4()
    zip_path = _make_zip(tmp_path, ["done.png", "need.png"])
    (settings.uploads_dir / f"{job_id}_t.zip").write_bytes(zip_path.read_bytes())

    result_dir = settings.results_dir / str(job_id)
    done_dir = result_dir / "done.png"
    done_dir.mkdir(parents=True)
    (done_dir / "result.json").write_text("{}", encoding="utf-8")
    (done_dir / "result.txt").write_text("ok", encoding="utf-8")

    processed: list[str] = []

    def fake_process_single_image(
        _job_id, input_path, *, source_name, update_job_progress,
        allow_vision_escalation=None, gpu_id=None,
    ):
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
        summary = process_zip_missing(job_id)

    assert processed == ["need.png"]
    assert member_has_results(result_dir, "done.png")
    assert member_has_results(result_dir, "need.png")
    assert summary["still_missing"] == 0
    assert find_job_zip_upload(job_id).is_file()
