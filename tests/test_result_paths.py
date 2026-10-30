"""Result directory naming from source filenames."""

from uuid import uuid4

import pytest

from app.core.config import settings
from app.services.jobs import job_store
from app.services.result_paths import (
    allocate_result_dir,
    get_result_dir,
    sanitize_results_folder_name,
)


def test_sanitize_results_folder_name():
    assert sanitize_results_folder_name("paragon_1.jpg") == "paragon_1"
    assert sanitize_results_folder_name("../../evil/name.pdf") == "name"
    assert sanitize_results_folder_name("SKMBT_C36026051111470-part-1.pdf") == (
        "SKMBT_C36026051111470-part-1"
    )


@pytest.fixture
def sync_job_store(monkeypatch):
    monkeypatch.setattr(settings, "job_execution_mode", "sync")
    monkeypatch.setattr(settings, "use_background_worker", False)


def test_allocate_and_get_result_dir(tmp_path, monkeypatch, sync_job_store):
    monkeypatch.setattr(settings, "results_dir", tmp_path / "results")
    monkeypatch.setattr(settings, "results_dir_use_source_name", True)

    job = job_store.create()
    result_dir = allocate_result_dir(job.job_id, "paragon_1.jpg")
    assert result_dir == tmp_path / "results" / "paragon_1"
    assert result_dir.is_dir()
    assert get_result_dir(job.job_id) == result_dir
    refreshed = job_store.get(job.job_id)
    assert refreshed is not None
    assert refreshed.results_folder == "paragon_1"


def test_collision_suffix(tmp_path, monkeypatch, sync_job_store):
    monkeypatch.setattr(settings, "results_dir", tmp_path / "results")
    monkeypatch.setattr(settings, "results_dir_use_source_name", True)

    first = allocate_result_dir(job_store.create().job_id, "dup.pdf")
    assert first.name == "dup"

    second = allocate_result_dir(job_store.create().job_id, "dup.pdf")
    assert second.name == "dup-2"
    assert second != first


def test_uuid_fallback_when_disabled(tmp_path, monkeypatch, sync_job_store):
    monkeypatch.setattr(settings, "results_dir", tmp_path / "results")
    monkeypatch.setattr(settings, "results_dir_use_source_name", False)

    job = job_store.create()
    job_id = job.job_id
    result_dir = allocate_result_dir(job_id, "ignored.jpg")
    assert result_dir.name == str(job_id)
    assert get_result_dir(job_id) == result_dir
