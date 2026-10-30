"""Result directory naming from source filenames."""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import UUID

from app.core.config import settings

_MAX_FOLDER_LEN = 180


def sanitize_results_folder_name(source_name: str) -> str:
    """Filesystem-safe folder name from an upload / sample filename."""
    name = Path(source_name).name.strip()
    if not name:
        return "document"
    stem = Path(name).stem.strip() or name
    safe = re.sub(r"[^\w.\-+]", "_", stem, flags=re.UNICODE)
    safe = re.sub(r"_+", "_", safe).strip("._-")
    if not safe:
        safe = "document"
    if len(safe) > _MAX_FOLDER_LEN:
        safe = safe[:_MAX_FOLDER_LEN].rstrip("._-")
    return safe


def _unique_folder_name(base: str) -> str:
    """Pick ``base`` or ``base-2``, ``base-3``, … if the directory already exists."""
    candidate = base
    n = 2
    while (settings.results_dir / candidate).exists():
        suffix = f"-{n}"
        max_base = _MAX_FOLDER_LEN - len(suffix)
        trimmed = base[:max_base].rstrip("._-") if len(base) > max_base else base
        candidate = f"{trimmed}{suffix}"
        n += 1
    return candidate


def _folder_from_result_files(result_files: list[str]) -> str | None:
    for rel in result_files:
        p = Path(rel)
        if p.name in ("result.json", "result.txt") and len(p.parts) >= 2:
            return p.parent.name
    return None


def _folder_from_job_json(job_id: UUID) -> str | None:
    needle = str(job_id)
    root = settings.results_dir
    if not root.is_dir():
        return None
    for job_json in root.glob("*/job.json"):
        try:
            data = json.loads(job_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("job_id") != needle:
            continue
        folder = data.get("results_folder")
        if isinstance(folder, str) and folder:
            return folder
        return job_json.parent.name
    return None


def allocate_result_dir(job_id: UUID, source_name: str | None) -> Path:
    """
    Create and register the results directory for a job.

    Folder name is stored on the job record (``results_folder``), not in a sidecar index.
    """
    from app.services.jobs import job_store

    if settings.results_dir_use_source_name and source_name:
        folder = _unique_folder_name(sanitize_results_folder_name(source_name))
    else:
        folder = str(job_id)

    job_store.update(job_id, results_folder=folder)
    result_dir = settings.results_dir / folder
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def get_result_dir(job_id: UUID) -> Path:
    """Locate ``storage/results/`` directory for *job_id*."""
    from app.services.jobs import job_store

    job = job_store.get(job_id)
    if job is not None:
        if job.results_folder:
            path = settings.results_dir / job.results_folder
            if path.is_dir():
                return path
        folder = _folder_from_result_files(job.result_files)
        if folder:
            path = settings.results_dir / folder
            if path.is_dir():
                return path

    folder = _folder_from_job_json(job_id)
    if folder:
        path = settings.results_dir / folder
        if path.is_dir():
            return path

    legacy = settings.results_dir / str(job_id)
    if legacy.is_dir():
        return legacy

    if job is not None and job.results_folder:
        return settings.results_dir / job.results_folder

    return legacy
