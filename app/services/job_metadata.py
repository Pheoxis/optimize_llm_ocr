"""Persist job snapshot to disk (A2) — recovery without DB."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.constants import JobStatus
from app.models.schemas import JobStatusResponse
from app.services.result_paths import get_result_dir

logger = logging.getLogger(__name__)


def persist_job_snapshot(job: JobStatusResponse) -> Path | None:
    """
    Write ``storage/results/{results_folder}/job.json`` when job reaches a terminal state.
    """
    if job.status not in (JobStatus.DONE, JobStatus.FAILED):
        return None

    result_dir = get_result_dir(job.job_id)
    result_dir.mkdir(parents=True, exist_ok=True)

    duration_ms = int((job.updated_at - job.created_at).total_seconds() * 1000)
    payload: dict = {
        "job_id": str(job.job_id),
        "results_folder": job.results_folder,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "duration_ms": duration_ms,
        "error": job.error,
        "result_files": job.result_files,
    }
    if job.progress:
        payload["progress"] = job.progress.model_dump()
        if job.progress.unit == "files":
            payload["files_total"] = job.progress.total
            payload["files_done"] = job.progress.current

    path = result_dir / "job.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.debug("Wrote job metadata %s", path)
    return path
