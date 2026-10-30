"""RQ tasks — OCR runs only in worker processes."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)


def process_document_task(
    job_id: str,
    input_path: str,
    source_name: str,
) -> None:
    """
    Background OCR job. Requires ``USE_BACKGROUND_WORKER=true`` so
    ``job_store`` persists to Redis (same as API).
    """
    os.environ["DOCOCR_WORKER"] = "1"
    os.environ["USE_BACKGROUND_WORKER"] = "true"

    from app.core.config import settings
    from app.services.pipeline import run_job

    settings.ensure_dirs()
    logger.info("Worker starting job %s (%s)", job_id, source_name)
    run_job(
        UUID(job_id),
        Path(input_path),
        source_name=source_name,
    )
