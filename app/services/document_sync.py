"""Synchronous document OCR without job store (invoice API)."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.core.constants import SourceKind
from app.models.schemas import ProcessingResult
from app.services.pipeline import (
    _process_bulk_document,
    _process_single_image,
    _source_kind,
)


def ocr_document_sync(
    input_path: Path,
    *,
    source_name: str | None = None,
    allow_vision_escalation: bool = False,
) -> ProcessingResult:
    """
    Run OCR pipeline in-process (no Redis job, no result persistence).

    Supports PDF and image uploads. ZIP is rejected by the API layer.
    """
    name = source_name or input_path.name
    kind = _source_kind(input_path)
    work_dir = settings.temp_dir / f"invoice_ocr_{uuid4()}"
    try:
        if kind == SourceKind.IMAGE:
            return _process_single_image(
                None,
                input_path,
                source_name=name,
                update_job_progress=False,
                allow_vision_escalation=allow_vision_escalation,
            )
        if kind == SourceKind.PDF:
            work_dir.mkdir(parents=True, exist_ok=True)
            return _process_bulk_document(
                input_path,
                work_dir,
                source_name=name,
                job_id=None,
            )
        raise ValueError(f"Unsupported source kind: {kind}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
