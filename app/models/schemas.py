from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.core.constants import JobStatus, SourceKind


class JobProgress(BaseModel):
    """Progress for multi-page / multi-file jobs (A2)."""

    current: int = 0
    total: int = 0
    unit: str = "pages"  # pages | files
    percent: float = Field(0.0, ge=0.0, le=100.0)


class OcrLine(BaseModel):
    """
    One visual text line from OCR (for crop escalation, bbox exports, analytics).

    ``bbox`` is ``[x0, y0, x1, y1]`` in page/crop pixel coordinates.
    """

    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: list[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        min_length=4,
        max_length=4,
        description="x0, y0, x1, y1",
    )
    page: int = Field(1, ge=1)
    engine: str = "fast_ocr"
    document: int | None = Field(
        None,
        description="Segment index on page (bulk PDF), when applicable",
    )
    line_index: int | None = Field(
        None,
        description="0-based line order on page/document",
    )
    escalated: bool | None = Field(
        None,
        description="True when line-smart/VLM replaced the initial Rapid line",
    )
    suspicious: bool | None = Field(
        None,
        description="Rapid vs Tesseract disagreement exceeds suspicious threshold",
    )
    disagreement: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Normalized Levenshtein distance Rapid↔Tesseract (0=agree)",
    )
    rapid_text: str | None = Field(
        None,
        description="RapidOCR text before verification / escalation",
    )
    tesseract_text: str | None = Field(
        None,
        description="Tesseract text on the same line crop",
    )
    verification_skipped: bool | None = Field(
        None,
        description="Tesseract skipped (e.g. very high Rapid confidence)",
    )
    escalation_hint: str | None = Field(
        None,
        description="Future escalation hook: suspicious_disagreement, ocr_disagreement, …",
    )
    validation_passed: bool | None = Field(
        None,
        description="False when semantic checksum/format validation failed on this line",
    )
    validation_issues: list[dict[str, Any]] | None = Field(
        None,
        description="Invalid semantic field detections (kind, value, reason, severity)",
    )
    validation_failed_kinds: list[str] | None = Field(
        None,
        description="Field kinds that failed validation (pesel, nip, …)",
    )


class ProcessingResult(BaseModel):
    """
    Final output contract: one text block per source file.
    Exported as result.json + result.txt (A1).
    """

    source: str = Field(..., description="Original filename")
    source_kind: SourceKind
    processed_at: datetime
    page_count: int = 0
    language_detected: list[str] = Field(default_factory=list)
    text: str = Field(..., description="Single merged text block for the whole file")
    lines: list[OcrLine] = Field(
        default_factory=list,
        description="Structured per-line OCR (empty when engine has no line geometry)",
    )
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    engine: str = "tesseract"  # tesseract | paddleocr | hybrid
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobCreateResponse(BaseModel):
    job_id: UUID
    status: JobStatus = JobStatus.QUEUED
    message: str = "Job accepted"


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    progress: JobProgress | None = None
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    results_folder: str | None = Field(
        None,
        description="Subdirectory name under storage/results/ (source filename stem)",
    )
    result_files: list[str] = Field(
        default_factory=list,
        description="Relative paths to result.json / result.txt when done",
    )
