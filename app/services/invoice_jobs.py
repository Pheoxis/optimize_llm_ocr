"""In-memory invoice job store + background processing."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from app.core.config import settings
from app.core.constants import JobStatus
from app.models.propos_invoice import (
    InvoiceExtractResponse,
    InvoiceJobResponse,
    InvoiceProcessingInfo,
)
from app.services.invoice_extraction import extract_invoice_from_file
from app.services.invoice_format import to_invoice_job_response

logger = logging.getLogger(__name__)


async def _extract_internal(
    upload_path: Path,
    *,
    source_name: str,
) -> InvoiceExtractResponse:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                extract_invoice_from_file,
                upload_path,
                source_name=source_name,
            ),
            timeout=settings.invoice_process_timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.warning("Invoice extract timed out for %s", upload_path.name)
        return InvoiceExtractResponse(
            status="failed",
            processing=InvoiceProcessingInfo(
                elapsed_ms=int(settings.invoice_process_timeout_sec * 1000),
                warnings=["timeout"],
            ),
            error={
                "code": "TIMEOUT",
                "message": (
                    f"Processing exceeded {settings.invoice_process_timeout_sec}s limit"
                ),
            },
        )
    except Exception as exc:
        logger.exception("Invoice extract failed for %s", upload_path.name)
        return InvoiceExtractResponse(
            status="failed",
            processing=InvoiceProcessingInfo(warnings=[str(exc)]),
            error={"code": "OCR_FAILED", "message": str(exc)},
        )


def _job_status_from_internal(internal: InvoiceExtractResponse) -> JobStatus:
    return JobStatus.DONE if internal.status == "done" else JobStatus.FAILED


@dataclass
class _InvoiceJobRecord:
    job_id: UUID
    external_id: str | None
    status: JobStatus
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    internal: InvoiceExtractResponse | None = None
    upload_path: Path | None = None


class InvoiceJobStore:
    def __init__(self) -> None:
        self._jobs: dict[UUID, _InvoiceJobRecord] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def create(self, *, external_id: str | None = None) -> UUID:
        job_id = uuid4()
        self._jobs[job_id] = _InvoiceJobRecord(
            job_id=job_id,
            external_id=external_id,
            status=JobStatus.QUEUED,
        )
        return job_id

    def get(self, job_id: UUID) -> InvoiceJobResponse | None:
        rec = self._jobs.get(job_id)
        if rec is None:
            return None
        return to_invoice_job_response(
            job_id,
            rec.external_id,
            job_status=rec.status,
            internal=rec.internal,
        )

    def attach_upload(self, job_id: UUID, path: Path) -> None:
        rec = self._jobs.get(job_id)
        if rec is None:
            raise KeyError(job_id)
        rec.upload_path = path
        rec.updated_at = datetime.now(UTC)

    def _set_status(self, job_id: UUID, status: JobStatus) -> None:
        rec = self._jobs.get(job_id)
        if rec is None:
            return
        rec.status = status
        rec.updated_at = datetime.now(UTC)

    def _complete(self, job_id: UUID, internal: InvoiceExtractResponse) -> None:
        rec = self._jobs.get(job_id)
        if rec is None:
            return
        rec.internal = internal
        rec.status = JobStatus.DONE if internal.status == "done" else JobStatus.FAILED
        rec.updated_at = datetime.now(UTC)

    def schedule(
        self,
        job_id: UUID,
        upload_path: Path,
        *,
        source_name: str,
    ) -> None:
        task = asyncio.create_task(
            self._run_job(job_id, upload_path, source_name=source_name),
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_job(self, job_id: UUID, upload_path: Path, *, source_name: str) -> None:
        self._set_status(job_id, JobStatus.PROCESSING)
        try:
            internal = await _extract_internal(upload_path, source_name=source_name)
        finally:
            upload_path.unlink(missing_ok=True)
        self._complete(job_id, internal)

    async def extract_sync(
        self,
        upload_path: Path,
        *,
        source_name: str,
        external_id: str | None = None,
        job_id: UUID | None = None,
    ) -> InvoiceJobResponse:
        """
        Synchronous path: process in-request, return §4.1 / §4.2 JSON.

        The HTTP connection stays open until OCR + extraction finish or timeout.
        """
        jid = job_id or uuid4()
        try:
            internal = await _extract_internal(upload_path, source_name=source_name)
        finally:
            upload_path.unlink(missing_ok=True)
        return to_invoice_job_response(
            jid,
            external_id,
            job_status=_job_status_from_internal(internal),
            internal=internal,
        )


invoice_job_store = InvoiceJobStore()
