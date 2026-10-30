"""propOS invoice API — sync POST (default) or async upload + poll."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.models.propos_invoice import InvoiceJobResponse, InvoiceUploadResponse
from app.services.ingest import IngestError, validate_extension, validate_size
from app.services.invoice_jobs import invoice_job_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["invoice"])

_INVOICE_EXTENSIONS = frozenset({".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"})


def _check_bearer(authorization: str | None) -> None:
    token = settings.ocr_service_token.strip()
    if not token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    provided = authorization.removeprefix("Bearer ").strip()
    if provided != token:
        raise HTTPException(401, "Invalid token")


def _validate_invoice_file(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in _INVOICE_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(_INVOICE_EXTENSIONS))}",
        )
    if suffix == ".zip":
        raise HTTPException(400, "ZIP not supported on invoice endpoint — upload a single PDF or image")
    try:
        validate_extension(path)
        validate_size(path, max_bytes=settings.invoice_max_upload_bytes)
    except IngestError as exc:
        msg = str(exc)
        if "exceeds limit" in msg.lower():
            raise HTTPException(413, msg) from exc
        raise HTTPException(400, msg) from exc


async def _save_upload(
    file: UploadFile,
    *,
    dest: Path,
) -> None:
    content = await file.read()
    if len(content) > settings.invoice_max_upload_bytes:
        raise HTTPException(
            413,
            f"File exceeds limit ({len(content)} > {settings.invoice_max_upload_bytes} bytes)",
        )
    dest.write_bytes(content)
    try:
        _validate_invoice_file(dest)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise


@router.post(
    "/invoice/extract",
    response_model=None,
    summary="Extract invoice JSON",
    description=(
        "**Default (sync):** connection stays open until OCR + extraction complete "
        f"(timeout {settings.invoice_process_timeout_sec}s), then returns mapped "
        "propOS JSON (§4.1 / §4.2) in the same HTTP response.\n\n"
        "**Async:** pass `async=true` for immediate 202 + job_id, then poll "
        "GET /invoice/jobs/{job_id}."
    ),
    responses={
        200: {"model": InvoiceJobResponse, "description": "Sync — full invoice JSON"},
        202: {"model": InvoiceUploadResponse, "description": "Async — job accepted"},
    },
)
async def extract_invoice(
    file: UploadFile = File(..., description="Invoice PDF or image"),
    external_id: str | None = Form(
        None,
        description="Client correlation id (propOS external_id)",
    ),
    async_mode: bool = Query(
        False,
        alias="async",
        description="If true, return 202 immediately and process in background",
    ),
    authorization: str | None = Header(None),
):
    _check_bearer(authorization)

    if not file.filename:
        raise HTTPException(400, "Missing filename")

    settings.ensure_dirs()
    safe_name = Path(file.filename).name

    if async_mode:
        job_id = invoice_job_store.create(external_id=external_id)
        upload_path = settings.uploads_dir / f"invoice_{job_id}_{safe_name}"
        await _save_upload(file, dest=upload_path)
        invoice_job_store.attach_upload(job_id, upload_path)
        invoice_job_store.schedule(job_id, upload_path, source_name=safe_name)
        body = InvoiceUploadResponse(job_id=job_id, status="queued")
        return JSONResponse(status_code=202, content=body.model_dump(mode="json"))

    job_id = invoice_job_store.create(external_id=external_id)
    upload_path = settings.temp_dir / f"invoice_sync_{job_id}_{safe_name}"
    await _save_upload(file, dest=upload_path)
    result = await invoice_job_store.extract_sync(
        upload_path,
        source_name=safe_name,
        external_id=external_id,
        job_id=job_id,
    )
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))


@router.get(
    "/invoice/jobs/{job_id}",
    response_model=InvoiceJobResponse,
    summary="Get invoice extraction result (async mode)",
    description="Returns queued/processing status or final propOS payload when done/failed.",
)
async def get_invoice_job(
    job_id: UUID,
    authorization: str | None = Header(None),
) -> InvoiceJobResponse:
    _check_bearer(authorization)

    response = invoice_job_store.get(job_id)
    if response is None:
        raise HTTPException(404, "Invoice job not found")
    return response
