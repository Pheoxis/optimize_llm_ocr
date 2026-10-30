from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.constants import JobStatus
from app.models.schemas import JobCreateResponse, JobStatusResponse
from app.services.ingest import IngestError, validate_extension, validate_size
from app.services.jobs import job_store, redis_job_store_available
from app.services.pipeline import resume_zip_checkpoint, retry_zip_failed, run_job
from app.services.result_paths import get_result_dir
from app.services.queue import enqueue_document_job
from app.services.vision_llm import ocr_image_vision

try:
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover
    RedisError = ConnectionError  # type: ignore[misc, assignment]

router = APIRouter()


def _resolve_job_execution_mode() -> str:
    """
    Resolve per-request mode.

    Returns:
      - "queue" when Redis-backed queue should be used
      - "sync" when request-local OCR should run immediately
    """
    mode = settings.effective_job_execution_mode()
    if mode == "sync":
        return "sync"
    if mode == "queue":
        return "queue"
    # auto mode: queue only when Redis is reachable
    return "queue" if redis_job_store_available() else "sync"


def _job_result_json_path(job_id: UUID, member: str | None = None) -> Path:
    base = get_result_dir(job_id)
    if member:
        return base.joinpath(*Path(member).parts) / "result.json"
    return base / "result.json"


def _job_result_txt_path(job_id: UUID, member: str | None = None) -> Path:
    base = get_result_dir(job_id)
    if member:
        return base.joinpath(*Path(member).parts) / "result.txt"
    return base / "result.txt"


def _require_job_done(job_id: UUID) -> JobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(409, f"Job not ready: {job.status}")
    return job


@router.post("/jobs", response_model=JobCreateResponse)
async def create_job(file: UploadFile = File(...)) -> JobCreateResponse:
    settings.ensure_dirs()
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    job = job_store.create()
    safe_name = Path(file.filename).name
    dest = settings.uploads_dir / f"{job.job_id}_{safe_name}"
    content = await file.read()
    dest.write_bytes(content)

    try:
        validate_extension(dest)
        validate_size(dest)
    except IngestError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc


    mode = _resolve_job_execution_mode()

    if mode == "queue":
        if not redis_job_store_available():
            dest.unlink(missing_ok=True)
            raise HTTPException(503, "Redis unavailable — start redis-server")
        try:
            enqueue_document_job(job.job_id, dest, safe_name)
        except RedisError as exc:
            job_store.update(job.job_id, status=JobStatus.FAILED, error=str(exc))
            dest.unlink(missing_ok=True)
            raise HTTPException(503, f"Failed to enqueue job: {exc}") from exc
        selected = settings.effective_job_execution_mode()
        msg = (
            "Job queued — poll GET /jobs/{id}"
            if selected != "auto"
            else "Job queued (auto mode) — poll GET /jobs/{id}"
        )
        return JobCreateResponse(
            job_id=job.job_id,
            status=JobStatus.QUEUED,
            message=msg,
        )

    run_job(job.job_id, dest, source_name=safe_name)
    refreshed = job_store.get(job.job_id)
    selected = settings.effective_job_execution_mode()
    msg = "Job completed (sync mode)"
    if selected == "auto":
        msg = "Job completed (auto fallback to sync mode)"
    return JobCreateResponse(
        job_id=job.job_id,
        status=refreshed.status if refreshed else JobStatus.QUEUED,
        message=msg,
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: UUID) -> JobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@router.post("/jobs/{job_id}/resume")
def resume_zip_job(job_id: UUID) -> dict:
    """
    Resume a crashed or partial ZIP job from ``checkpoint.json``.

    Skips members already recorded as completed with on-disk results.
    Requires the original upload in ``storage/uploads/{job_id}_*.zip``.
    """
    result_dir = get_result_dir(job_id)
    if not (result_dir / "checkpoint.json").is_file():
        raise HTTPException(
            404,
            f"No checkpoint.json for job {job_id} — nothing to resume",
        )
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    try:
        return resume_zip_checkpoint(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/jobs/{job_id}/retry-failed")
def retry_failed_zip_members(job_id: UUID) -> dict:
    """
    Re-process only ZIP members from ``errors.json``.

    Requires the original upload in ``storage/uploads/{job_id}_*.zip``.
    """
    result_dir = get_result_dir(job_id)
    if not (result_dir / "errors.json").is_file():
        raise HTTPException(
            404,
            f"No errors.json for job {job_id} — nothing to retry",
        )
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    try:
        return retry_zip_failed(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/jobs/{job_id}/results")
def list_job_results(job_id: UUID) -> dict:
    """Relative paths to all result.json / result.txt (ZIP → one pair per member)."""
    job = _require_job_done(job_id)
    return {"job_id": str(job_id), "result_files": job.result_files}


@router.get("/jobs/{job_id}/result.json")
def download_result_json(job_id: UUID) -> dict:
    _require_job_done(job_id)
    path = _job_result_json_path(job_id)
    if not path.is_file():
        raise HTTPException(
            404,
            "Top-level result.json not found (ZIP jobs use "
            "/jobs/{id}/files/{member}/result.json — see /results)",
        )
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/jobs/{job_id}/result.txt")
def download_result_txt(job_id: UUID) -> str:
    _require_job_done(job_id)
    path = _job_result_txt_path(job_id)
    if not path.is_file():
        raise HTTPException(
            404,
            "Top-level result.txt not found (ZIP jobs use "
            "/jobs/{id}/files/{member}/result.txt — see /results)",
        )
    return path.read_text(encoding="utf-8")


@router.get("/jobs/{job_id}/files/{member_path:path}/result.json")
def download_member_result_json(job_id: UUID, member_path: str) -> dict:
    _require_job_done(job_id)
    path = _job_result_json_path(job_id, member_path)
    if not path.is_file():
        raise HTTPException(404, f"Result not found for member: {member_path}")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/jobs/{job_id}/files/{member_path:path}/result.txt")
def download_member_result_txt(job_id: UUID, member_path: str) -> str:
    _require_job_done(job_id)
    path = _job_result_txt_path(job_id, member_path)
    if not path.is_file():
        raise HTTPException(404, f"Result not found for member: {member_path}")
    return path.read_text(encoding="utf-8")


# ── streaming OCR (single-image fast path, no job tracking) ──────────────────
@router.post("/ocr/stream", response_class=StreamingResponse)
async def ocr_stream(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Fast-path OCR for a single image: streams tokens as they're produced
    by the vision LLM, so the user sees the text appearing in real-time.

    Suitable only for image uploads (not PDF / ZIP).  Use POST /jobs for
    archive / multi-page workflows.
    """
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}:
        raise HTTPException(400, f"Streaming OCR supports images only, got {suffix}")

    settings.ensure_dirs()
    tmp_path = settings.temp_dir / f"stream_{Path(file.filename).name}"
    tmp_path.write_bytes(await file.read())

    import queue
    import threading

    q: queue.Queue[str | None] = queue.Queue()

    def _worker() -> None:
        try:
            ocr_image_vision(tmp_path, on_token=lambda piece: q.put(piece))
        except Exception as exc:  # surface error to client
            q.put(f"\n[ERROR] {exc}")
        finally:
            q.put(None)
            tmp_path.unlink(missing_ok=True)

    threading.Thread(target=_worker, daemon=True).start()

    def _gen():
        while True:
            piece = q.get()
            if piece is None:
                return
            yield piece

    return StreamingResponse(_gen(), media_type="text/plain; charset=utf-8")
