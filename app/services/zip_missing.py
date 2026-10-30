"""Detect and re-process ZIP members missing on-disk results after a crash."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.core.constants import JobStatus
from app.models.schemas import JobProgress
from app.services.ingest import IngestError, list_zip_documents
from app.services.zip_streaming import plan_zip_members_from_archive
from app.services.jobs import job_store
from app.services.zip_checkpoint import (
    atomic_write_json,
    checkpoint_path,
    load_checkpoint,
    member_has_results,
)
from app.services.result_paths import get_result_dir
from app.services.zip_queue import ZipMember

logger = logging.getLogger(__name__)

REPORT_FILENAME = "missing_report.json"


@dataclass
class ZipMissingReport:
    """Summary of ZIP archive vs on-disk OCR outputs."""

    job_id: str
    zip_path: str
    total_in_archive: int
    complete_count: int
    missing_count: int
    incomplete_on_disk_count: int
    checkpoint_stale_count: int
    complete_members: list[str] = field(default_factory=list)
    missing_members: list[str] = field(default_factory=list)
    incomplete_on_disk: list[str] = field(default_factory=list)
    checkpoint_stale: list[str] = field(default_factory=list)
    checkpoint_status: str | None = None
    errors_json_failed: list[str] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _member_result_paths(result_dir: Path, rel: str) -> tuple[Path, Path]:
    base = result_dir.joinpath(*Path(rel).parts)
    return base / "result.json", base / "result.txt"


def _disk_status(result_dir: Path, rel: str) -> str:
    json_p, txt_p = _member_result_paths(result_dir, rel)
    has_json = json_p.is_file()
    has_txt = txt_p.is_file()
    if has_json and has_txt:
        return "complete"
    if has_json or has_txt:
        return "incomplete"
    return "absent"


def _load_errors_failed_members(result_dir: Path) -> list[str]:
    path = result_dir / "errors.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    failures = data.get("failures")
    if not isinstance(failures, list):
        return []
    out: list[str] = []
    for entry in failures:
        if isinstance(entry, dict) and entry.get("member"):
            out.append(str(entry["member"]))
    return out


def analyze_zip_missing(job_id: UUID) -> ZipMissingReport:
    """
    Compare ZIP archive index to ``result.json`` / ``result.txt`` / checkpoint.

    Does not extract the archive body (safe for large ZIPs).
    """
    from app.services.pipeline import find_job_zip_upload

    result_dir = get_result_dir(job_id)
    zip_path = find_job_zip_upload(job_id)
    archive_entries = list_zip_documents(zip_path)
    archive_members = [name for name, _size in archive_entries]

    complete: list[str] = []
    missing: list[str] = []
    incomplete: list[str] = []

    for rel in archive_members:
        status = _disk_status(result_dir, rel)
        if status == "complete":
            complete.append(rel)
        elif status == "incomplete":
            incomplete.append(rel)
            missing.append(rel)
        else:
            missing.append(rel)

    ckpt = load_checkpoint(result_dir)
    checkpoint_stale: list[str] = []
    ckpt_status: str | None = None
    if ckpt is not None:
        ckpt_status = str(ckpt.get("status") or "in_progress")
        for rel in ckpt.get("completed_members") or []:
            if isinstance(rel, str) and rel in archive_members:
                if not member_has_results(result_dir, rel):
                    checkpoint_stale.append(rel)
                    if rel not in missing:
                        missing.append(rel)

    missing_set = sorted(set(missing), key=str.lower)
    errors_failed = _load_errors_failed_members(result_dir)
    for rel in errors_failed:
        if rel in archive_members and rel not in complete and rel not in missing_set:
            missing_set.append(rel)
    missing_set = sorted(set(missing_set), key=str.lower)

    return ZipMissingReport(
        job_id=str(job_id),
        zip_path=str(zip_path),
        total_in_archive=len(archive_members),
        complete_count=len(complete),
        missing_count=len(missing_set),
        incomplete_on_disk_count=len(incomplete),
        checkpoint_stale_count=len(checkpoint_stale),
        complete_members=sorted(complete, key=str.lower),
        missing_members=missing_set,
        incomplete_on_disk=sorted(incomplete, key=str.lower),
        checkpoint_stale=sorted(checkpoint_stale, key=str.lower),
        checkpoint_status=ckpt_status,
        errors_json_failed=errors_failed,
    )


def write_missing_report(result_dir: Path, report: ZipMissingReport) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / REPORT_FILENAME
    atomic_write_json(path, report.to_dict())
    return path


def process_zip_missing(
    job_id: UUID,
    *,
    dry_run: bool = False,
    allow_vision_escalation: bool | None = None,
) -> dict[str, Any]:
    """
    Re-queue OCR for archive members without complete ``result.json`` + ``result.txt``.

    Extracts only missing members from the ZIP (not the full archive).
    """
    from app.services.pipeline import _run_zip_job, find_job_zip_upload

    report = analyze_zip_missing(job_id)
    result_dir = get_result_dir(job_id)
    report_path = write_missing_report(result_dir, report)

    summary: dict[str, Any] = {
        **report.to_dict(),
        "report_path": str(report_path),
        "dry_run": dry_run,
        "processed": 0,
        "still_missing": report.missing_count,
    }

    if report.missing_count == 0:
        summary["message"] = "All archive members have complete results on disk"
        logger.info("ZIP missing job %s: nothing to process", job_id)
        return summary

    if dry_run:
        summary["message"] = (
            f"{report.missing_count} members would be processed (dry run)"
        )
        return summary

    zip_path = find_job_zip_upload(job_id)
    work_dir = settings.temp_dir / f"{job_id}_missing"
    work_dir.mkdir(parents=True, exist_ok=True)
    extract_root = work_dir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
        missing_set = set(report.missing_members)
        all_members = plan_zip_members_from_archive(zip_path, extract_root)
        members = [m for m in all_members if m.rel_path in missing_set]
        if len(members) != len(missing_set):
            planned = {m.rel_path for m in members}
            not_planned = sorted(missing_set - planned, key=str.lower)
            if not_planned:
                logger.warning(
                    "ZIP missing job %s: members not in archive plan: %s",
                    job_id,
                    not_planned[:10],
                )
                summary["not_planned"] = not_planned

        job_store.update(job_id, status=JobStatus.PROCESSING)
        _run_zip_job(
            job_id,
            zip_path,
            work_dir,
            result_dir,
            source_name=zip_path.name,
            allow_vision_escalation=allow_vision_escalation,
            members_to_process=members,
            archive_total=report.total_in_archive,
        )
    finally:
        import shutil

        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    after = analyze_zip_missing(job_id)
    write_missing_report(result_dir, after)
    summary["processed"] = report.missing_count - after.missing_count
    summary["still_missing"] = after.missing_count
    summary["after"] = after.to_dict()
    summary["message"] = (
        f"Processed {summary['processed']} members; "
        f"{after.missing_count} still missing"
    )

    refreshed = job_store.get(job_id)
    if refreshed is not None:
        summary["job_status"] = refreshed.status.value
        if refreshed.progress:
            summary["progress"] = refreshed.progress.model_dump()

    return summary
