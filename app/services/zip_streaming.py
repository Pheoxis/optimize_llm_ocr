"""Per-member ZIP extract → OCR → cleanup (minimal disk footprint)."""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

from app.core.config import settings
from app.services.ingest import (
    IngestError,
    _safe_member_name,
    _skip_zip_entry,
    list_zip_documents,
)
from app.services.zip_queue import ZipMember, _member_kind, _sort_zip_members

logger = logging.getLogger(__name__)


def _validate_archive_total_size(archive_path: Path) -> None:
    limit = settings.max_zip_uncompressed_bytes
    total = sum(size for _name, size in list_zip_documents(archive_path))
    if total > limit:
        raise IngestError(
            f"ZIP uncompressed size exceeds limit ({total} > {limit} bytes)",
        )


def plan_zip_members_from_archive(
    archive_path: Path,
    extract_root: Path,
) -> list[ZipMember]:
    """
    Build processing plan from the ZIP central directory (no member bodies read).
    """
    _validate_archive_total_size(archive_path)
    root = extract_root.resolve()
    members: list[ZipMember] = []
    for rel, size in list_zip_documents(archive_path):
        target = (root / rel).resolve()
        if root not in target.parents and target != root:
            raise IngestError(f"Unsafe ZIP path: {rel}")
        members.append(
            ZipMember(
                path=target,
                rel_path=rel,
                kind=_member_kind(target),
                size_bytes=size,
            ),
        )
    return _sort_zip_members(members)


def extract_zip_member(
    archive_path: Path,
    extract_root: Path,
    member_rel: str,
) -> Path:
    """Extract a single archive member into ``extract_root``; return on-disk path."""
    dest_dir = extract_root.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            if zf.testzip() is not None:
                raise IngestError("ZIP archive is corrupted")

            info = None
            wanted = member_rel.replace("\\", "/")
            for zi in zf.infolist():
                safe = _safe_member_name(zi.filename)
                if safe == wanted:
                    info = zi
                    break
            if info is None or info.is_dir():
                raise IngestError(f"ZIP member not found: {member_rel}")

            safe_name = _safe_member_name(info.filename)
            if safe_name is None or _skip_zip_entry(safe_name):
                raise IngestError(f"ZIP member not extractable: {member_rel}")

            suffix = Path(safe_name).suffix.lower()
            if suffix not in settings.allowed_extensions:
                raise IngestError(f"Unsupported ZIP member type: {member_rel}")

            if info.file_size > settings.max_zip_uncompressed_bytes:
                raise IngestError(
                    f"ZIP member {member_rel} exceeds size limit ({info.file_size} bytes)",
                )

            target = (dest_dir / safe_name).resolve()
            if dest_dir not in target.parents and target != dest_dir:
                raise IngestError(f"Unsafe ZIP path: {info.filename}")

            if target.is_file():
                return target

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            return target

    except zipfile.BadZipFile as exc:
        raise IngestError(f"Invalid ZIP archive: {archive_path.name}") from exc


def ensure_zip_member_extracted(
    archive_path: Path,
    extract_root: Path,
    member: ZipMember,
) -> Path:
    """Extract member if absent; return path (must exist before OCR)."""
    if member.path.is_file():
        return member.path
    path = extract_zip_member(archive_path, extract_root, member.rel_path)
    return path


def cleanup_zip_member_extract(extract_root: Path, member: ZipMember) -> None:
    """Remove extracted member file and empty parent folders under *extract_root*."""
    target = member.path
    if target.is_file():
        try:
            target.unlink()
        except OSError as exc:
            logger.debug("Could not delete extracted member %s: %s", target, exc)

    root = extract_root.resolve()
    parent = target.parent
    while parent != root and parent.is_relative_to(root):
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def cleanup_zip_member_work(work_dir: Path, member: ZipMember) -> None:
    """Remove per-member OCR scratch (rendered pages, crops)."""
    member_work = work_dir / "members" / member.rel_path.replace("/", "__")
    if member_work.exists():
        shutil.rmtree(member_work, ignore_errors=True)


def release_zip_member_temp(
    extract_root: Path,
    work_dir: Path,
    member: ZipMember,
) -> None:
    """Drop extracted file and member work dir after OCR."""
    cleanup_zip_member_work(work_dir, member)
    cleanup_zip_member_extract(extract_root, member)
