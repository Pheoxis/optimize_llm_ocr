"""Upload validation, ZIP extraction (A2)."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path, PurePosixPath

from app.core.config import settings

logger = logging.getLogger(__name__)


class IngestError(ValueError):
    pass


def validate_extension(path: Path) -> None:
    if path.suffix.lower() not in settings.allowed_extensions and path.suffix.lower() != ".zip":
        raise IngestError(f"Unsupported file type: {path.suffix}")


def validate_size(path: Path, max_bytes: int | None = None) -> None:
    limit = max_bytes or settings.max_upload_bytes
    size = path.stat().st_size
    if size > limit:
        raise IngestError(f"File exceeds limit ({size} > {limit} bytes)")


def _safe_member_name(name: str) -> str | None:
    """Normalize ZIP entry path; reject absolute paths and traversal."""
    posix = PurePosixPath(name.replace("\\", "/"))
    parts: list[str] = []
    for part in posix.parts:
        if part in ("", ".", "/"):
            continue
        if part == "..":
            return None
        parts.append(part)
    if not parts:
        return None
    return "/".join(parts)


def _skip_zip_entry(safe_name: str) -> bool:
    parts = safe_name.split("/")
    if any(p.startswith(".") for p in parts):
        return True
    if parts[0] in ("__MACOSX", "__macosx"):
        return True
    return False


def extract_zip(
    archive_path: Path,
    dest_dir: Path,
    *,
    max_uncompressed_bytes: int | None = None,
) -> list[Path]:
    """
    Extract supported documents from a ZIP into ``dest_dir``.

    Returns paths to extracted files (sorted). Raises ``IngestError`` on
    invalid archive, path traversal, unsupported entries, or size limit exceeded.
    """
    if not archive_path.is_file():
        raise IngestError(f"Archive not found: {archive_path}")

    limit = max_uncompressed_bytes or settings.max_zip_uncompressed_bytes
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[Path] = []
    total_uncompressed = 0

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            if zf.testzip() is not None:
                raise IngestError("ZIP archive is corrupted")

            for info in zf.infolist():
                if info.is_dir():
                    continue

                safe_name = _safe_member_name(info.filename)
                if safe_name is None or _skip_zip_entry(safe_name):
                    continue

                suffix = Path(safe_name).suffix.lower()
                if suffix not in settings.allowed_extensions:
                    logger.debug("Skipping unsupported ZIP entry: %s", safe_name)
                    continue

                uncompressed = info.file_size
                if uncompressed < 0:
                    raise IngestError(f"Invalid entry size for {safe_name}")
                total_uncompressed += uncompressed
                if total_uncompressed > limit:
                    raise IngestError(
                        f"ZIP uncompressed size exceeds limit "
                        f"({total_uncompressed} > {limit} bytes)"
                    )

                target = (dest_dir / safe_name).resolve()
                if dest_dir not in target.parents and target != dest_dir:
                    raise IngestError(f"Unsafe ZIP path: {info.filename}")

                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

                extracted.append(target)

    except zipfile.BadZipFile as exc:
        raise IngestError(f"Invalid ZIP archive: {archive_path.name}") from exc

    if not extracted:
        raise IngestError("ZIP contains no supported documents (pdf, jpg, png, …)")

    return sorted(extracted, key=lambda p: str(p).lower())


def list_zip_documents(archive_path: Path) -> list[tuple[str, int]]:
    """
    List supported documents in a ZIP using the central directory only.

    Does not decompress member bodies — suitable for very large archives when
    comparing archive contents to on-disk results.
    """
    if not archive_path.is_file():
        raise IngestError(f"Archive not found: {archive_path}")

    entries: list[tuple[str, int]] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                safe_name = _safe_member_name(info.filename)
                if safe_name is None or _skip_zip_entry(safe_name):
                    continue
                suffix = Path(safe_name).suffix.lower()
                if suffix not in settings.allowed_extensions:
                    continue
                entries.append((safe_name, int(info.file_size)))
    except zipfile.BadZipFile as exc:
        raise IngestError(f"Invalid ZIP archive: {archive_path.name}") from exc

    if not entries:
        raise IngestError("ZIP contains no supported documents (pdf, jpg, png, …)")

    return sorted(entries, key=lambda item: item[0].lower())


def extract_zip_members(
    archive_path: Path,
    dest_dir: Path,
    only_members: set[str],
    *,
    max_uncompressed_bytes: int | None = None,
) -> list[Path]:
    """
    Extract only the listed ZIP members (posix paths as returned by ``list_zip_documents``).
    """
    if not only_members:
        return []

    limit = max_uncompressed_bytes or settings.max_zip_uncompressed_bytes
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[Path] = []
    total_uncompressed = 0
    wanted = {m.lower() for m in only_members}

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            if zf.testzip() is not None:
                raise IngestError("ZIP archive is corrupted")

            for info in zf.infolist():
                if info.is_dir():
                    continue
                safe_name = _safe_member_name(info.filename)
                if safe_name is None or safe_name.lower() not in wanted:
                    continue
                if _skip_zip_entry(safe_name):
                    continue

                suffix = Path(safe_name).suffix.lower()
                if suffix not in settings.allowed_extensions:
                    continue

                uncompressed = info.file_size
                if uncompressed < 0:
                    raise IngestError(f"Invalid entry size for {safe_name}")
                total_uncompressed += uncompressed
                if total_uncompressed > limit:
                    raise IngestError(
                        f"ZIP uncompressed size exceeds limit "
                        f"({total_uncompressed} > {limit} bytes)"
                    )

                target = (dest_dir / safe_name).resolve()
                if dest_dir not in target.parents and target != dest_dir:
                    raise IngestError(f"Unsafe ZIP path: {info.filename}")

                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

                extracted.append(target)

    except zipfile.BadZipFile as exc:
        raise IngestError(f"Invalid ZIP archive: {archive_path.name}") from exc

    return sorted(extracted, key=lambda p: str(p).lower())
