"""Incremental, crash-safe checkpointing for ZIP OCR jobs."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.services.zip_queue import ZipMember

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "checkpoint.json"
CHECKPOINT_VERSION = 1


def checkpoint_path(result_dir: Path) -> Path:
    return result_dir / CHECKPOINT_FILENAME


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically (temp file + fsync + replace + dir fsync)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if tmp.exists() and not path.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def load_checkpoint(result_dir: Path) -> dict[str, Any] | None:
    path = checkpoint_path(result_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Corrupt checkpoint %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None
    return data


def member_has_results(result_dir: Path, member_rel: str) -> bool:
    """True when both ``result.json`` and ``result.txt`` exist for a member."""
    base = result_dir.joinpath(*Path(member_rel).parts)
    return (base / "result.json").is_file() and (base / "result.txt").is_file()


class ZipCheckpointStore:
    """
    In-memory ZIP job checkpoint with atomic flush after each member update.

    Thread-safe for parallel ZIP ``on_complete`` callbacks in the parent process.
    """

    def __init__(
        self,
        job_id: UUID,
        result_dir: Path,
        *,
        total_files: int,
        source_archive: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.job_id = job_id
        self.result_dir = result_dir
        self.path = checkpoint_path(result_dir)
        self._lock = threading.Lock()
        if data is None:
            data = self._empty_payload(total_files, source_archive)
        self._data = data

    @classmethod
    def load_or_create(
        cls,
        job_id: UUID,
        result_dir: Path,
        *,
        total_files: int,
        source_archive: str | None = None,
    ) -> ZipCheckpointStore:
        existing = load_checkpoint(result_dir)
        if existing is not None and str(existing.get("job_id")) == str(job_id):
            existing["total_files"] = total_files
            if source_archive:
                existing["source_archive"] = source_archive
            return cls(
                job_id,
                result_dir,
                total_files=total_files,
                source_archive=source_archive,
                data=existing,
            )
        return cls(
            job_id,
            result_dir,
            total_files=total_files,
            source_archive=source_archive,
        )

    def _empty_payload(
        self,
        total_files: int,
        source_archive: str | None,
    ) -> dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "job_id": str(self.job_id),
            "source_archive": source_archive,
            "total_files": total_files,
            "status": "in_progress",
            "updated_at": _utc_now_iso(),
            "completed_members": [],
            "failed_members": [],
            "completed": [],
            "failed": [],
        }

    @property
    def status(self) -> str:
        return str(self._data.get("status") or "in_progress")

    @property
    def completed_members(self) -> list[str]:
        raw = self._data.get("completed_members")
        return list(raw) if isinstance(raw, list) else []

    @property
    def failed_members(self) -> list[str]:
        raw = self._data.get("failed_members")
        return list(raw) if isinstance(raw, list) else []

    @property
    def processed_count(self) -> int:
        return len(self.completed_members) + len(self.failed_members)

    def has_progress(self) -> bool:
        return self.processed_count > 0 or self.path.is_file()

    def should_skip_member(self, member: ZipMember) -> bool:
        """Skip members already completed with on-disk results."""
        rel = member.rel_path
        if rel not in self.completed_members:
            return False
        return member_has_results(self.result_dir, rel)

    def pending_members(self, members: list[ZipMember]) -> list[ZipMember]:
        return [m for m in members if not self.should_skip_member(m)]

    def seed_result_files(self) -> list[str]:
        """Rebuild ``result_files`` paths from checkpoint completed entries."""
        files: list[str] = []
        completed = self._data.get("completed")
        if not isinstance(completed, list):
            return files
        for entry in completed:
            if not isinstance(entry, dict):
                continue
            for key in ("result_json", "result_txt"):
                val = entry.get(key)
                if isinstance(val, str) and val not in files:
                    files.append(val)
        return files

    def failure_dicts(self) -> list[dict[str, Any]]:
        """Failure records compatible with ``errors.json`` / pipeline."""
        out: list[dict[str, Any]] = []
        failed = self._data.get("failed")
        if not isinstance(failed, list):
            return out
        for entry in failed:
            if not isinstance(entry, dict):
                continue
            out.append({
                "member": entry.get("member"),
                "kind": entry.get("kind"),
                "size_bytes": entry.get("size_bytes"),
                "error": entry.get("error"),
                "error_type": entry.get("error_type"),
                "retry_count": int(entry.get("retry_count", 0)),
            })
        return out

    def _storage_relative(self, path: Path) -> str:
        """Project-relative path for ``result_files`` (falls back for tests)."""
        resolved = path.resolve()
        root = settings.project_root.resolve()
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            results_root = settings.results_dir.resolve()
            try:
                return str(resolved.relative_to(results_root.parent))
            except ValueError:
                return str(path)

    def record_completed(
        self,
        member: ZipMember,
        *,
        json_path: Path,
        txt_path: Path,
        retry_count: int = 0,
    ) -> None:
        rel = member.rel_path
        json_rel = self._storage_relative(json_path)
        txt_rel = self._storage_relative(txt_path)
        record = {
            "member": rel,
            "kind": member.kind.value,
            "processed_at": _utc_now_iso(),
            "retry_count": retry_count,
            "result_json": json_rel,
            "result_txt": txt_rel,
        }
        with self._lock:
            self._remove_failed_locked(rel)
            self._upsert_completed_locked(record)
            completed_list = self._data.setdefault("completed_members", [])
            if rel not in completed_list:
                completed_list.append(rel)
            self._flush_locked()

    def record_failed(
        self,
        member: ZipMember,
        *,
        error: str,
        error_type: str,
        retry_count: int = 0,
    ) -> None:
        rel = member.rel_path
        record = {
            "member": rel,
            "kind": member.kind.value,
            "size_bytes": member.size_bytes,
            "processed_at": _utc_now_iso(),
            "retry_count": retry_count,
            "error": error,
            "error_type": error_type,
        }
        with self._lock:
            self._remove_completed_locked(rel)
            self._upsert_failed_locked(record)
            failed_list = self._data.setdefault("failed_members", [])
            if rel not in failed_list:
                failed_list.append(rel)
            self._flush_locked()

    def mark_finished(self) -> None:
        with self._lock:
            self._data["status"] = "finished"
            self._data["updated_at"] = _utc_now_iso()
            self._flush_locked()

    def _upsert_completed_locked(self, record: dict[str, Any]) -> None:
        completed = self._data.setdefault("completed", [])
        if not isinstance(completed, list):
            completed = []
            self._data["completed"] = completed
        rel = record["member"]
        for i, existing in enumerate(completed):
            if isinstance(existing, dict) and existing.get("member") == rel:
                completed[i] = record
                return
        completed.append(record)

    def _upsert_failed_locked(self, record: dict[str, Any]) -> None:
        failed = self._data.setdefault("failed", [])
        if not isinstance(failed, list):
            failed = []
            self._data["failed"] = failed
        rel = record["member"]
        for i, existing in enumerate(failed):
            if isinstance(existing, dict) and existing.get("member") == rel:
                completed_retry = int(existing.get("retry_count", 0))
                if record["retry_count"] < completed_retry:
                    record["retry_count"] = completed_retry
                failed[i] = record
                return
        failed.append(record)

    def _remove_completed_locked(self, rel: str) -> None:
        completed = self._data.get("completed")
        if isinstance(completed, list):
            self._data["completed"] = [
                e for e in completed
                if not (isinstance(e, dict) and e.get("member") == rel)
            ]
        members = self._data.get("completed_members")
        if isinstance(members, list) and rel in members:
            members.remove(rel)

    def _remove_failed_locked(self, rel: str) -> None:
        failed = self._data.get("failed")
        if isinstance(failed, list):
            self._data["failed"] = [
                e for e in failed
                if not (isinstance(e, dict) and e.get("member") == rel)
            ]
        members = self._data.get("failed_members")
        if isinstance(members, list) and rel in members:
            members.remove(rel)

    def _flush_locked(self) -> None:
        self._data["updated_at"] = _utc_now_iso()
        self._data["total_files"] = int(
            self._data.get("total_files") or 0,
        )
        atomic_write_json(self.path, self._data)
