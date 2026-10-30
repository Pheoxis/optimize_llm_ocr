"""Job store — in-memory (sync dev) or Redis (background worker)."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import redis
from redis.exceptions import RedisError

from app.core.config import settings
from app.core.constants import JobStatus
from app.models.schemas import JobProgress, JobStatusResponse

logger = logging.getLogger(__name__)

JOB_KEY_PREFIX = "dococr:job:"
JOB_TTL = timedelta(days=7)


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[UUID, JobStatusResponse] = {}

    def create(self) -> JobStatusResponse:
        job_id = uuid4()
        now = datetime.now(UTC)
        job = JobStatusResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            progress=None,
            created_at=now,
            updated_at=now,
        )
        self._jobs[job_id] = job
        return job

    def get(self, job_id: UUID) -> JobStatusResponse | None:
        return self._jobs.get(job_id)

    def update(self, job_id: UUID, **fields: object) -> JobStatusResponse | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        data = job.model_dump()
        data.update(fields)
        data["updated_at"] = datetime.now(UTC)
        updated = JobStatusResponse(**data)
        self._jobs[job_id] = updated
        return updated


class RedisJobStore:
    """Shared job state between API and RQ workers."""

    def __init__(self, url: str) -> None:
        self._redis = redis.from_url(url, decode_responses=True)

    def _key(self, job_id: UUID) -> str:
        return f"{JOB_KEY_PREFIX}{job_id}"

    def create(self) -> JobStatusResponse:
        job_id = uuid4()
        now = datetime.now(UTC)
        job = JobStatusResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            progress=None,
            created_at=now,
            updated_at=now,
        )
        self._save(job)
        return job

    def get(self, job_id: UUID) -> JobStatusResponse | None:
        raw = self._redis.get(self._key(job_id))
        if not raw:
            return None
        return JobStatusResponse.model_validate_json(raw)

    def update(self, job_id: UUID, **fields: object) -> JobStatusResponse | None:
        job = self.get(job_id)
        if job is None:
            return None
        data = job.model_dump()
        data.update(fields)
        data["updated_at"] = datetime.now(UTC)
        updated = JobStatusResponse(**data)
        self._save(updated)
        return updated

    def _save(self, job: JobStatusResponse) -> None:
        self._redis.setex(
            self._key(job.job_id),
            JOB_TTL,
            job.model_dump_json(),
        )


_memory_store = InMemoryJobStore()
_redis_store: RedisJobStore | None = None


def _get_redis_store() -> RedisJobStore:
    global _redis_store
    if _redis_store is None:
        _redis_store = RedisJobStore(settings.redis_url)
    return _redis_store


def _use_redis_store() -> bool:
    """Pick Redis backend from execution mode or worker env."""
    if os.environ.get("DOCOCR_WORKER") == "1":
        return True
    return settings.effective_job_execution_mode() == "queue"


class JobStoreFacade:
    """Pick backend from ``USE_BACKGROUND_WORKER`` / worker env."""

    def _backend(self) -> InMemoryJobStore | RedisJobStore:
        if _use_redis_store():
            return _get_redis_store()
        return _memory_store

    def create(self) -> JobStatusResponse:
        return self._backend().create()

    def get(self, job_id: UUID) -> JobStatusResponse | None:
        return self._backend().get(job_id)

    def update(self, job_id: UUID, **fields: object) -> JobStatusResponse | None:
        updated = self._backend().update(job_id, **fields)
        if updated is not None:
            _maybe_persist_metadata(updated)
        return updated


job_store = JobStoreFacade()


def _maybe_persist_metadata(job: JobStatusResponse) -> None:
    from app.core.constants import JobStatus
    from app.services.job_metadata import persist_job_snapshot

    if job.status in (JobStatus.DONE, JobStatus.FAILED):
        persist_job_snapshot(job)


def redis_job_store_available() -> bool:
    mode = settings.effective_job_execution_mode()
    if os.environ.get("DOCOCR_WORKER") != "1" and mode == "sync":
        return True
    try:
        _get_redis_store()._redis.ping()
        return True
    except RedisError as exc:
        logger.warning("Redis unavailable: %s", exc)
        return False
