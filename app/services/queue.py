"""RQ enqueue — API must not run OCR in the request lifecycle."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

import redis
from rq import Queue

from app.core.config import settings

logger = logging.getLogger(__name__)

_connection: redis.Redis | None = None


def get_redis_connection() -> redis.Redis:
    global _connection
    if _connection is None:
        _connection = redis.from_url(settings.redis_url, decode_responses=True)
    return _connection


def get_task_queue() -> Queue:
    return Queue(
        settings.rq_queue_name,
        connection=get_redis_connection(),
        default_timeout=settings.rq_job_timeout,
    )


def ping_redis() -> bool:
    try:
        return bool(get_redis_connection().ping())
    except redis.RedisError:
        return False


def enqueue_document_job(
    job_id: UUID,
    input_path: Path,
    source_name: str,
) -> str:
    """
    Enqueue ``process_document_task``; returns RQ job id.
    Raises ``redis.RedisError`` if Redis is down.
    """
    queue = get_task_queue()
    rq_job = queue.enqueue(
        "app.workers.tasks.process_document_task",
        str(job_id),
        str(input_path),
        source_name,
        job_id=str(job_id),
    )
    logger.info("Enqueued job %s → RQ %s (%s)", job_id, rq_job.id, source_name)
    return rq_job.id
