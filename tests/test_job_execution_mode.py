from app.api import routes
from app.core.config import settings


def test_resolve_mode_sync() -> None:
    settings.job_execution_mode = "sync"
    settings.use_background_worker = True
    assert routes._resolve_job_execution_mode() == "sync"


def test_resolve_mode_queue() -> None:
    settings.job_execution_mode = "queue"
    settings.use_background_worker = False
    assert routes._resolve_job_execution_mode() == "queue"


def test_resolve_mode_auto_queue_when_redis_available(monkeypatch) -> None:
    settings.job_execution_mode = "auto"
    settings.use_background_worker = False
    monkeypatch.setattr(routes, "redis_job_store_available", lambda: True)
    assert routes._resolve_job_execution_mode() == "queue"


def test_resolve_mode_auto_sync_when_redis_unavailable(monkeypatch) -> None:
    settings.job_execution_mode = "auto"
    settings.use_background_worker = False
    monkeypatch.setattr(routes, "redis_job_store_available", lambda: False)
    assert routes._resolve_job_execution_mode() == "sync"


def test_effective_mode_legacy_fallback() -> None:
    settings.job_execution_mode = ""
    settings.use_background_worker = True
    assert settings.effective_job_execution_mode() == "queue"
    settings.use_background_worker = False
    assert settings.effective_job_execution_mode() == "sync"
