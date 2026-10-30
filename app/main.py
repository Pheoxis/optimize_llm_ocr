import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.invoice_routes import router as invoice_router
from app.api.routes import router
from app.core.config import settings
from app.services.fast_ocr import warmup as warmup_fast_ocr
from app.services.vision_llm import warmup as warmup_vision_model

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_SWAGGER_DIR = _STATIC_DIR / "swagger"


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()

    # Warm up the fast OCR engine (RapidOCR on GPU) — needed by "smart"/"fast".
    if settings.ocr_engine in ("smart", "fast", "auto"):
        try:
            from app.services.gpu import (
                detect_cuda_device_count,
                get_bulk_ocr_gpu_ids,
                get_primary_ocr_gpu_id,
            )

            await asyncio.to_thread(warmup_fast_ocr)
            logger.info(
                "Fast OCR pre-loaded on GPU %s (bulk pool: %s, %d CUDA device(s))",
                get_primary_ocr_gpu_id(),
                get_bulk_ocr_gpu_ids(),
                detect_cuda_device_count(),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Fast OCR warmup skipped: %s", exc)

    # Warm up the vision LLM (Ollama) — needed by "smart" (escalation),
    # "vision" and "auto".
    if settings.ocr_engine in ("smart", "vision", "auto"):
        try:
            await asyncio.to_thread(warmup_vision_model)
            logger.info("Vision model %s pre-loaded", settings.vision_model)
        except Exception as exc:  # pragma: no cover
            logger.warning("Vision model warmup skipped: %s", exc)

    yield


def _openapi_url() -> str:
    prefix = (settings.api_root_path or "").rstrip("/")
    return f"{prefix}/openapi.json" if prefix else "/openapi.json"


def _static_url(path: str) -> str:
    return f"/static/{path}"


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    lifespan=lifespan,
    root_path=settings.api_root_path,
    docs_url=None,
    redoc_url=None,
    openapi_url=_openapi_url(),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

app.include_router(router, prefix=settings.api_prefix)
app.include_router(invoice_router, prefix=settings.api_prefix)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    public = (settings.api_public_url or "").strip().rstrip("/")
    schema["servers"] = [
        {"url": public or "/", "description": "Current host (Tailscale / LAN)"},
    ]
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/docs", include_in_schema=False)
async def swagger_ui() -> object:
    """Swagger UI with self-hosted assets (no external CDN — works on Tailscale)."""
    return get_swagger_ui_html(
        openapi_url=_openapi_url(),
        title=f"{settings.app_name} - Swagger UI",
        swagger_js_url=_static_url("swagger/swagger-ui-bundle.js"),
        swagger_css_url=_static_url("swagger/swagger-ui.css"),
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )


@app.get("/docs/oauth2-redirect", include_in_schema=False)
async def swagger_ui_redirect() -> object:
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def redoc_ui() -> object:
    return get_redoc_html(
        openapi_url=_openapi_url(),
        title=f"{settings.app_name} - ReDoc",
    )


@app.get("/health")
def health() -> dict[str, str | bool]:
    from app.services.queue import ping_redis

    mode = settings.effective_job_execution_mode()
    payload: dict[str, str | bool] = {
        "status": "ok",
        "version": __version__,
        "background_worker": mode == "queue",
        "job_execution_mode": mode,
    }
    if mode in {"queue", "auto"}:
        redis_ok = ping_redis()
        payload["redis"] = redis_ok
        if mode == "queue" and not redis_ok:
            payload["status"] = "degraded"
        if mode == "auto":
            payload["effective_runtime_mode"] = "queue" if redis_ok else "sync"
    return payload
