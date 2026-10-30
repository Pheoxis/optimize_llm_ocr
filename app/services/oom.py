"""CUDA OOM detection + retry helper.

OCR engines on GPU occasionally hit transient OOM (CUBLAS / cudnn alloc) when
many ZIP members compete for VRAM. We detect the error string, free torch
caches and retry once on the same GPU before propagating.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


_OOM_RE = re.compile(
    r"(out of memory"
    r"|CUDA out of memory"
    r"|CUBLAS_STATUS_ALLOC_FAILED"
    r"|CUBLAS_STATUS_NOT_INITIALIZED"
    r"|CUDNN_STATUS_ALLOC_FAILED"
    r"|CUDNN_STATUS_NOT_INITIALIZED"
    r"|cudaErrorMemoryAllocation"
    r"|failed to allocate"
    r"|CUDA_ERROR_OUT_OF_MEMORY"
    r"|OrtCudaError.*out of memory"
    r"|onnxruntime.*alloc"
    r")",
    re.IGNORECASE,
)


def is_cuda_oom_error(exc: BaseException) -> bool:
    """True for any flavour of CUDA / cuBLAS / cuDNN / ORT OOM."""
    if exc is None:
        return False
    if _OOM_RE.search(str(exc)):
        return True
    cls_name = type(exc).__name__
    if "OutOfMemory" in cls_name or "OOM" in cls_name:
        return True
    cause = getattr(exc, "__cause__", None)
    if cause and cause is not exc and is_cuda_oom_error(cause):
        return True
    context = getattr(exc, "__context__", None)
    if context and context is not exc and is_cuda_oom_error(context):
        return True
    return False


def empty_cuda_cache(gpu_id: int | None = None) -> None:
    """Free torch CUDA cache (no-op without torch / CUDA)."""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return
        if gpu_id is None:
            torch.cuda.empty_cache()
            return
        try:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.cuda.empty_cache() failed: %s", exc)


def with_oom_retry(
    fn: Callable[..., T],
    *args: object,
    gpu_id: int | None = None,
    max_retries: int = 1,
    label: str = "ocr",
    **kwargs: object,
) -> T:
    """Call ``fn``; on CUDA OOM clear cache and retry up to ``max_retries`` times.

    On final failure the original exception is re-raised so the caller
    (typically the ZIP per-member wrapper) can record it as a partial failure.
    """
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if not is_cuda_oom_error(exc):
                raise
            if attempt >= max_retries:
                logger.error(
                    "CUDA OOM in %s (GPU=%s) — no more retries (%s)",
                    label, gpu_id, exc,
                )
                raise
            attempt += 1
            logger.warning(
                "CUDA OOM in %s (GPU=%s, attempt %d/%d) — empty_cache + retry",
                label, gpu_id, attempt, max_retries,
            )
            empty_cuda_cache(gpu_id)
            time.sleep(0.25 * attempt)
