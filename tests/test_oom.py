"""CUDA OOM detection + retry helper."""

from __future__ import annotations

import pytest

from app.services.oom import is_cuda_oom_error, with_oom_retry


class FakeCudaError(RuntimeError):
    """Stand-in for torch.cuda.OutOfMemoryError without importing torch."""


def test_is_cuda_oom_detects_torch_message() -> None:
    exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert is_cuda_oom_error(exc)


def test_is_cuda_oom_detects_cublas() -> None:
    exc = RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
    assert is_cuda_oom_error(exc)


def test_is_cuda_oom_detects_cudnn() -> None:
    exc = RuntimeError("CUDNN_STATUS_NOT_INITIALIZED on GPU 0")
    assert is_cuda_oom_error(exc)


def test_is_cuda_oom_detects_onnxruntime() -> None:
    exc = RuntimeError(
        "[ONNXRuntimeError] : 1 : FAIL : cudaErrorMemoryAllocation: "
        "failed to allocate memory"
    )
    assert is_cuda_oom_error(exc)


def test_is_cuda_oom_detects_via_class_name() -> None:
    exc = FakeCudaError("OutOfMemoryError on device 0")
    assert is_cuda_oom_error(exc)


def test_is_cuda_oom_negative_for_regular_error() -> None:
    assert not is_cuda_oom_error(ValueError("bad input"))
    assert not is_cuda_oom_error(FileNotFoundError("missing"))


def test_is_cuda_oom_walks_cause_chain() -> None:
    inner = RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
    outer = RuntimeError("OCR job failed")
    try:
        try:
            raise inner
        except RuntimeError as exc:
            raise outer from exc
    except RuntimeError as final:
        assert is_cuda_oom_error(final)


def test_with_oom_retry_retries_once_then_succeeds() -> None:
    attempts = {"n": 0}

    def fn() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    result = with_oom_retry(fn, gpu_id=0, max_retries=1, label="t")
    assert result == "ok"
    assert attempts["n"] == 2


def test_with_oom_retry_reraises_after_max() -> None:
    def fn() -> str:
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        with_oom_retry(fn, gpu_id=0, max_retries=1, label="t")


def test_with_oom_retry_does_not_retry_non_oom() -> None:
    attempts = {"n": 0}

    def fn() -> str:
        attempts["n"] += 1
        raise ValueError("bad image")

    with pytest.raises(ValueError):
        with_oom_retry(fn, gpu_id=0, max_retries=3, label="t")
    assert attempts["n"] == 1
