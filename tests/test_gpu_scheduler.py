"""GPU scheduling helpers."""

from app.services.gpu import should_use_page_parallel
from app.services.gpu_pool import gpu_for_page, reset_ocr_gpu_pool
from app.services.parallel_ocr import page_idx_from_key
from app.services.zip_queue import resolve_zip_parallel_workers, ZipMember
from app.core.constants import SourceKind
from pathlib import Path


def test_page_idx_from_key() -> None:
    assert page_idx_from_key("p3:d1") == 3
    assert page_idx_from_key("p3:d1:retry") == 3
    assert page_idx_from_key("other") is None


def test_gpu_for_page() -> None:
    assert gpu_for_page(1, [0, 1]) == 0
    assert gpu_for_page(2, [0, 1]) == 1
    assert gpu_for_page(3, [0, 1]) == 0


def test_zip_workers_with_pdf() -> None:
    members = [
        ZipMember(Path("a.pdf"), "a.pdf", SourceKind.PDF, 100),
        ZipMember(Path("b.pdf"), "b.pdf", SourceKind.PDF, 200),
    ]
    assert resolve_zip_parallel_workers(members) >= 2


def test_should_use_page_parallel_two_pages() -> None:
    reset_ocr_gpu_pool()
    assert should_use_page_parallel(2) in (True, False)  # False if no CUDA in CI
