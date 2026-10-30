"""Per-archive worker resolution (PDF-only cap, explicit workers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import settings
from app.core.constants import SourceKind
from app.services.zip_queue import ZipMember, resolve_zip_parallel_workers


@pytest.fixture(autouse=True)
def _restore_settings() -> None:
    saved = (
        settings.zip_parallel_workers,
        settings.zip_pdf_max_parallel,
    )
    yield
    settings.zip_parallel_workers, settings.zip_pdf_max_parallel = saved


def _pdf(name: str, size: int = 100) -> ZipMember:
    return ZipMember(Path(name), name, SourceKind.PDF, size)


def _jpg(name: str, size: int = 100) -> ZipMember:
    return ZipMember(Path(name), name, SourceKind.IMAGE, size)


def test_pdf_only_cap_limits_workers() -> None:
    settings.zip_parallel_workers = 4
    settings.zip_pdf_max_parallel = 1
    workers = resolve_zip_parallel_workers([_pdf("a.pdf"), _pdf("b.pdf")])
    assert workers == 1


def test_pdf_only_cap_ignored_when_zero() -> None:
    settings.zip_parallel_workers = 4
    settings.zip_pdf_max_parallel = 0
    workers = resolve_zip_parallel_workers([_pdf("a.pdf"), _pdf("b.pdf")])
    assert workers >= 2


def test_pdf_cap_ignored_for_mixed_archive() -> None:
    settings.zip_parallel_workers = 4
    settings.zip_pdf_max_parallel = 1
    members = [_pdf("a.pdf"), _jpg("b.jpg")]
    workers = resolve_zip_parallel_workers(members)
    assert workers == 2  # bounded only by member count


def test_explicit_workers_is_upper_bound() -> None:
    settings.zip_parallel_workers = 2
    settings.zip_pdf_max_parallel = 0
    members = [_pdf(f"f{i}.pdf") for i in range(8)]
    assert resolve_zip_parallel_workers(members) == 2
