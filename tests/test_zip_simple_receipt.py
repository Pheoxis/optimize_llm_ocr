"""ZIP simple-receipt PDF routing (1-page OpenPDF vs multi-page bulk)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.core.config import settings
from app.services.pipeline import should_use_zip_simple_receipt


def test_simple_when_one_page_and_probe_disabled(tmp_path: Path) -> None:
    pdf = tmp_path / "one.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    saved_probe = settings.zip_simple_receipt_probe_regions
    settings.zip_simple_receipt_probe_regions = False
    try:
        with patch("app.services.pipeline.pdf_page_count", return_value=1):
            assert should_use_zip_simple_receipt(pdf, tmp_path / "work") is True
    finally:
        settings.zip_simple_receipt_probe_regions = saved_probe


def test_simple_when_two_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "two.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    saved_probe = settings.zip_simple_receipt_probe_regions
    settings.zip_simple_receipt_probe_regions = False
    try:
        with patch("app.services.pipeline.pdf_page_count", return_value=2):
            assert should_use_zip_simple_receipt(pdf, tmp_path / "work") is True
    finally:
        settings.zip_simple_receipt_probe_regions = saved_probe


def test_bulk_when_multi_page(tmp_path: Path) -> None:
    pdf = tmp_path / "multi.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with patch("app.services.pipeline.pdf_page_count", return_value=5):
        assert should_use_zip_simple_receipt(pdf, tmp_path / "work") is False


def test_bulk_when_disabled(tmp_path: Path) -> None:
    pdf = tmp_path / "one.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    saved = settings.zip_simple_receipt_enabled
    settings.zip_simple_receipt_enabled = False
    try:
        with patch("app.services.pipeline.pdf_page_count", return_value=1):
            assert should_use_zip_simple_receipt(pdf, tmp_path / "work") is False
    finally:
        settings.zip_simple_receipt_enabled = saved


def test_simple_when_probe_few_regions(tmp_path: Path) -> None:
    pdf = tmp_path / "one.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    saved_probe = settings.zip_simple_receipt_probe_regions
    settings.zip_simple_receipt_probe_regions = True
    try:
        with (
            patch("app.services.pipeline.pdf_page_count", return_value=1),
            patch(
                "app.services.pipeline._zip_simple_receipt_region_count",
                return_value=1,
            ),
        ):
            assert should_use_zip_simple_receipt(pdf, tmp_path / "work") is True
    finally:
        settings.zip_simple_receipt_probe_regions = saved_probe


def test_bulk_when_probe_many_regions(tmp_path: Path) -> None:
    pdf = tmp_path / "dense.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    saved_probe = settings.zip_simple_receipt_probe_regions
    settings.zip_simple_receipt_probe_regions = True
    try:
        with (
            patch("app.services.pipeline.pdf_page_count", return_value=1),
            patch(
                "app.services.pipeline._zip_simple_receipt_region_count",
                return_value=4,
            ),
        ):
            assert should_use_zip_simple_receipt(pdf, tmp_path / "work") is False
    finally:
        settings.zip_simple_receipt_probe_regions = saved_probe
