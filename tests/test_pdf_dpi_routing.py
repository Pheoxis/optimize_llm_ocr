"""Dynamic PDF DPI routing heuristics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.pdf_dpi_routing import (
    DpiRoute,
    PageOutline,
    analyze_pdf_dpi,
    reset_dpi_routing_metrics,
    route_page,
)


def test_simple_receipt_route() -> None:
    reset_dpi_routing_metrics()
    outline = PageOutline(index=0, width_pt=200, height_pt=400)
    with patch.object(settings, "pdf_dpi_routing_enabled", True):
        decision = route_page(
            outline,
            page_count=1,
            file_bytes=50_000,
            probe=None,
            region_count=1,
        )
    assert decision.route == DpiRoute.SIMPLE_RECEIPT
    assert decision.dpi == settings.pdf_dpi_simple


def test_dense_document_route() -> None:
    reset_dpi_routing_metrics()
    outline = PageOutline(index=0, width_pt=612, height_pt=792)
    with patch.object(settings, "pdf_dpi_routing_enabled", True):
        decision = route_page(
            outline,
            page_count=5,
            file_bytes=5_000_000,
            probe=None,
            region_count=0,
        )
    assert decision.route == DpiRoute.DENSE_DOCUMENT
    assert decision.dpi == settings.pdf_dpi_dense


def test_photo_scan_adaptive_dpi() -> None:
    reset_dpi_routing_metrics()
    # High-variance synthetic "photo"
    arr = np.random.randint(0, 256, (200, 100), dtype=np.uint8)
    probe = Image.fromarray(np.stack([arr, arr, arr], axis=-1))
    outline = PageOutline(index=0, width_pt=200, height_pt=400)
    with patch.object(settings, "pdf_dpi_routing_enabled", True):
        decision = route_page(
            outline,
            page_count=1,
            file_bytes=80_000,
            probe=probe,
            region_count=1,
        )
    assert decision.route == DpiRoute.PHOTO_SCAN
    assert settings.pdf_dpi_photo_base <= decision.dpi <= settings.pdf_dpi_photo_max


def test_analyze_pdf_disabled_uses_fallback(tmp_path: Path) -> None:
    reset_dpi_routing_metrics()
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"placeholder")
    with (
        patch.object(settings, "pdf_dpi_routing_enabled", False),
        patch(
            "app.services.pdf_dpi_routing.collect_page_outlines",
            return_value=[PageOutline(0, 200, 400)],
        ),
    ):
        analysis = analyze_pdf_dpi(pdf)
    assert analysis.document_decision.dpi == settings.pdf_dpi
