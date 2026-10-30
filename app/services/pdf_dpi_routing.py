"""
Heuristic PDF render DPI routing (throughput vs OCR quality).

Routes:
  * simple_receipt — low DPI (1–2 page, few regions, clean scan)
  * dense_document — high DPI (many pages, large pages, multi-region)
  * photo_scan — adaptive DPI (phone-photo-like luminance)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


class DpiRoute(str, Enum):
    SIMPLE_RECEIPT = "simple_receipt"
    DENSE_DOCUMENT = "dense_document"
    PHOTO_SCAN = "photo_scan"
    DEFAULT = "default"


@dataclass
class PageOutline:
    """PDF page dimensions without full render."""

    index: int
    width_pt: float
    height_pt: float

    @property
    def area_sq_in(self) -> float:
        return (self.width_pt / 72.0) * (self.height_pt / 72.0)


@dataclass
class DpiRoutingDecision:
    """Chosen DPI for one page or whole document."""

    dpi: int
    route: DpiRoute
    reasons: list[str] = field(default_factory=list)
    page_index: int | None = None
    probe_luma_std: float | None = None
    region_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dpi": self.dpi,
            "route": self.route.value,
            "reasons": self.reasons,
            "page_index": self.page_index,
            "probe_luma_std": self.probe_luma_std,
            "region_count": self.region_count,
        }


@dataclass
class PdfDpiAnalysis:
    """Full-document analysis (may use per-page DPI)."""

    page_count: int
    file_bytes: int
    pages: list[PageOutline]
    document_decision: DpiRoutingDecision
    page_decisions: list[DpiRoutingDecision] = field(default_factory=list)
    uniform_dpi: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_count": self.page_count,
            "file_bytes": self.file_bytes,
            "document_route": self.document_decision.to_dict(),
            "uniform_dpi": self.uniform_dpi,
            "page_decisions": [d.to_dict() for d in self.page_decisions],
        }


@dataclass
class DpiRoutingMetrics:
    """Aggregate routing stats (in-process)."""

    routed: int = 0
    by_route: dict[str, int] = field(default_factory=dict)
    dpi_sum: int = 0

    def record(self, decision: DpiRoutingDecision) -> None:
        self.routed += 1
        self.by_route[decision.route.value] = (
            self.by_route.get(decision.route.value, 0) + 1
        )
        self.dpi_sum += decision.dpi

    def to_dict(self) -> dict[str, Any]:
        return {
            "routed": self.routed,
            "by_route": dict(self.by_route),
            "avg_dpi": round(self.dpi_sum / max(self.routed, 1), 1),
        }


_metrics = DpiRoutingMetrics()
_metrics_lock = threading.Lock()


def get_dpi_routing_metrics() -> DpiRoutingMetrics:
    with _metrics_lock:
        return DpiRoutingMetrics(
            routed=_metrics.routed,
            by_route=dict(_metrics.by_route),
            dpi_sum=_metrics.dpi_sum,
        )


def reset_dpi_routing_metrics() -> None:
    global _metrics
    with _metrics_lock:
        _metrics = DpiRoutingMetrics()


def log_dpi_routing_metrics(*, label: str = "") -> None:
    m = get_dpi_routing_metrics()
    prefix = f"{label}: " if label else ""
    logger.info(
        "%sPDF DPI routing — routed=%d by_route=%s avg_dpi=%.1f",
        prefix,
        m.routed,
        m.by_route,
        m.dpi_sum / max(m.routed, 1),
    )


def _record_decision(decision: DpiRoutingDecision) -> None:
    with _metrics_lock:
        _metrics.record(decision)


def collect_page_outlines(pdf_path: Path) -> list[PageOutline]:
    import pypdfium2 as pdfium

    outlines: list[PageOutline] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            outlines.append(PageOutline(
                index=index,
                width_pt=float(page.get_width()),
                height_pt=float(page.get_height()),
            ))
    finally:
        pdf.close()
    return outlines


def _luminance_std(image: Image.Image) -> float:
    arr = np.array(image.convert("L"), dtype=np.float32)
    return float(arr.std())


def _probe_page_image(pdf_path: Path, page_index: int) -> Image.Image | None:
    if not settings.pdf_dpi_routing_probe_render:
        return None
    from app.services.pdf_render import render_pdf_page_pil

    return render_pdf_page_pil(
        pdf_path,
        page_index=page_index,
        dpi=settings.pdf_dpi_probe,
    )


def _region_count_probe(pdf_path: Path, page_index: int, probe: Image.Image) -> int:
    """Segmentation count on a low-DPI probe (optional)."""
    if not settings.pdf_dpi_routing_region_probe:
        return 0
    from app.services.document_segmentation import extract_documents
    from app.services.scan_enhance import prepare_page_for_ocr

    page_for_ocr = prepare_page_for_ocr(probe)
    try:
        regions = extract_documents(
            probe,
            ocr_page_image=page_for_ocr,
            enable_segmentation=True,
            enable_auto_rotate=settings.pdf_auto_rotate,
            min_area_pct=settings.pdf_min_region_area_pct,
            min_side_pct=settings.pdf_min_region_side_pct,
            max_regions=settings.pdf_max_regions_per_page,
            osd_min_confidence=settings.pdf_osd_min_confidence,
        )
        return len(regions)
    finally:
        probe.close()


def _dpi_for_photo(luma_std: float) -> int:
    """Adaptive DPI: noisier / higher-variance scans → higher render resolution."""
    base = settings.pdf_dpi_photo_base
    max_dpi = settings.pdf_dpi_photo_max
    threshold = settings.pdf_dpi_photo_luma_std_threshold
    if luma_std <= threshold:
        return base
    # Scale roughly linearly above threshold up to max.
    excess = min(luma_std - threshold, 40.0)
    boost = int(excess * settings.pdf_dpi_photo_std_scale)
    return min(max_dpi, base + boost)


def route_page(
    outline: PageOutline,
    *,
    page_count: int,
    file_bytes: int,
    probe: Image.Image | None = None,
    region_count: int | None = None,
) -> DpiRoutingDecision:
    """Heuristic DPI for one page."""
    reasons: list[str] = []
    luma_std: float | None = None

    if probe is not None:
        luma_std = _luminance_std(probe)
        if luma_std >= settings.pdf_dpi_photo_luma_std_threshold:
            dpi = _dpi_for_photo(luma_std)
            decision = DpiRoutingDecision(
                dpi=dpi,
                route=DpiRoute.PHOTO_SCAN,
                reasons=[f"photo_luma_std={luma_std:.1f}"],
                page_index=outline.index,
                probe_luma_std=luma_std,
                region_count=region_count,
            )
            _record_decision(decision)
            return decision

    bytes_per_page = file_bytes / max(page_count, 1)
    dense = False
    if page_count >= settings.pdf_dpi_dense_min_pages:
        dense = True
        reasons.append(f"pages={page_count}>={settings.pdf_dpi_dense_min_pages}")
    if bytes_per_page >= settings.pdf_dpi_dense_min_bytes_per_page:
        dense = True
        reasons.append(
            f"bytes_per_page={int(bytes_per_page)}>="
            f"{settings.pdf_dpi_dense_min_bytes_per_page}",
        )
    if outline.area_sq_in >= settings.pdf_dpi_dense_min_area_sq_in:
        dense = True
        reasons.append(
            f"area_sq_in={outline.area_sq_in:.1f}>="
            f"{settings.pdf_dpi_dense_min_area_sq_in}",
        )
    if region_count is not None and region_count > settings.pdf_dpi_simple_max_regions:
        dense = True
        reasons.append(
            f"regions={region_count}>{settings.pdf_dpi_simple_max_regions}",
        )

    if dense:
        decision = DpiRoutingDecision(
            dpi=settings.pdf_dpi_dense,
            route=DpiRoute.DENSE_DOCUMENT,
            reasons=reasons or ["dense_heuristic"],
            page_index=outline.index,
            probe_luma_std=luma_std,
            region_count=region_count,
        )
        _record_decision(decision)
        return decision

    simple = (
        page_count <= settings.pdf_dpi_simple_max_pages
        and (region_count is None or region_count <= settings.pdf_dpi_simple_max_regions)
        and outline.area_sq_in <= settings.pdf_dpi_simple_max_area_sq_in
    )
    if simple:
        decision = DpiRoutingDecision(
            dpi=settings.pdf_dpi_simple,
            route=DpiRoute.SIMPLE_RECEIPT,
            reasons=[
                f"pages={page_count}",
                f"area_sq_in={outline.area_sq_in:.1f}",
            ],
            page_index=outline.index,
            probe_luma_std=luma_std,
            region_count=region_count,
        )
        _record_decision(decision)
        return decision

    decision = DpiRoutingDecision(
        dpi=settings.pdf_dpi_default,
        route=DpiRoute.DEFAULT,
        reasons=["default_fallback"],
        page_index=outline.index,
        probe_luma_std=luma_std,
        region_count=region_count,
    )
    _record_decision(decision)
    return decision


def analyze_pdf_dpi(pdf_path: Path) -> PdfDpiAnalysis:
    """
    Analyze PDF and choose per-page DPI (may differ for photo vs text pages).
    """
    if not settings.pdf_dpi_routing_enabled:
        fallback = DpiRoutingDecision(
            dpi=settings.pdf_dpi,
            route=DpiRoute.DEFAULT,
            reasons=["routing_disabled"],
        )
        outlines = collect_page_outlines(pdf_path)
        return PdfDpiAnalysis(
            page_count=len(outlines),
            file_bytes=pdf_path.stat().st_size,
            pages=outlines,
            document_decision=fallback,
            page_decisions=[fallback] * len(outlines),
            uniform_dpi=True,
        )

    outlines = collect_page_outlines(pdf_path)
    file_bytes = pdf_path.stat().st_size
    page_count = len(outlines)
    page_decisions: list[DpiRoutingDecision] = []

    for outline in outlines:
        probe = _probe_page_image(pdf_path, outline.index)
        regions: int | None = None
        if probe is not None and settings.pdf_dpi_routing_region_probe:
            regions = _region_count_probe(pdf_path, outline.index, probe)
        decision = route_page(
            outline,
            page_count=page_count,
            file_bytes=file_bytes,
            probe=probe,
            region_count=regions,
        )
        if probe is not None:
            probe.close()
        page_decisions.append(decision)

    dpis = {d.dpi for d in page_decisions}
    uniform = len(dpis) <= 1
    doc_dpi = max(dpis) if page_decisions else settings.pdf_dpi_default
    dominant = max(page_decisions, key=lambda d: d.dpi) if page_decisions else None
    document_decision = DpiRoutingDecision(
        dpi=doc_dpi,
        route=dominant.route if dominant else DpiRoute.DEFAULT,
        reasons=[f"document_max_dpi={doc_dpi}"],
    )

    logger.info(
        "PDF DPI routing %s: pages=%d uniform=%s routes=%s dpis=%s",
        pdf_path.name,
        page_count,
        uniform,
        {d.route.value for d in page_decisions},
        sorted(dpis),
    )

    return PdfDpiAnalysis(
        page_count=page_count,
        file_bytes=file_bytes,
        pages=outlines,
        document_decision=document_decision,
        page_decisions=page_decisions,
        uniform_dpi=uniform,
    )


def resolve_page_dpi(
    pdf_path: Path,
    page_index: int,
    *,
    analysis: PdfDpiAnalysis | None = None,
) -> DpiRoutingDecision:
    """DPI for a single page render (zip_simple, probes)."""
    if not settings.pdf_dpi_routing_enabled:
        dpi = settings.zip_simple_pdf_dpi if settings.zip_simple_receipt_enabled else settings.pdf_dpi
        return DpiRoutingDecision(
            dpi=dpi,
            route=DpiRoute.DEFAULT,
            reasons=["routing_disabled"],
            page_index=page_index,
        )
    if analysis is None:
        analysis = analyze_pdf_dpi(pdf_path)
    if page_index < len(analysis.page_decisions):
        return analysis.page_decisions[page_index]
    return analysis.document_decision
