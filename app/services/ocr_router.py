"""
Adaptive OCR routing based on per-image/page difficulty.

Maps difficulty score → cheapest pipeline tier that can handle the document.

Tiers
-----
  0  (score 0–30)   FAST       RapidOCR only, no verification, no VLM
  1  (score 30–60)  VERIFY     RapidOCR + Tesseract line verification
  2  (score 60–80)  ESCALATE   RapidOCR + crop-level VLM escalation
  3  (score 80–100) PREMIUM    All features, unlimited Tesseract + VLM

Usage
-----
    decision = route_image(image_path)          # or route_pdf_page(pdf_path, idx)
    with use_profile(decision.profile):
        text, conf, meta = ocr_image(image_path, ...)

Integration with existing pipeline
------------------------------------
When ``settings.adaptive_routing_enabled`` is True, ``ocr_image()`` calls
``route_image()`` automatically and merges the decision into the thread-local
profile.  The existing static-profile path is preserved as a fallback.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.difficulty_analyzer import (
    DifficultyResult,
    DocumentType,
    analyze_image,
    analyze_pdf_page,
)
from app.services.pipeline_profile import (
    IMAGE_PROFILE,
    PDF_CROP_PROFILE,
    PREMIUM_PROFILE,
    PipelineProfile,
)

logger = logging.getLogger(__name__)


# ── tier enum ─────────────────────────────────────────────────────────────────


class RoutingTier(IntEnum):
    FAST = 0       # RapidOCR only
    VERIFY = 1     # RapidOCR + Tesseract verification
    ESCALATE = 2   # RapidOCR + crop VLM escalation
    PREMIUM = 3    # all features, no budget caps


_TIER_NAMES = {
    RoutingTier.FAST: "fast",
    RoutingTier.VERIFY: "verify",
    RoutingTier.ESCALATE: "escalate",
    RoutingTier.PREMIUM: "premium",
}


# ── routing decision ──────────────────────────────────────────────────────────


@dataclass
class RoutingDecision:
    """Complete routing result for one image or PDF page."""

    tier: RoutingTier
    profile: PipelineProfile
    difficulty: DifficultyResult
    # Engine override: None = keep settings.ocr_engine; str = explicit override.
    engine_override: str | None = None
    # Whether Tesseract full-page Polish fallback should be attempted.
    allow_tesseract_fullpage: bool = False
    routing_ms: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def tier_name(self) -> str:
        return _TIER_NAMES[self.tier]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier_name,
            "difficulty_score": round(self.difficulty.difficulty_score, 1),
            "document_type": self.difficulty.document_type.value,
            "layout_type": self.difficulty.layout_type.value,
            "expected_ocr_quality": self.difficulty.expected_ocr_quality.value,
            "profile": self.profile.name,
            "engine_override": self.engine_override,
            "allow_tesseract_fullpage": self.allow_tesseract_fullpage,
            "routing_ms": round(self.routing_ms, 2),
            "reasons": self.reasons,
        }


# ── dynamic profile factory ───────────────────────────────────────────────────


def _build_profile_for_tier(tier: RoutingTier) -> PipelineProfile:
    """
    Construct a PipelineProfile tuned to the chosen tier using the
    per-tier budget settings from config.  The base profiles are used
    as templates; only the budget fields are overridden.
    """
    if tier == RoutingTier.FAST:
        return PipelineProfile(
            name="adaptive_fast",
            line_verification_enabled=False,
            tesseract_budget_per_page=settings.adaptive_tier0_tesseract_budget,
            skip_high_conf_tesseract=True,
            allow_crop_vlm=settings.adaptive_tier0_vlm_calls > 0,
            max_crop_vlm_calls=settings.adaptive_tier0_vlm_calls,
            merge_adjacent_crops=True,
            layout_reconstruction_enabled=True,
            layout_detect_barcodes=False,
            layout_detect_forms=False,
            layout_detect_columns=True,
            layout_flag_handwriting=False,
            layout_dedup_enabled=True,
        )

    if tier == RoutingTier.VERIFY:
        return PipelineProfile(
            name="adaptive_verify",
            line_verification_enabled=True,
            tesseract_budget_per_page=settings.adaptive_tier1_tesseract_budget,
            skip_high_conf_tesseract=True,
            allow_crop_vlm=settings.adaptive_tier1_vlm_calls > 0,
            max_crop_vlm_calls=settings.adaptive_tier1_vlm_calls,
            merge_adjacent_crops=True,
            layout_reconstruction_enabled=True,
            layout_detect_barcodes=True,
            layout_detect_forms=True,
            layout_detect_columns=True,
            layout_flag_handwriting=False,
            layout_dedup_enabled=True,
        )

    if tier == RoutingTier.ESCALATE:
        return PipelineProfile(
            name="adaptive_escalate",
            line_verification_enabled=True,
            tesseract_budget_per_page=settings.adaptive_tier2_tesseract_budget,
            skip_high_conf_tesseract=True,
            allow_crop_vlm=settings.adaptive_tier2_vlm_calls > 0,
            max_crop_vlm_calls=settings.adaptive_tier2_vlm_calls,
            merge_adjacent_crops=True,
            layout_reconstruction_enabled=True,
            layout_detect_barcodes=True,
            layout_detect_forms=True,
            layout_detect_columns=True,
            layout_flag_handwriting=True,
            layout_dedup_enabled=True,
        )

    # PREMIUM
    return PipelineProfile(
        name="adaptive_premium",
        line_verification_enabled=True,
        tesseract_budget_per_page=settings.adaptive_tier3_tesseract_budget,
        skip_high_conf_tesseract=False,
        allow_crop_vlm=True,
        max_crop_vlm_calls=settings.adaptive_tier3_vlm_calls,
        merge_adjacent_crops=True,
        layout_reconstruction_enabled=True,
        layout_detect_barcodes=True,
        layout_detect_forms=True,
        layout_detect_columns=True,
        layout_flag_handwriting=True,
        layout_dedup_enabled=True,
    )


# ── score → tier ──────────────────────────────────────────────────────────────


def _tier_for_score(score: float) -> RoutingTier:
    if score < settings.adaptive_tier1_threshold:
        return RoutingTier.FAST
    if score < settings.adaptive_tier2_threshold:
        return RoutingTier.VERIFY
    if score < settings.adaptive_tier3_threshold:
        return RoutingTier.ESCALATE
    return RoutingTier.PREMIUM


# ── engine override ───────────────────────────────────────────────────────────


def _engine_for_tier(tier: RoutingTier, doc_type: DocumentType) -> str | None:
    """
    Return an explicit engine override when the default "smart" mode is
    inappropriate for this tier.  None keeps settings.ocr_engine.
    """
    if tier == RoutingTier.FAST:
        # Clean document: RapidOCR result is accepted without escalation gate.
        return "fast"
    # Tiers 1–3: keep "smart" so that the quality gate can still trigger
    # crop escalation; the profile caps VLM calls as needed.
    return None


# ── routing entry points ──────────────────────────────────────────────────────


def route_image(
    source,    # PIL.Image.Image | Path | str
    *,
    fast_probe: bool | None = None,
) -> RoutingDecision:
    """
    Analyze a raster image and return a RoutingDecision.

    Parameters
    ----------
    source : PIL.Image | Path | str
    fast_probe : bool | None
        Override ``settings.adaptive_probe_fast``.  None = use setting.
    """
    t0 = time.perf_counter()
    _fast = fast_probe if fast_probe is not None else settings.adaptive_probe_fast

    try:
        difficulty = analyze_image(source, fast=_fast)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DifficultyAnalyzer failed for image, using default profile: %s", exc)
        difficulty = _fallback_difficulty()

    tier = _tier_for_score(difficulty.difficulty_score)
    profile = _build_profile_for_tier(tier)
    engine = _engine_for_tier(tier, difficulty.document_type)
    allow_tess = tier >= RoutingTier.VERIFY and settings.ocr_tesseract_polish_fullpage_fallback

    reasons = list(difficulty.reasons)
    reasons.insert(0, f"score={difficulty.difficulty_score:.1f} → tier={tier.name}")

    elapsed = (time.perf_counter() - t0) * 1000.0

    decision = RoutingDecision(
        tier=tier,
        profile=profile,
        difficulty=difficulty,
        engine_override=engine,
        allow_tesseract_fullpage=allow_tess,
        routing_ms=elapsed,
        reasons=reasons,
    )

    _log_decision(decision, label="image")
    return decision


def route_pdf_page(
    pdf_path: Path | str,
    page_index: int = 0,
    *,
    probe_dpi: int | None = None,
    fast_probe: bool | None = None,
) -> RoutingDecision:
    """
    Analyze a single PDF page and return a RoutingDecision.

    Parameters
    ----------
    pdf_path : Path | str
    page_index : int
        0-based page index.
    probe_dpi : int | None
        Override ``settings.adaptive_probe_dpi``.
    fast_probe : bool | None
        Override ``settings.adaptive_probe_fast``.
    """
    t0 = time.perf_counter()
    dpi = probe_dpi if probe_dpi is not None else settings.adaptive_probe_dpi
    _fast = fast_probe if fast_probe is not None else settings.adaptive_probe_fast

    try:
        difficulty = analyze_pdf_page(pdf_path, page_index, probe_dpi=dpi, fast=_fast)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "DifficultyAnalyzer failed for PDF %s[%d], using default profile: %s",
            pdf_path, page_index, exc,
        )
        difficulty = _fallback_difficulty()

    # Digital PDFs with embedded text are already easy regardless of visual
    # complexity — override tier to FAST when text layer is present.
    pf = difficulty.pdf_page_features
    forced_fast = pf is not None and pf.has_embedded_text

    if forced_fast:
        tier = RoutingTier.FAST
        extra_reason = "digital_pdf_override→FAST"
    else:
        tier = _tier_for_score(difficulty.difficulty_score)
        extra_reason = None

    profile = _build_profile_for_tier(tier)
    engine = _engine_for_tier(tier, difficulty.document_type)
    allow_tess = tier >= RoutingTier.VERIFY and settings.ocr_tesseract_polish_fullpage_fallback

    reasons = list(difficulty.reasons)
    reasons.insert(0, f"score={difficulty.difficulty_score:.1f} → tier={tier.name}")
    if extra_reason:
        reasons.append(extra_reason)

    elapsed = (time.perf_counter() - t0) * 1000.0

    decision = RoutingDecision(
        tier=tier,
        profile=profile,
        difficulty=difficulty,
        engine_override=engine,
        allow_tesseract_fullpage=allow_tess,
        routing_ms=elapsed,
        reasons=reasons,
    )

    _log_decision(decision, label=f"pdf[{page_index}]")
    return decision


# ── helpers ───────────────────────────────────────────────────────────────────


def _fallback_difficulty() -> DifficultyResult:
    """Synthetic DifficultyResult used when the analyzer itself fails."""
    from app.services.difficulty_analyzer import DifficultyResult, LayoutType, OcrQuality
    return DifficultyResult(
        difficulty_score=50.0,          # land in VERIFY tier — safe default
        document_type=DocumentType.UNKNOWN,
        layout_type=LayoutType.UNKNOWN,
        expected_ocr_quality=OcrQuality.FAIR,
        reasons=["analyzer_failed→fallback_score_50"],
    )


def _log_decision(decision: RoutingDecision, label: str) -> None:
    if not settings.adaptive_routing_log_decisions:
        return
    logger.debug(
        "OCR route [%s]: tier=%s  score=%.1f  doc=%s  quality=%s  "
        "profile=%s  engine=%s  probe=%.1f ms  reasons=%s",
        label,
        decision.tier_name,
        decision.difficulty.difficulty_score,
        decision.difficulty.document_type.value,
        decision.difficulty.expected_ocr_quality.value,
        decision.profile.name,
        decision.engine_override or "keep",
        decision.difficulty.analysis_ms,
        decision.reasons[:4],
    )


# ── batch routing ─────────────────────────────────────────────────────────────


def route_images_batch(
    sources: list,
    *,
    fast_probe: bool = True,
) -> list[RoutingDecision]:
    """Route a list of images/paths. Uses fast probe by default for batch speed."""
    return [route_image(src, fast_probe=fast_probe) for src in sources]


def route_pdf_pages(
    pdf_path: Path | str,
    *,
    probe_dpi: int | None = None,
    fast_probe: bool = True,
    max_pages: int | None = None,
) -> list[RoutingDecision]:
    """Route all pages of a PDF."""
    pdf_path = Path(pdf_path)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        n = doc.page_count
        doc.close()
    except Exception:  # noqa: BLE001
        return []

    n = n if max_pages is None else min(n, max_pages)
    return [
        route_pdf_page(pdf_path, i, probe_dpi=probe_dpi, fast_probe=fast_probe)
        for i in range(n)
    ]


# ── benchmark ─────────────────────────────────────────────────────────────────


def benchmark_routing(
    image_path: Path | str | None = None,
    pdf_path: Path | str | None = None,
    *,
    n_image: int = 20,
    n_pdf_page: int = 10,
) -> dict[str, Any]:
    """
    Benchmark the full routing pipeline (difficulty analysis + tier assignment).

    Returns a dict with timing statistics and tier distribution per input.
    """
    results: dict[str, Any] = {}

    if image_path is not None:
        from PIL import Image as _PILImage
        img = _PILImage.open(image_path)
        # Warm up
        route_image(img)
        timings, tiers = [], []
        for _ in range(n_image):
            d = route_image(img)
            timings.append(d.routing_ms)
            tiers.append(d.tier_name)
        results["image"] = {
            "samples": n_image,
            "mean_ms": round(statistics.mean(timings), 2),
            "min_ms": round(min(timings), 2),
            "max_ms": round(max(timings), 2),
            "p95_ms": round(sorted(timings)[int(0.95 * len(timings))], 2),
            "tier_distribution": {t: tiers.count(t) for t in set(tiers)},
        }
        logger.info(
            "Routing benchmark [image]: mean=%.1f ms  tier=%s",
            results["image"]["mean_ms"],
            results["image"]["tier_distribution"],
        )

    if pdf_path is not None:
        timings, tiers = [], []
        for i in range(n_pdf_page):
            try:
                d = route_pdf_page(pdf_path, page_index=0)
                timings.append(d.routing_ms)
                tiers.append(d.tier_name)
            except Exception:  # noqa: BLE001
                break
        if timings:
            results["pdf"] = {
                "samples": len(timings),
                "mean_ms": round(statistics.mean(timings), 2),
                "min_ms": round(min(timings), 2),
                "max_ms": round(max(timings), 2),
                "p95_ms": round(sorted(timings)[int(0.95 * len(timings))], 2),
                "tier_distribution": {t: tiers.count(t) for t in set(tiers)},
            }
            logger.info(
                "Routing benchmark [pdf]: mean=%.1f ms  tier=%s",
                results["pdf"]["mean_ms"],
                results["pdf"]["tier_distribution"],
            )

    return results
