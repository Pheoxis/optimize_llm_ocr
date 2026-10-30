"""
Ultra-fast 3-tier OCR difficulty classifier.

Classifies any document (image / PDF page / ZIP member) into EASY / MEDIUM /
HARD *before* OCR starts.  The primary goal is maximum throughput for simple
documents by skipping expensive processing steps they do not need.

Timing target
  * <20 ms per image (raster)
  * <50 ms per PDF page (first-page probe at 72 dpi)

Tiers
-----
  EASY   – RapidOCR only, no segmentation, no VLM, no Tesseract,
             minimal preprocessing.
             Typical: clean digital-PDF crops, high-res clear receipts.

  MEDIUM – Lightweight layout segmentation + optional line verification.
             Typical: scanned invoices / receipts with moderate noise or
             mild skew, standard multi-column layouts.

  HARD   – Full segmentation, retries, crop-level VLM (MiniCPM),
             semantic validators (PESEL, NIP, IBAN …).
             Typical: phone photos, heavy noise, handwriting, dense forms,
             barcoded documents, severely skewed scans.

Benchmark metadata
------------------
Every DifficultyDecision carries:
  routing_time_ms    – wall-clock time from call to decision (target <20 ms)
  selected_pipeline  – human-readable pipeline label
  difficulty_tier    – "easy" / "medium" / "hard"
  estimated_layout   – layout hint (single_column, form, receipt_strip, …)
  escalation_used    – filled in by the pipeline *after* OCR completes

Integration
-----------
    from app.services.ocr_difficulty_analyzer import classify_image, Tier

    dd = classify_image(path)
    # EASY: skip expensive steps for max throughput
    with use_profile(dd.profile):
        text, conf, meta = ocr_image(path, routing_decision=dd.to_routing_decision())
    dd.escalation_used = meta.get("crop_vlm_calls", 0) > 0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── tier enum ─────────────────────────────────────────────────────────────────


class Tier(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ── pipeline pipeline labels ───────────────────────────────────────────────────

_PIPELINE_LABELS: dict[Tier, str] = {
    Tier.EASY: "easy/rapid_only",
    Tier.MEDIUM: "medium/light_verify",
    Tier.HARD: "hard/full_pipeline",
}


# ── lazy profile factories ─────────────────────────────────────────────────────

def _easy_profile():
    """RapidOCR only — no Tesseract, no VLM, no segmentation."""
    from app.services.pipeline_profile import PipelineProfile
    return PipelineProfile(
        name="difficulty_easy",
        line_verification_enabled=False,
        tesseract_budget_per_page=0,
        skip_high_conf_tesseract=True,
        allow_crop_vlm=False,
        max_crop_vlm_calls=0,
        merge_adjacent_crops=True,
        skip_noise_lines=True,
        noise_min_chars=3,
        noise_max_mean_conf=0.55,
        layout_reconstruction_enabled=False,
        layout_detect_barcodes=False,
        layout_detect_forms=False,
        layout_detect_columns=False,
        layout_flag_handwriting=False,
        layout_dedup_enabled=True,
    )


def _medium_profile():
    """RapidOCR + lightweight layout + optional Tesseract (capped budget)."""
    from app.services.pipeline_profile import PipelineProfile
    return PipelineProfile(
        name="difficulty_medium",
        line_verification_enabled=True,
        tesseract_budget_per_page=4,
        skip_high_conf_tesseract=True,
        allow_crop_vlm=False,
        max_crop_vlm_calls=0,
        merge_adjacent_crops=True,
        skip_noise_lines=True,
        noise_min_chars=3,
        noise_max_mean_conf=0.55,
        layout_reconstruction_enabled=True,
        layout_detect_barcodes=True,
        layout_detect_forms=True,
        layout_detect_columns=True,
        layout_flag_handwriting=False,
        layout_dedup_enabled=True,
    )


def _hard_profile():
    """Full pipeline — segmentation, Tesseract, VLM, validators."""
    from app.services.pipeline_profile import PipelineProfile
    return PipelineProfile(
        name="difficulty_hard",
        line_verification_enabled=True,
        tesseract_budget_per_page=10,
        skip_high_conf_tesseract=False,
        allow_crop_vlm=True,
        max_crop_vlm_calls=6,
        merge_adjacent_crops=True,
        skip_noise_lines=False,
        noise_min_chars=3,
        noise_max_mean_conf=0.55,
        layout_reconstruction_enabled=True,
        layout_detect_barcodes=True,
        layout_detect_forms=True,
        layout_detect_columns=True,
        layout_flag_handwriting=True,
        layout_dedup_enabled=True,
    )


_PROFILE_FACTORIES = {
    Tier.EASY: _easy_profile,
    Tier.MEDIUM: _medium_profile,
    Tier.HARD: _hard_profile,
}


# ── difficulty decision ────────────────────────────────────────────────────────


@dataclass
class DifficultyDecision:
    """
    Output of the 3-tier difficulty classifier for one image or PDF page.

    Fields
    ------
    tier               EASY / MEDIUM / HARD
    confidence         classifier certainty in [0, 1] — higher = more certain
    reasons            human-readable list of contributing factors
    estimated_layout   layout type hint ("single_column", "form", …)
    recommended_pipeline  pipeline label string

    Benchmark
    ---------
    routing_time_ms    wall-clock analysis time (target: <20 ms)
    difficulty_tier    same as tier.value — convenience string
    selected_pipeline  same as recommended_pipeline — convenience string
    escalation_used    set to True by the pipeline *after* OCR if VLM was used
    """

    tier: Tier
    confidence: float
    reasons: list[str] = field(default_factory=list)
    estimated_layout: str = "unknown"
    recommended_pipeline: str = "easy/rapid_only"

    # ── benchmark metadata ─────────────────────────────────────────────────────
    routing_time_ms: float = 0.0
    difficulty_tier: str = field(init=False)
    selected_pipeline: str = field(init=False)
    escalation_used: bool = False

    # ── internal cached profile (not part of public API) ──────────────────────
    _profile: Any = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        self.difficulty_tier = self.tier.value
        self.selected_pipeline = self.recommended_pipeline

    # ── profile property ───────────────────────────────────────────────────────

    @property
    def profile(self):
        """Return the ``PipelineProfile`` that corresponds to this tier (lazily built)."""
        if self._profile is None:
            self._profile = _PROFILE_FACTORIES[self.tier]()
        return self._profile

    # ── routing decision bridge ────────────────────────────────────────────────

    def to_routing_decision(self):
        """
        Wrap this decision in a ``RoutingDecision`` compatible with
        ``ocr_image(routing_decision=…)``.

        This avoids re-running the difficulty analyzer inside ``ocr_image``
        when ``settings.adaptive_routing_enabled`` is True, by providing a
        pre-computed decision that is accepted directly.
        """
        from app.services.difficulty_analyzer import (
            DifficultyResult,
            DocumentType,
            LayoutType,
            OcrQuality,
        )
        from app.services.ocr_router import RoutingDecision, RoutingTier

        _tier_map: dict[Tier, RoutingTier] = {
            Tier.EASY: RoutingTier.FAST,
            Tier.MEDIUM: RoutingTier.VERIFY,
            Tier.HARD: RoutingTier.ESCALATE,
        }
        routing_tier = _tier_map[self.tier]

        try:
            layout_enum = LayoutType(self.estimated_layout)
        except ValueError:
            layout_enum = LayoutType.UNKNOWN

        synthetic = DifficultyResult(
            difficulty_score=_tier_to_midpoint_score(self.tier),
            document_type=DocumentType.UNKNOWN,
            layout_type=layout_enum,
            expected_ocr_quality=(
                OcrQuality.GOOD if self.tier == Tier.EASY else OcrQuality.FAIR
            ),
            reasons=list(self.reasons),
            analysis_ms=self.routing_time_ms,
        )

        allow_tess = (
            self.tier == Tier.HARD
            and settings.ocr_tesseract_polish_fullpage_fallback
        )
        engine_override = "fast" if self.tier == Tier.EASY else None

        return RoutingDecision(
            tier=routing_tier,
            profile=self.profile,
            difficulty=synthetic,
            engine_override=engine_override,
            allow_tesseract_fullpage=allow_tess,
            routing_ms=self.routing_time_ms,
            reasons=list(self.reasons),
        )

    # ── serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "difficulty_tier": self.tier.value,
            "confidence": round(self.confidence, 3),
            "estimated_layout": self.estimated_layout,
            "recommended_pipeline": self.recommended_pipeline,
            "selected_pipeline": self.selected_pipeline,
            "routing_time_ms": round(self.routing_time_ms, 2),
            "escalation_used": self.escalation_used,
            "reasons": self.reasons,
        }


# ── score → tier conversion ────────────────────────────────────────────────────


def _score_to_tier(score: float) -> tuple[Tier, float]:
    """
    Map a difficulty score in [0, 100] to (Tier, confidence).

    Thresholds are read from settings so they can be tuned per deployment:
      score < ocr_difficulty_easy_max   → EASY
      score < ocr_difficulty_hard_min   → MEDIUM
      score ≥ ocr_difficulty_hard_min   → HARD

    Confidence is a [0.5, 1.0] value that reflects how far the score is
    from the nearest tier boundary (farther = more certain).
    """
    easy_max: float = settings.ocr_difficulty_easy_max
    hard_min: float = settings.ocr_difficulty_hard_min

    if score < easy_max:
        gap = easy_max - score
        conf = min(1.0, 0.5 + 0.5 * (gap / max(easy_max, 1.0)))
        return Tier.EASY, round(conf, 3)

    if score < hard_min:
        mid = (easy_max + hard_min) / 2.0
        span = (hard_min - easy_max) / 2.0
        gap = abs(score - mid)
        conf = min(1.0, 0.5 + 0.5 * (gap / max(span, 1.0)))
        return Tier.MEDIUM, round(conf, 3)

    gap = score - hard_min
    remaining = 100.0 - hard_min
    conf = min(1.0, 0.5 + 0.5 * (gap / max(remaining, 1.0)))
    return Tier.HARD, round(conf, 3)


def _tier_to_midpoint_score(tier: Tier) -> float:
    """Return a representative score for a tier (used when building synthetic DifficultyResult)."""
    return {Tier.EASY: 15.0, Tier.MEDIUM: 48.0, Tier.HARD: 75.0}[tier]


# ── public entry points ────────────────────────────────────────────────────────


def classify_image(
    source,          # PIL.Image.Image | Path | str
    *,
    fast: bool = True,
) -> DifficultyDecision:
    """
    Classify a raster image as EASY / MEDIUM / HARD.

    Delegates feature extraction to ``difficulty_analyzer.analyze_image`` (the
    existing fast CPU-only module) and maps its 0–100 score to the 3-tier
    system.

    Parameters
    ----------
    source : PIL.Image.Image | Path | str
    fast   : bool
        Use fast sub-sampled analysis (default True; targets <20 ms).

    Returns
    -------
    DifficultyDecision
    """
    t0 = time.perf_counter()
    try:
        from app.services.difficulty_analyzer import analyze_image
        result = analyze_image(source, fast=fast)
    except Exception as exc:  # noqa: BLE001
        logger.debug("difficulty_analyzer.analyze_image failed: %s", exc)
        return _fallback(t0, "analyze_image_failed")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return _build(result, elapsed_ms)


def classify_pdf_page(
    pdf_path,        # Path | str
    page_index: int = 0,
    *,
    probe_dpi: int | None = None,
    fast: bool = True,
) -> DifficultyDecision:
    """
    Classify a single PDF page as EASY / MEDIUM / HARD.

    A digital PDF with an embedded text layer is always EASY regardless of
    visual complexity (same logic as ``ocr_router.route_pdf_page``).

    Parameters
    ----------
    pdf_path   : Path | str
    page_index : int       0-based page index (default 0)
    probe_dpi  : int|None  render DPI for the low-res probe
                           (defaults to ``settings.adaptive_probe_dpi``)
    fast       : bool      fast analysis mode (default True; targets <50 ms)

    Returns
    -------
    DifficultyDecision
    """
    t0 = time.perf_counter()
    dpi = probe_dpi if probe_dpi is not None else settings.adaptive_probe_dpi

    try:
        from app.services.difficulty_analyzer import analyze_pdf_page
        result = analyze_pdf_page(pdf_path, page_index, probe_dpi=dpi, fast=fast)
    except Exception as exc:  # noqa: BLE001
        logger.debug("difficulty_analyzer.analyze_pdf_page failed: %s", exc)
        return _fallback(t0, "analyze_pdf_page_failed")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Digital PDFs with an embedded text layer: no image OCR complexity → EASY
    pf = result.pdf_page_features
    if pf is not None and pf.has_embedded_text:
        reasons = list(result.reasons) + ["digital_pdf_text_layer→EASY"]
        return DifficultyDecision(
            tier=Tier.EASY,
            confidence=0.95,
            reasons=reasons,
            estimated_layout=result.layout_type.value,
            recommended_pipeline=_PIPELINE_LABELS[Tier.EASY],
            routing_time_ms=round(elapsed_ms, 2),
        )

    return _build(result, elapsed_ms)


def classify_zip_member(member) -> DifficultyDecision:
    """
    Classify a ``ZipMember`` as EASY / MEDIUM / HARD using fast image analysis.

    Falls back to MEDIUM on any error.

    Parameters
    ----------
    member : ZipMember

    Returns
    -------
    DifficultyDecision
    """
    t0 = time.perf_counter()
    try:
        from app.core.constants import SourceKind
        if member.kind == SourceKind.IMAGE:
            return classify_image(member.path, fast=True)
        if member.kind == SourceKind.PDF:
            return classify_pdf_page(member.path, page_index=0, fast=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "classify_zip_member failed for %s: %s",
            getattr(member, "path", "?"), exc,
        )
    return _fallback(t0, "classify_zip_member_failed")


# ── batch helpers ──────────────────────────────────────────────────────────────


def classify_images_batch(sources: list, *, fast: bool = True) -> list[DifficultyDecision]:
    """Classify a list of image paths/PIL images. Returns decisions in the same order."""
    return [classify_image(src, fast=fast) for src in sources]


# ── private helpers ────────────────────────────────────────────────────────────


def _build(result, elapsed_ms: float) -> DifficultyDecision:
    """Translate a ``DifficultyResult`` from ``difficulty_analyzer`` into ``DifficultyDecision``."""
    tier, confidence = _score_to_tier(result.difficulty_score)
    reasons = list(result.reasons)
    reasons.insert(0, f"score={result.difficulty_score:.1f}→{tier.value}")
    return DifficultyDecision(
        tier=tier,
        confidence=confidence,
        reasons=reasons,
        estimated_layout=result.layout_type.value,
        recommended_pipeline=_PIPELINE_LABELS[tier],
        routing_time_ms=round(elapsed_ms, 2),
    )


def _fallback(t0: float, reason: str) -> DifficultyDecision:
    """Safe MEDIUM decision returned when analysis fails."""
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return DifficultyDecision(
        tier=Tier.MEDIUM,
        confidence=0.5,
        reasons=[f"{reason}→fallback_medium"],
        estimated_layout="unknown",
        recommended_pipeline=_PIPELINE_LABELS[Tier.MEDIUM],
        routing_time_ms=round(elapsed_ms, 2),
    )
