"""
Line-level smart OCR: RapidOCR + line scoring + crop-only vision escalation.

Delegates to ``crop_escalation`` (no full-page vision fallback).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.core.config import settings
from app.services.crop_escalation import apply_crop_escalation
from app.services.fast_ocr import ocr_image_fast

logger = logging.getLogger(__name__)


@dataclass
class LineDebugMeta:
    text: str
    confidence: float
    engine: str
    bbox: tuple[int, int, int, int]
    quality_score: float
    escalated: bool
    disagreement_score: float
    suspicious: bool = False
    rapid_text: str = ""
    tesseract_text: str = ""
    tess_skipped: bool = False
    escalation_hint: str | None = None
    preprocessing_variant: str | None = None
    escalation_reason: str | None = None


@dataclass
class LineSmartResult:
    text: str
    mean_confidence: float | None
    lines_debug: list[LineDebugMeta] = field(default_factory=list)
    engines_used: set[str] = field(default_factory=set)
    page_vlm_used: bool = False
    elapsed_sec: float = 0.0
    verification_metrics: object | None = None


def ocr_image_line_smart(
    image_path: Path,
    lang: str | None = None,
    *,
    allow_vision_escalation: bool | None = None,
    gpu_id: int | None = None,
) -> tuple[str, float | None, dict]:
    """
    Line-level OCR pipeline. Returns (text, confidence, metadata) like ``ocr_image``.
    """
    if allow_vision_escalation is None:
        allow_vision_escalation = True

    t0 = time.time()
    fast = ocr_image_fast(image_path, gpu_id=gpu_id)

    with Image.open(image_path) as page:
        page.load()
        rgb = page.convert("RGB")

    crop_result = apply_crop_escalation(
        rgb,
        fast,
        lang,
        allow_vlm=allow_vision_escalation,
        gpu_id=gpu_id,
        page_quality_poor=False,
    )

    debug_lines = [
        LineDebugMeta(
            text=r.text,
            confidence=r.confidence,
            engine=r.engine,
            bbox=r.bbox,
            quality_score=r.quality_score,
            escalated=r.escalated,
            disagreement_score=r.disagreement_score,
            suspicious=r.suspicious,
            rapid_text=r.rapid_text,
            tesseract_text=r.tesseract_text,
            tess_skipped=r.tess_skipped,
            escalation_hint=r.escalation_hint,
            preprocessing_variant=r.preprocessing_variant,
            escalation_reason=r.escalation_reason,
        )
        for r in crop_result.line_results
    ]

    meta = {
        **crop_result.meta,
        "engine": "line_smart_ocr",
        "pipeline": "line_level",
        "page_vlm_used": False,
        "elapsed_sec": round(time.time() - t0, 3),
        "gpu_id": gpu_id,
    }
    return crop_result.text, crop_result.mean_confidence, meta
