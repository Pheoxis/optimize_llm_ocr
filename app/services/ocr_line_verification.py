"""Line-level RapidOCR vs Tesseract disagreement verification (no full-page rerun)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import pytesseract
from PIL import Image

from app.core.config import settings
from app.models.schemas import OcrLine
from app.services.ocr_quality import classify_line_escalation, line_quality_score, normalized_levenshtein
from app.services.semantic_validators.registry import semantic_validation_failed, validate_text
from app.services.semantic_validators.types import SemanticValidationResult

logger = logging.getLogger(__name__)


@dataclass
class LineVerificationResult:
    """One visual line after Rapid + Tesseract crop verification."""

    line_index: int
    rapid_text: str
    tesseract_text: str
    disagreement: float
    suspicious: bool
    tess_skipped: bool
    rapid_confidence: float
    rapid_min_confidence: float
    bbox: tuple[int, int, int, int]
    quality_score: float = 0.0
    escalation_hint: str | None = None
    escalation_tier: str = "none"
    validation: SemanticValidationResult | None = None
    validation_passed: bool = True

    @property
    def output_text(self) -> str:
        """Final line text after local verification and Polish cleanup."""
        from app.services.ocr_postprocess import polish_line_output

        return polish_line_output(self.rapid_text, self.tesseract_text)

    def to_ocr_line(
        self,
        *,
        page: int = 1,
        engine: str = "fast_ocr",
        document: int | None = None,
        final_text: str | None = None,
        final_engine: str | None = None,
        escalated: bool | None = None,
    ) -> OcrLine:
        text = final_text if final_text is not None else self.output_text
        return OcrLine(
            text=text,
            confidence=round(self.rapid_confidence, 4),
            bbox=list(self.bbox),
            page=page,
            engine=final_engine or engine,
            document=document,
            line_index=self.line_index,
            escalated=escalated,
            suspicious=self.suspicious,
            disagreement=round(self.disagreement, 4),
            rapid_text=self.rapid_text,
            tesseract_text=self.tesseract_text or None,
            verification_skipped=self.tess_skipped,
            escalation_hint=self.escalation_hint,
            validation_passed=self.validation_passed,
            validation_issues=(
                [i.to_dict() for i in self.validation.issues if not i.valid]
                if self.validation
                else None
            ),
            validation_failed_kinds=(
                self.validation.failed_kinds if self.validation else None
            ),
        )


@dataclass
class LineVerificationMetrics:
    """Aggregated metrics for one page / image verification pass."""

    total_lines: int = 0
    tess_ran: int = 0
    tess_skipped: int = 0
    suspicious_count: int = 0
    validation_fail_count: int = 0
    mean_disagreement: float = 0.0
    max_disagreement: float = 0.0
    suspicious_threshold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _crop_line(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    pad: int,
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    w, h = image.size
    return image.crop((
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(w, x1 + pad),
        min(h, y1 + pad),
    ))


def tesseract_line_crop(
    crop: Image.Image,
    lang: str | None = None,
    *,
    rapid_text: str | None = None,
) -> str:
    """Tesseract on a single line crop (PSM 7 = single text line)."""
    if lang is None:
        from app.services.ocr_lang_routing import resolve_tesseract_lang

        lang = resolve_tesseract_lang(rapid_text)
    raw = pytesseract.image_to_string(
        crop.convert("RGB"),
        lang=lang,
        config="--psm 7 --oem 3",
    )
    return " ".join(raw.split())


def _should_skip_tesseract(mean_conf: float, rapid_text: str = "") -> bool:
    if not settings.ocr_line_verification_skip_high_conf_tesseract:
        return False
    from app.services.ocr_postprocess import polish_ocr_needs_verification

    if polish_ocr_needs_verification(rapid_text):
        return False
    return mean_conf >= settings.ocr_line_skip_tesseract_above


def verify_line_crop(
    crop: Image.Image,
    rapid_line: Any,
    lang: str | None = None,
    *,
    line_index: int = 0,
    force_skip_tesseract: bool | None = None,
) -> LineVerificationResult:
    """
    Verify one RapidOCR line against Tesseract on the same crop.

    Does not re-run RapidOCR or process the full page.
    """
    rapid_text = (getattr(rapid_line, "text", "") or "").strip()
    mean_conf = float(getattr(rapid_line, "mean_score", 0.0))
    min_conf = float(getattr(rapid_line, "min_score", 0.0))
    bbox = tuple(getattr(rapid_line, "bbox", (0, 0, 0, 0)))

    threshold = settings.ocr_line_suspicious_threshold
    tess_skipped = (
        force_skip_tesseract
        if force_skip_tesseract is not None
        else _should_skip_tesseract(mean_conf, rapid_text)
    )
    tess_text = ""

    if not tess_skipped:
        try:
            tess_text = tesseract_line_crop(crop, lang, rapid_text=rapid_text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tesseract line verify failed idx=%s: %s", line_index, exc)
            tess_text = ""

    if tess_skipped or not tess_text.strip():
        disagreement = 0.0
        suspicious = False
        q_score = line_quality_score(
            mean_conf=mean_conf, rapid_text=rapid_text, tess_text="",
        )
        tier, reason = classify_line_escalation(
            mean_conf=mean_conf,
            min_conf=min_conf,
            rapid_text=rapid_text,
            tess_text="",
            quality_score=q_score,
        )
    else:
        disagreement = normalized_levenshtein(rapid_text, tess_text)
        suspicious = disagreement > threshold
        q_score = line_quality_score(
            mean_conf=mean_conf, rapid_text=rapid_text, tess_text=tess_text,
        )
        tier, reason = classify_line_escalation(
            mean_conf=mean_conf,
            min_conf=min_conf,
            rapid_text=rapid_text,
            tess_text=tess_text,
            quality_score=q_score,
        )

    validation = validate_text(rapid_text)
    validation_passed = validation.passed
    sem_failed, sem_reason = semantic_validation_failed(rapid_text)
    if sem_failed and settings.ocr_semantic_validation_escalate:
        tier = "hard"
        reason = sem_reason
        suspicious = True
        validation_passed = False

    escalation_hint: str | None = None
    if suspicious and sem_failed:
        escalation_hint = sem_reason or "semantic_validation_failed"
    elif suspicious:
        escalation_hint = "suspicious_disagreement"
    elif tier != "none" and reason:
        escalation_hint = reason

    if suspicious:
        logger.info(
            "OCR line %d suspicious: disagreement=%.3f > %.3f rapid=%r tess=%r",
            line_index,
            disagreement,
            threshold,
            rapid_text[:60],
            tess_text[:60],
        )

    return LineVerificationResult(
        line_index=line_index,
        rapid_text=rapid_text,
        tesseract_text=tess_text,
        disagreement=round(disagreement, 4),
        suspicious=suspicious,
        tess_skipped=tess_skipped,
        rapid_confidence=mean_conf,
        rapid_min_confidence=min_conf,
        bbox=bbox,
        quality_score=round(q_score, 4),
        escalation_hint=escalation_hint,
        escalation_tier=tier,
        validation=validation,
        validation_passed=validation_passed,
    )


def verify_fast_result_lines(
    image: Image.Image,
    fast: Any,
    lang: str | None = None,
    *,
    page: int = 1,
) -> tuple[list[LineVerificationResult], LineVerificationMetrics]:
    """Verify every RapidOCR line on *image* (line crops only)."""
    raw_lines = getattr(fast, "lines", None) or []
    pad = settings.ocr_line_crop_padding_px
    rgb = image.convert("RGB")

    results: list[LineVerificationResult] = []
    disagreements: list[float] = []

    for idx, line in enumerate(raw_lines):
        crop = _crop_line(rgb, line.bbox, pad) if any(line.bbox) else rgb
        verified = verify_line_crop(crop, line, lang, line_index=idx)
        results.append(verified)
        if not verified.tess_skipped:
            disagreements.append(verified.disagreement)

    suspicious_n = sum(1 for r in results if r.suspicious)
    validation_fail_n = sum(1 for r in results if not r.validation_passed)
    tess_ran = sum(1 for r in results if not r.tess_skipped)
    metrics = LineVerificationMetrics(
        total_lines=len(results),
        tess_ran=tess_ran,
        tess_skipped=len(results) - tess_ran,
        suspicious_count=suspicious_n,
        validation_fail_count=validation_fail_n,
        mean_disagreement=(
            sum(disagreements) / len(disagreements) if disagreements else 0.0
        ),
        max_disagreement=max(disagreements) if disagreements else 0.0,
        suspicious_threshold=settings.ocr_line_suspicious_threshold,
    )
    return results, metrics


def verified_lines_to_dicts(
    results: list[LineVerificationResult],
    *,
    page: int = 1,
    engine: str = "fast_ocr",
    document: int | None = None,
) -> list[dict[str, Any]]:
    return [
        r.to_ocr_line(page=page, engine=engine, document=document).model_dump()
        for r in results
    ]


def merge_verified_text(results: list[LineVerificationResult]) -> str:
    return "\n".join(r.output_text for r in results if r.output_text)


def log_verification_metrics(
    metrics: LineVerificationMetrics,
    *,
    label: str = "",
) -> None:
    prefix = f"{label}: " if label else ""
    logger.info(
        "%sline verification — lines=%d tess_ran=%d suspicious=%d "
        "validation_fail=%d mean_disagreement=%.3f max=%.3f threshold=%.3f",
        prefix,
        metrics.total_lines,
        metrics.tess_ran,
        metrics.suspicious_count,
        metrics.validation_fail_count,
        metrics.mean_disagreement,
        metrics.max_disagreement,
        metrics.suspicious_threshold,
    )


def attach_verification_to_meta(
    meta: dict[str, Any],
    results: list[LineVerificationResult],
    metrics: LineVerificationMetrics,
    *,
    page: int = 1,
    engine: str = "fast_ocr",
    document: int | None = None,
) -> dict[str, Any]:
    meta["lines"] = verified_lines_to_dicts(
        results, page=page, engine=engine, document=document,
    )
    meta["n_lines"] = len(results)
    meta["line_verification"] = metrics.to_dict()
    meta["suspicious_line_count"] = metrics.suspicious_count
    meta["semantic_validation_fail_count"] = metrics.validation_fail_count
    return meta
