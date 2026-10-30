"""
Lightweight PL/EN Tesseract language routing (verification layer only).

RapidOCR stays on PP-OCRv5 Latin (handles Polish + English in one model).
Tesseract lang packs are selected per line/document to avoid loading unused languages.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)

# Polish letters outside basic Latin-1 (plus common OCR confusions kept out).
_POLISH_CHAR_RE = re.compile(r"[\u0104\u0105\u0106\u0107\u0118\u0119\u0141\u0142\u0143\u0144\u00d3\u00f3\u015a\u015b\u0179\u017a\u017b\u017c\u017d]")

_metrics_lock = threading.Lock()


@dataclass
class OcrLangRoutingMetrics:
    polish_routed: int = 0
    english_routed: int = 0
    samples: int = 0

    def to_dict(self) -> dict[str, int]:
        total = max(self.samples, 1)
        return {
            "samples": self.samples,
            "polish_routed": self.polish_routed,
            "english_routed": self.english_routed,
            "polish_ratio": round(self.polish_routed / total, 4),
        }


_metrics = OcrLangRoutingMetrics()


def get_ocr_lang_routing_metrics() -> OcrLangRoutingMetrics:
    with _metrics_lock:
        return OcrLangRoutingMetrics(
            polish_routed=_metrics.polish_routed,
            english_routed=_metrics.english_routed,
            samples=_metrics.samples,
        )


def reset_ocr_lang_routing_metrics() -> None:
    global _metrics
    with _metrics_lock:
        _metrics = OcrLangRoutingMetrics()


def log_ocr_lang_routing_metrics(*, label: str = "") -> None:
    m = get_ocr_lang_routing_metrics()
    prefix = f"{label}: " if label else ""
    logger.info(
        "%sOCR lang routing — samples=%d pol=%d eng=%d polish_ratio=%.2f",
        prefix,
        m.samples,
        m.polish_routed,
        m.english_routed,
        m.polish_routed / max(m.samples, 1),
    )


def polish_chars_detected(text: str) -> bool:
    """True when sample contains enough Polish-specific letters."""
    if not text or not text.strip():
        return False
    matches = _POLISH_CHAR_RE.findall(text)
    return len(matches) >= settings.ocr_lang_polish_char_min


def resolve_tesseract_lang(
    ocr_sample: str | None = None,
    *,
    explicit: str | None = None,
) -> str:
    """
    Tesseract ``lang`` string for verification (not primary OCR).

    * Polish likely → ``pol+eng`` (configurable)
    * Otherwise → ``eng`` only (faster, less confusion)
    * Per-job hint ``pl`` / ``en`` overrides auto-detection (see ``document_language``)
    """
    if explicit is not None:
        return explicit

    from app.services.document_language import (
        DocumentLanguage,
        effective_primary_ocr_lang,
        get_document_language,
    )

    hint = get_document_language()
    if hint == DocumentLanguage.PL:
        return settings.ocr_lang_tesseract_polish
    if hint == DocumentLanguage.EN:
        return settings.ocr_lang_tesseract_english

    if not settings.ocr_lang_routing_enabled:
        return settings.ocr_lang

    sample = ocr_sample or ""
    if not sample.strip():
        return effective_primary_ocr_lang()

    use_polish = polish_chars_detected(sample)
    if use_polish:
        lang = settings.ocr_lang_tesseract_polish
        route = "pol+eng"
    else:
        lang = settings.ocr_lang_tesseract_english
        route = "eng"

    with _metrics_lock:
        _metrics.samples += 1
        if use_polish:
            _metrics.polish_routed += 1
        else:
            _metrics.english_routed += 1

    logger.debug(
        "Tesseract lang routed → %s (%s, sample_len=%d)",
        lang,
        route,
        len(ocr_sample or ""),
    )
    return lang


def languages_detected_from_text(text: str) -> list[str]:
    """API ``language_detected`` field — PL+EN vs EN only."""
    if polish_chars_detected(text):
        return ["pl", "en"]
    return ["en"]


def effective_ocr_lang_config() -> str:
    """Static fallback when no OCR sample is available yet."""
    from app.services.document_language import effective_primary_ocr_lang

    return effective_primary_ocr_lang()
