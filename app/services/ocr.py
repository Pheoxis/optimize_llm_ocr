"""OCR per page / image.

Three engines available:
  • fast_ocr    — RapidOCR PP-OCRv5 Latin on GPU.  ~0.25 s/img, 95%+ accuracy.
  • vision_llm  — Ollama crop OCR (minicpm-v / qwen2.5vl:7b).  Escalation only.
  • tesseract   — classic, ~5 s/img, decent on clean scans.

Routing is controlled by `settings.ocr_engine`:
  • "smart"     — fast_ocr → escalate to vision_llm if quality gate fails  (RECOMMENDED)
  • "fast"      — fast_ocr only, accept any result
  • "vision"    — always vision_llm
  • "tesseract" — always tesseract
  • "auto"      — photos → vision_llm, clean PDF scans → tesseract (legacy)
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import pytesseract
from PIL import Image

from app.core.config import settings
from app.services.preprocess import is_photo, is_receipt_like, prepare_for_ocr
from app.services.vision_llm import VisionLLMError, ocr_crop_vision, ocr_image_vision

logger = logging.getLogger(__name__)


def _load_rgb_image(
    image: Image.Image | None,
    image_path: Path | None,
) -> Image.Image:
    if image is not None:
        return image.convert("RGB")
    if image_path is None:
        raise ValueError("image or image_path required")
    with Image.open(image_path) as img:
        img.load()
        return img.convert("RGB")


def _crop_escalate_fast_result(
    image: Image.Image | None,
    image_path: Path | None,
    fast: object,
    lang: str | None,
    *,
    reason: str,
    allow_vision_escalation: bool,
    gpu_id: int | None,
    pipeline: str,
    page_quality_poor: bool = True,
    extra_meta: dict | None = None,
) -> tuple[str, float | None, dict]:
    """Replace full-page VLM: score lines and escalate only bad crops."""
    from app.services.crop_escalation import apply_crop_escalation

    rgb = _load_rgb_image(image, image_path)
    result = apply_crop_escalation(
        rgb,
        fast,
        lang,
        allow_vlm=allow_vision_escalation,
        gpu_id=gpu_id,
        page_quality_poor=page_quality_poor,
        escalation_reason=reason,
    )
    meta: dict = {
        **result.meta,
        "fallback_from": "fast_ocr",
        "escalation_reason": reason,
        "pipeline": pipeline,
        "escalated": result.crop_vlm_calls > 0,
        "ocr_passes": 2,
        "fast_ocr_meta": {
            "mean_score": round(getattr(fast, "mean_score", 0.0), 4),
            "n_boxes": getattr(fast, "n_boxes", 0),
        },
    }
    if extra_meta:
        meta.update(extra_meta)
    return result.text, result.mean_confidence, meta


def _vision_escalation_allowed() -> bool:
    return settings.ocr_page_vision_escalation_enabled


def _single_crop_vision_fallback(
    image: Image.Image | None,
    image_path: Path | None,
    *,
    reason: str,
    pipeline: str,
) -> tuple[str, float | None, dict]:
    """Last resort: one crop-sized VLM call on the full frame (not legacy full-page path)."""
    rgb = _load_rgb_image(image, image_path)
    text, _, vmeta = ocr_crop_vision(rgb)
    return text, None, {
        **vmeta,
        "engine": "vision_llm",
        "pipeline": pipeline,
        "escalation_reason": reason,
        "crop_escalation": True,
        "page_vision_escalation": False,
    }


# ── public API ────────────────────────────────────────────────────────────────

def ocr_image(
    image_path: Path,
    lang: str | None = None,
    *,
    allow_vision_escalation: bool | None = None,
    gpu_id: int | None = None,
    line_level: bool | None = None,
    routing_decision=None,   # RoutingDecision | None — pre-computed by caller
) -> tuple[str, float | None, dict]:
    """
    Return (text, confidence, metadata).

    Metadata always contains 'engine' so the caller can record what produced
    each result.  In smart mode the metadata may also contain a
    'fallback_from' and an 'escalation_reason' field.

    ``allow_vision_escalation`` overrides smart-mode escalation (e.g. PDF
    scans pass ``False`` so RapidOCR results are kept — faster and more
    complete on multi-receipt pages).

    ``line_level``: ``True`` = line-smart (zdjęcia); ``False`` = szybki RapidOCR
    na cropach PDF z eskalacją line-smart tylko przy słabej jakości.

    ``routing_decision``: pre-computed RoutingDecision from ``route_image()``.
    When None and ``settings.adaptive_routing_enabled`` is True, the router
    is invoked automatically.  Pass an existing decision to avoid re-analysis.
    """
    # ── adaptive routing ──────────────────────────────────────────────────
    if settings.adaptive_routing_enabled:
        decision = routing_decision or _compute_routing(image_path)
        if decision is not None:
            return _ocr_image_with_routing(
                image_path, lang,
                decision=decision,
                allow_vision_escalation=allow_vision_escalation,
                gpu_id=gpu_id,
                line_level=line_level,
            )

    # ── static pipeline (fallback / routing disabled) ─────────────────────
    return _ocr_image_static(
        image_path, lang,
        allow_vision_escalation=allow_vision_escalation,
        gpu_id=gpu_id,
        line_level=line_level,
    )


def _compute_routing(image_path: Path):
    """Run the difficulty analyzer + router. Returns None on failure."""
    try:
        from app.services.ocr_router import route_image
        return route_image(image_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR router failed for %s, falling back to static pipeline: %s", image_path, exc)
        return None


def _ocr_image_with_routing(
    image_path: Path,
    lang: str | None,
    *,
    decision,
    allow_vision_escalation: bool | None,
    gpu_id: int | None,
    line_level: bool | None,
) -> tuple[str, float | None, dict]:
    """Execute OCR under the profile chosen by the difficulty router."""
    from app.services.pipeline_profile import use_profile

    engine_choice = decision.engine_override or settings.ocr_engine

    with use_profile(decision.profile):
        text, conf, meta = _dispatch_engine(
            engine_choice,
            image_path, lang,
            allow_vision_escalation=allow_vision_escalation,
            gpu_id=gpu_id,
            line_level=line_level,
        )

    meta["adaptive_routing"] = decision.to_dict()
    return text, conf, meta


def _ocr_image_static(
    image_path: Path,
    lang: str | None,
    *,
    allow_vision_escalation: bool | None,
    gpu_id: int | None,
    line_level: bool | None,
) -> tuple[str, float | None, dict]:
    """Original static pipeline — unchanged from before adaptive routing."""
    return _dispatch_engine(
        settings.ocr_engine,
        image_path, lang,
        allow_vision_escalation=allow_vision_escalation,
        gpu_id=gpu_id,
        line_level=line_level,
    )


def _dispatch_engine(
    engine_choice: str,
    image_path: Path,
    lang: str | None,
    *,
    allow_vision_escalation: bool | None,
    gpu_id: int | None,
    line_level: bool | None,
) -> tuple[str, float | None, dict]:
    """Route to the correct engine implementation."""
    if engine_choice == "smart":
        return _smart_pipeline(
            image_path, lang,
            allow_vision_escalation=allow_vision_escalation,
            gpu_id=gpu_id,
            line_level=line_level,
        )

    if engine_choice == "fast":
        return _fast_only(image_path, lang, gpu_id=gpu_id)

    if engine_choice == "vision":
        return _vision_with_fallback(image_path, lang)

    if engine_choice == "tesseract":
        text, conf = _tesseract_ocr(image_path, lang)
        return text, conf, {"engine": "tesseract"}

    # "auto" (legacy)
    with Image.open(image_path) as img:
        use_vision = is_photo(img.convert("RGB"))
    if use_vision:
        return _vision_with_fallback(image_path, lang)
    text, conf = _tesseract_ocr(image_path, lang)
    return text, conf, {"engine": "tesseract"}


# ── smart pipeline (fast OCR + quality-gated vision fallback) ────────────────

def _should_escalate_pdf_crop_to_line_smart(fast) -> tuple[bool, str | None]:
    """Stricter than page-VLM gate — only truly bad crops get line-smart (keeps SKMBT ~11s)."""
    if not fast.text.strip():
        return True, "empty_crop"
    if fast.n_boxes < settings.pdf_crop_line_escalate_min_boxes:
        return True, f"too_few_boxes ({fast.n_boxes})"
    if fast.mean_score < settings.pdf_crop_line_escalate_mean_conf:
        return True, (
            f"mean_conf_low ({fast.mean_score:.3f} < "
            f"{settings.pdf_crop_line_escalate_mean_conf})"
        )
    if fast.min_score < settings.pdf_crop_line_escalate_min_conf:
        return True, (
            f"min_conf_low ({fast.min_score:.3f} < "
            f"{settings.pdf_crop_line_escalate_min_conf})"
        )
    return False, None


def _use_line_smart(line_level: bool | None) -> bool:
    if line_level is not None:
        return line_level
    return settings.ocr_line_level_enabled


def _fast_ocr_meta_from_result(
    fast,
    *,
    gpu_id: int | None,
    pipeline: str,
    page: int = 1,
    document: int | None = None,
    attach_lines: bool = True,
) -> dict:
    meta = {
        "engine": "fast_ocr",
        "model": f"PP-OCRv5/{settings.fast_lang_type}/{settings.fast_model_type}",
        "elapsed_sec": round(fast.elapsed_sec, 3),
        "mean_score": round(fast.mean_score, 4),
        "min_score": round(fast.min_score, 4),
        "low_conf_ratio": round(fast.low_conf_ratio, 4),
        "n_boxes": fast.n_boxes,
        "gpu_id": gpu_id,
        "pipeline": pipeline,
    }
    # Attach block-aware reading order debug metadata when present
    layout_meta = getattr(fast, "layout_meta", None) or {}
    if layout_meta:
        meta["reading_order"] = layout_meta
    if attach_lines:
        from app.services.ocr_lines import attach_fast_lines_to_meta

        attach_fast_lines_to_meta(
            meta, fast, page=page, engine="fast_ocr", document=document,
        )
    return meta


def _finalize_fast_ocr_output(
    image: Image.Image | None,
    image_path: Path | None,
    fast: object,
    lang: str | None,
    meta: dict,
    *,
    page: int = 1,
) -> tuple[str, dict]:
    """
    Attach line objects and optionally run per-line Tesseract verification.

    Never re-runs RapidOCR on the full page — only line crops when verifying.
    """
    lines = getattr(fast, "lines", None) or []

    if settings.ocr_line_verification_enabled and lines:
        from app.services.ocr_line_verification import (
            attach_verification_to_meta,
            log_verification_metrics,
            merge_verified_text,
            verify_fast_result_lines,
        )

        rgb: Image.Image | None = image.convert("RGB") if image is not None else None
        if rgb is None and image_path is not None:
            with Image.open(image_path) as img:
                img.load()
                rgb = img.convert("RGB")
        if rgb is not None:
            results, metrics = verify_fast_result_lines(rgb, fast, page=page)
            log_verification_metrics(metrics, label=str(meta.get("pipeline", "")))
            attach_verification_to_meta(meta, results, metrics, page=page)
            text_out = merge_verified_text(results)
            from app.services.ocr_quality_pipeline import attach_quality_to_meta
            attach_quality_to_meta(meta, text_out, lines=lines)
            return text_out, meta

    from app.services.ocr_lines import attach_fast_lines_to_meta

    attach_fast_lines_to_meta(
        meta, fast, page=page, engine=str(meta.get("engine", "fast_ocr")),
    )
    text_out = getattr(fast, "text", "").strip()

    from app.services.ocr_quality_pipeline import attach_quality_to_meta
    attach_quality_to_meta(meta, text_out, lines=lines)

    return text_out, meta


def _save_pil_temp(image: Image.Image) -> Path:
    """Write PIL to a temp PNG for engines that only accept paths (line-smart / VLM)."""
    fd, raw = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    path = Path(raw)
    image.convert("RGB").save(path, format="PNG")
    return path


_PL_DIAC_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def _polish_quality_score(text: str) -> float:
    """
    Rough quality score for Polish OCR output: higher is better.

    Combines:
      - share of Polish diacritics among alphabetic characters,
      - share of "clean" tokens (mostly alpha) on each line,
      - total useful character count (capped, to avoid massive noise winning).
    Used to compare RapidOCR vs Tesseract pol+eng without trusting their
    internal scores (different scales and miscalibrated for scanned PL).
    """
    if not text:
        return 0.0
    letters = sum(1 for c in text if c.isalpha())
    if letters < 50:
        return 0.0
    diac = sum(1 for c in text if c in _PL_DIAC_CHARS)
    diac_ratio = diac / letters
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    clean = 0
    total_lines = max(1, len(lines))
    for ln in lines:
        tokens = [t for t in ln.split() if t]
        if not tokens:
            continue
        good_tokens = 0
        for t in tokens:
            alpha = sum(1 for c in t if c.isalpha())
            if alpha >= max(2, len(t) - 2):
                good_tokens += 1
        if good_tokens / len(tokens) >= 0.6:
            clean += 1
    clean_ratio = clean / total_lines
    char_bonus = min(1.0, letters / 1500.0)
    return diac_ratio * 1.2 + clean_ratio * 1.0 + char_bonus * 0.3


def _prepare_for_tesseract_polish(img: "Image.Image") -> "Image.Image":
    """Upscale + binarise so Tesseract pol+eng hits peak accuracy.

    Tesseract accuracy peaks at 300 DPI.  Scanned invoices are often at
    150–250 DPI; upscaling them 1.5× before Tesseract significantly reduces
    diacritic recognition errors (ł→l, ę→e, etc.).
    Adaptive Gaussian binarisation removes slight yellowing and unevenness
    that causes Tesseract to confuse 8/B, O/0, I/l, ł/t.
    """
    import numpy as np
    import cv2

    rgb = img.convert("RGB")
    w, h = rgb.size
    # Upscale if smaller than ~300 DPI equivalent (A4 ≈ 2480 px wide at 300 DPI)
    target_w = max(w, 1600)  # ensure at least 1600 px wide for decent DPI
    if w < target_w:
        scale = target_w / w
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Convert to grayscale, apply adaptive binarisation
    gray = np.array(rgb.convert("L"))
    # Adaptive Gaussian threshold: good for uneven illumination in scanned docs
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    from PIL import Image as PILImage
    return PILImage.fromarray(binary).convert("RGB")


def _tesseract_polish_fullpage(image_path: Path) -> tuple[str, float | None]:
    """Tesseract pol+eng with enhanced preprocessing for Polish invoice scans."""
    try:
        with Image.open(image_path) as img:
            prepared = _prepare_for_tesseract_polish(img)
            # psm 6 = uniform block of text (better for single-column crops);
            # oem 3 = LSTM engine (highest accuracy for Polish diacritics).
            text = pytesseract.image_to_string(
                prepared, lang="pol+eng", config="--psm 6 --oem 3",
            )
            text = _clean_text(text)
            conf = _mean_confidence(prepared, "pol+eng")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tesseract pol+eng fallback failed: %s", exc)
        return "", None
    return text.strip(), conf
def _maybe_tesseract_polish_fallback(
    image_path: Path | None,
    image: Image.Image | None,
    fast_text: str,
    fast_meta: dict,
) -> tuple[str, dict, bool]:
    """
    If RapidOCR result looks like Polish OCR garble, try Tesseract pol+eng
    on the same page and keep whichever output scores higher on Polish quality.
    Returns ``(final_text, meta, used_tesseract)``.
    """
    from app.services.fast_ocr import _looks_polish_garbled

    if not settings.ocr_tesseract_polish_fullpage_fallback:
        return fast_text, fast_meta, False
    if not _looks_polish_garbled(fast_text):
        return fast_text, fast_meta, False

    target_path = image_path
    cleanup = False
    if target_path is None and image is not None:
        target_path = _save_pil_temp(image)
        cleanup = True
    if target_path is None:
        return fast_text, fast_meta, False

    try:
        tess_text, tess_conf = _tesseract_polish_fullpage(target_path)
    finally:
        if cleanup:
            target_path.unlink(missing_ok=True)

    if not tess_text:
        return fast_text, fast_meta, False

    fast_q = _polish_quality_score(fast_text)
    tess_q = _polish_quality_score(tess_text)
    fast_meta = {
        **fast_meta,
        "polish_fallback_evaluated": True,
        "polish_quality_fast": round(fast_q, 3),
        "polish_quality_tesseract": round(tess_q, 3),
    }
    if tess_q >= fast_q * settings.ocr_tesseract_polish_fallback_min_ratio:
        logger.info(
            "Polish fullpage fallback adopted Tesseract pol+eng "
            "(fast_q=%.2f tess_q=%.2f)", fast_q, tess_q,
        )
        fast_meta = {
            **fast_meta,
            "engine": "tesseract",
            "polish_fallback_used": True,
            "polish_fallback_lang": "pol+eng",
            "tesseract_polish_confidence": tess_conf,
        }
        return tess_text, fast_meta, True
    return fast_text, fast_meta, False


def _smart_pdf_crop_pipeline_pil(
    image: Image.Image,
    lang: str | None,
    *,
    allow_vision_escalation: bool | None = None,
    gpu_id: int | None = None,
) -> tuple[str, float | None, dict]:
    """
    In-memory variant of ``_smart_pdf_crop_pipeline`` (zip_simple_receipt path).
    """
    from app.services.fast_ocr import ocr_pil_fast, should_escalate

    if allow_vision_escalation is None:
        allow_vision_escalation = False

    try:
        fast = ocr_pil_fast(image, gpu_id=gpu_id)
    except Exception as exc:  # noqa: BLE001
        if not allow_vision_escalation:
            logger.warning("fast_ocr failed (%s) on PIL page — no vision escalation", exc)
            return "", None, {"engine": "fast_ocr", "error": str(exc), "gpu_id": gpu_id}
        logger.warning("fast_ocr failed (%s) — single-crop vision fallback", exc)
        return _single_crop_vision_fallback(
            image, None, reason="fast_ocr_error", pipeline="pdf_crop_crop_vision",
        )

    fast_meta = _fast_ocr_meta_from_result(
        fast, gpu_id=gpu_id, pipeline="pdf_crop_fast", page=1, attach_lines=False,
    )
    empty_detection = fast.n_boxes == 0
    ocr_passes = 1
    escalated = False

    line_escalate = False
    line_reason: str | None = None
    if settings.ocr_line_level_escalate_bad_pdf_crops:
        line_escalate, line_reason = _should_escalate_pdf_crop_to_line_smart(fast)

    if line_escalate:
        from app.services.line_smart_ocr import ocr_image_line_smart

        logger.info("PDF crop (PIL) → line-smart (%s)", line_reason or "empty")
        tmp = _save_pil_temp(image)
        try:
            text, conf, meta = ocr_image_line_smart(
                tmp,
                lang,
                allow_vision_escalation=allow_vision_escalation,
                gpu_id=gpu_id,
            )
        finally:
            tmp.unlink(missing_ok=True)
        return text, conf, {
            **meta,
            "fallback_from": "fast_ocr",
            "escalation_reason": line_reason,
            "fast_ocr_meta": fast_meta,
            "pipeline": "pdf_crop_line_smart",
            "ocr_in_memory": True,
            "empty_detection": empty_detection,
            "escalated": True,
            "ocr_passes": 2,
        }

    escalate, reason = should_escalate(fast)
    if allow_vision_escalation and escalate:
        if _vision_escalation_allowed():
            logger.info("PDF crop (PIL) → full-page vision (%s)", reason)
            tmp = _save_pil_temp(image)
            try:
                text, conf, meta = ocr_image_vision(tmp)
                return text, conf, {
                    **meta,
                    "fallback_from": "fast_ocr",
                    "escalation_reason": reason,
                    "fast_ocr_meta": fast_meta,
                    "pipeline": "pdf_crop_vision",
                    "ocr_in_memory": True,
                    "empty_detection": empty_detection,
                    "escalated": True,
                    "ocr_passes": 2,
                    "page_vision_escalation": True,
                }
            except VisionLLMError as exc:
                logger.warning("full-page vision failed (%s) — crop escalation", exc)
            finally:
                tmp.unlink(missing_ok=True)
        logger.info("PDF crop (PIL) → crop-level vision (%s)", reason)
        try:
            text, conf, meta = _crop_escalate_fast_result(
                image, None, fast, lang,
                reason=reason or "quality_gate",
                allow_vision_escalation=allow_vision_escalation,
                gpu_id=gpu_id,
                pipeline="pdf_crop_crop_escalation",
                page_quality_poor=True,
                extra_meta={
                    "ocr_in_memory": True,
                    "empty_detection": empty_detection,
                    "fast_ocr_meta": fast_meta,
                },
            )
            return text, conf, meta
        except VisionLLMError as exc:
            logger.warning("crop escalation failed (%s) — keeping fast result", exc)
            fast_meta["escalation_attempted"] = True
            fast_meta["escalation_reason"] = reason
            fast_meta["escalation_error"] = str(exc)

    fast_meta["ocr_in_memory"] = True
    fast_meta["empty_detection"] = empty_detection
    fast_meta["escalated"] = escalated
    fast_meta["ocr_passes"] = ocr_passes
    out_text, fast_meta = _finalize_fast_ocr_output(
        image, None, fast, lang, fast_meta, page=1,
    )
    out_text, fast_meta, _ = _maybe_tesseract_polish_fallback(
        None, image, out_text, fast_meta,
    )
    return out_text, round(fast.mean_score, 4), fast_meta


def _smart_pdf_crop_pipeline(
    image_path: Path,
    lang: str | None,
    *,
    allow_vision_escalation: bool | None = None,
    gpu_id: int | None = None,
) -> tuple[str, float | None, dict]:
    """
    Bulk PDF crops: RapidOCR first (~11 s / 5 str. SKMBT), line-smart only on bad crops.
    """
    from app.services.fast_ocr import ocr_image_fast, should_escalate

    if allow_vision_escalation is None:
        allow_vision_escalation = False

    try:
        fast = ocr_image_fast(image_path, gpu_id=gpu_id)
    except Exception as exc:  # noqa: BLE001
        if not allow_vision_escalation:
            logger.warning("fast_ocr failed (%s) on PDF crop — no vision escalation", exc)
            return "", None, {"engine": "fast_ocr", "error": str(exc), "gpu_id": gpu_id}
        logger.warning("fast_ocr failed (%s) — single-crop vision fallback", exc)
        return _single_crop_vision_fallback(
            None, image_path, reason="fast_ocr_error", pipeline="pdf_crop_crop_vision",
        )

    fast_meta = _fast_ocr_meta_from_result(
        fast, gpu_id=gpu_id, pipeline="pdf_crop_fast", page=1, attach_lines=False,
    )

    line_escalate = False
    line_reason: str | None = None
    if settings.ocr_line_level_escalate_bad_pdf_crops:
        line_escalate, line_reason = _should_escalate_pdf_crop_to_line_smart(fast)

    if line_escalate:
        from app.services.line_smart_ocr import ocr_image_line_smart

        logger.info("PDF crop → line-smart (%s)", line_reason or "empty")
        text, conf, meta = ocr_image_line_smart(
            image_path,
            lang,
            allow_vision_escalation=allow_vision_escalation,
            gpu_id=gpu_id,
        )
        return text, conf, {
            **meta,
            "fallback_from": "fast_ocr",
            "escalation_reason": line_reason,
            "fast_ocr_meta": fast_meta,
            "pipeline": "pdf_crop_line_smart",
        }

    escalate, reason = should_escalate(fast)
    if allow_vision_escalation and escalate:
        if _vision_escalation_allowed():
            logger.info("PDF crop → full-page vision (%s)", reason)
            try:
                text, conf, meta = ocr_image_vision(image_path)
                return text, conf, {
                    **meta,
                    "fallback_from": "fast_ocr",
                    "escalation_reason": reason,
                    "fast_ocr_meta": fast_meta,
                    "pipeline": "pdf_crop_vision",
                    "page_vision_escalation": True,
                }
            except VisionLLMError as exc:
                logger.warning("full-page vision failed (%s) — crop escalation", exc)
        logger.info("PDF crop → crop-level vision (%s)", reason)
        try:
            return _crop_escalate_fast_result(
                None, image_path, fast, lang,
                reason=reason or "quality_gate",
                allow_vision_escalation=allow_vision_escalation,
                gpu_id=gpu_id,
                pipeline="pdf_crop_crop_escalation",
                page_quality_poor=True,
                extra_meta={"fast_ocr_meta": fast_meta},
            )
        except VisionLLMError as exc:
            logger.warning("crop escalation failed (%s) — keeping fast result", exc)
            fast_meta["escalation_attempted"] = True
            fast_meta["escalation_reason"] = reason
            fast_meta["escalation_error"] = str(exc)

    out_text, fast_meta = _finalize_fast_ocr_output(
        None, image_path, fast, lang, fast_meta, page=1,
    )
    out_text, fast_meta, _ = _maybe_tesseract_polish_fallback(
        image_path, None, out_text, fast_meta,
    )
    return out_text, round(fast.mean_score, 4), fast_meta


def _smart_page_level_legacy(
    image_path: Path,
    lang: str | None,
    *,
    allow_vision_escalation: bool | None = None,
    gpu_id: int | None = None,
) -> tuple[str, float | None, dict]:
    """Legacy page-level gate (when ``OCR_LINE_LEVEL_ENABLED=false`` globally)."""
    from app.services.fast_ocr import ocr_image_fast, should_escalate

    if allow_vision_escalation is None:
        allow_vision_escalation = True

    try:
        fast = ocr_image_fast(image_path, gpu_id=gpu_id)
    except Exception as exc:  # noqa: BLE001
        if not allow_vision_escalation:
            logger.warning("fast_ocr failed (%s) on PDF path — no vision escalation", exc)
            return "", None, {"engine": "fast_ocr", "error": str(exc), "gpu_id": gpu_id}
        logger.warning("fast_ocr failed (%s) — escalating to vision_llm", exc)
        return _vision_with_fallback(image_path, lang, reason="fast_ocr_error")

    escalate, reason = should_escalate(fast)
    if not allow_vision_escalation:
        escalate = False
    fast_meta = _fast_ocr_meta_from_result(
        fast, gpu_id=gpu_id, pipeline="page_level_legacy", page=1, attach_lines=False,
    )
    if not escalate:
        out_text, fast_meta = _finalize_fast_ocr_output(
            None, image_path, fast, lang, fast_meta, page=1,
        )
        return out_text, round(fast.mean_score, 4), fast_meta

    if _vision_escalation_allowed():
        logger.info("smart legacy: full-page vision (%s)", reason)
        try:
            text, conf, meta = ocr_image_vision(image_path)
            return text, conf, {
                **meta,
                "fallback_from": "fast_ocr",
                "escalation_reason": reason,
                "fast_ocr_meta": fast_meta,
                "pipeline": "page_level_legacy",
                "page_vision_escalation": True,
            }
        except VisionLLMError as exc:
            logger.warning("full-page vision failed (%s) — crop escalation", exc)

    logger.info("smart legacy: crop-level vision (%s)", reason)
    try:
        return _crop_escalate_fast_result(
            None, image_path, fast, lang,
            reason=reason or "quality_gate",
            allow_vision_escalation=allow_vision_escalation,
            gpu_id=gpu_id,
            pipeline="page_level_crop_escalation",
            page_quality_poor=True,
            extra_meta={"fast_ocr_meta": fast_meta},
        )
    except VisionLLMError as exc:
        logger.warning("crop escalation failed (%s) — keeping fast result", exc)
        fast_meta["escalation_attempted"] = True
        fast_meta["escalation_reason"] = reason
        fast_meta["escalation_error"] = str(exc)
        out_text, fast_meta = _finalize_fast_ocr_output(
            None, image_path, fast, lang, fast_meta, page=1,
        )
        return out_text, round(fast.mean_score, 4), fast_meta


def _smart_pipeline(
    image_path: Path,
    lang: str | None,
    *,
    allow_vision_escalation: bool | None = None,
    gpu_id: int | None = None,
    line_level: bool | None = None,
) -> tuple[str, float | None, dict]:
    """Photos: line-smart. PDF crops: RapidOCR + selective line-smart. Legacy if disabled."""
    if _use_line_smart(line_level):
        from app.services.line_smart_ocr import ocr_image_line_smart

        return ocr_image_line_smart(
            image_path,
            lang,
            allow_vision_escalation=allow_vision_escalation,
            gpu_id=gpu_id,
        )

    if line_level is False:
        return _smart_pdf_crop_pipeline(
            image_path, lang,
            allow_vision_escalation=allow_vision_escalation,
            gpu_id=gpu_id,
        )

    return _smart_page_level_legacy(
        image_path, lang,
        allow_vision_escalation=allow_vision_escalation,
        gpu_id=gpu_id,
    )


def _fast_only(
    image_path: Path, lang: str | None, *, gpu_id: int | None = None,
) -> tuple[str, float | None, dict]:
    from app.services.fast_ocr import ocr_image_fast
    try:
        fast = ocr_image_fast(image_path, gpu_id=gpu_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fast_ocr failed (%s) — falling back to tesseract", exc)
        text, conf = _tesseract_ocr(image_path, lang)
        return text, conf, {"engine": "tesseract", "fallback_from": "fast_ocr"}
    fast_meta = _fast_ocr_meta_from_result(
        fast, gpu_id=gpu_id, pipeline="fast_only", page=1, attach_lines=False,
    )
    out_text, fast_meta = _finalize_fast_ocr_output(
        None, image_path, fast, lang, fast_meta, page=1,
    )
    return out_text, round(fast.mean_score, 4), fast_meta


def _vision_with_fallback(
    image_path: Path, lang: str | None, *, reason: str | None = None
) -> tuple[str, float | None, dict]:
    try:
        text, conf, meta = ocr_image_vision(image_path)
        if reason:
            meta["escalation_reason"] = reason
        return text, conf, meta
    except VisionLLMError as exc:
        logger.warning("vision_llm failed (%s) — falling back to Tesseract", exc)
        text, conf = _tesseract_ocr(image_path, lang)
        return text, conf, {"engine": "tesseract", "fallback_from": "vision_llm"}


# ── tesseract path ────────────────────────────────────────────────────────────

def _tesseract_ocr(
    image_path: Path, lang: str | None = None
) -> tuple[str, float | None]:
    from app.services.ocr_lang_routing import resolve_tesseract_lang

    language = lang if lang is not None else resolve_tesseract_lang(None)

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        photo = is_photo(rgb)
        receipt = is_receipt_like(rgb)
        scale = settings.ocr_receipt_scale if photo else 1.0

        prepared = prepare_for_ocr(rgb, scale=scale, force_photo=photo)

        if photo:
            psm = settings.ocr_psm_receipt if receipt else settings.ocr_psm_document
            text = _conf_filtered_ocr(prepared, language=language, psm=psm)
        else:
            psm = settings.ocr_psm_document
            raw = pytesseract.image_to_string(
                prepared, lang=language, config=f"--psm {psm} --oem 3"
            )
            text = _clean_text(raw)

        confidence = _mean_confidence(prepared, language)

    return text.strip(), confidence


# ── confidence-filtered OCR (for phone photos) ────────────────────────────────

def _conf_filtered_ocr(
    img: Image.Image,
    *,
    language: str,
    psm: int,
    min_line_conf: int = 30,
    min_word_conf: int = 45,
) -> str:
    """
    Run Tesseract in word-data mode and reconstruct text using only words
    whose confidence meets the threshold.  Also filters words whose X position
    falls in the right-edge noise zone (shadow / texture artefacts outside the
    text column of a phone photo of a receipt).
    """
    data = pytesseract.image_to_data(
        img, lang=language,
        config=f"--psm {psm} --oem 3",
        output_type=pytesseract.Output.DICT,
    )

    img_w = img.width
    # Receipt text never extends past ~60 % of the image width in a typical
    # phone photo; everything to the right is a dark shadow / paper texture
    # that Tesseract misreads.  Use a generous cutoff (65 %) to avoid clipping
    # valid prices on wide receipts.
    right_cutoff = int(img_w * 0.65)

    lines_words: dict[tuple, list[tuple[str, int]]] = defaultdict(list)
    lines_confs: dict[tuple, list[int]] = defaultdict(list)

    for i, conf_raw in enumerate(data["conf"]):
        word = data["text"][i]
        if not word.strip():
            continue
        try:
            conf = int(conf_raw)
        except (ValueError, TypeError):
            continue

        # Skip words whose left edge is in the noise zone
        if data["left"][i] >= right_cutoff:
            continue

        # Drop short non-digit tokens with low confidence: shadow/texture artefacts
        # (noise like "gS", "ees", "AO") always come in with conf < 55.
        # Legitimate short tokens ("ml", "SL", "3", "750") have conf ≥ 55.
        if conf < 55 and len(word) <= 3 and not word.isdigit():
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines_words[key].append((word, conf))
        lines_confs[key].append(conf)

    result_lines: list[str] = []
    for key in sorted(lines_words):
        confs = lines_confs[key]
        mean_conf = sum(confs) / len(confs)
        if mean_conf < min_line_conf:
            continue

        kept = [w for w, c in lines_words[key] if c >= min_word_conf]
        if not kept:
            continue

        # Strip isolated special-char tokens from the ends of each line
        while kept and len(kept[0]) == 1 and not kept[0].isalnum():
            kept.pop(0)
        while kept and len(kept[-1]) == 1 and not kept[-1].isalnum():
            kept.pop()

        line = " ".join(kept)
        if len(line.strip()) >= 2:
            result_lines.append(line)

    return _dedup_adjacent(result_lines)


def _dedup_adjacent(lines: list[str]) -> str:
    """Remove consecutive duplicate lines (same normalised content)."""
    out: list[str] = []
    prev = None
    for line in lines:
        norm = re.sub(r"\s+", " ", line).strip().lower()
        if norm != prev:
            out.append(line)
            prev = norm
    return "\n".join(out)


# ── text cleaning (for clean scans) ──────────────────────────────────────────

_SEPARATOR_RE = re.compile(r"^[\-=_.•·*#~<>|\\/ ]{4,}$")


def _noise_ratio(line: str) -> float:
    stripped = line.strip()
    if not stripped:
        return 1.0
    useful = sum(1 for c in stripped if c.isalnum() or c in " ,./:%-+@#&()'\"")
    return 1.0 - useful / len(stripped)


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _SEPARATOR_RE.match(stripped):
        return False
    if len(stripped) <= 3 and not any(c.isalnum() for c in stripped):
        return True
    return _noise_ratio(stripped) >= 0.55


def _clean_text(raw: str) -> str:
    """Remove OCR garbage lines; collapse multiple blanks."""
    lines = raw.splitlines()
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        if _is_noise_line(line):
            if not prev_blank:
                cleaned.append("")
                prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = line.strip() == ""
    return "\n".join(cleaned).strip()


# ── confidence ────────────────────────────────────────────────────────────────

def _mean_confidence(img: Image.Image, lang: str) -> float | None:
    try:
        data = pytesseract.image_to_data(
            img, lang=lang, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError:
        return None

    vals = [
        int(v)
        for v in data.get("conf", [])
        if str(v).lstrip("-").isdigit() and int(v) >= 0
    ]
    if not vals:
        return None
    return round(sum(vals) / len(vals) / 100.0, 4)
