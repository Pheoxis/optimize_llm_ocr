"""Bridge OCR cache ↔ FastOCR / vision engines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import settings
from app.services.ocr_cache import (
    OcrCacheEntry,
    fast_ocr_preprocess_variant,
    file_bytes_for_cache,
    get_or_compute,
    image_bytes_for_cache,
    vision_preprocess_variant,
)


def _fast_result_to_entry(result: Any) -> OcrCacheEntry:
    from app.services.fast_ocr import FastOCRLine

    lines_payload: list[dict[str, Any]] = []
    for ln in getattr(result, "lines", None) or []:
        if isinstance(ln, FastOCRLine):
            lines_payload.append({
                "text": ln.text,
                "mean_score": ln.mean_score,
                "min_score": ln.min_score,
                "n_boxes": ln.n_boxes,
                "bbox": list(ln.bbox),
            })
        elif isinstance(ln, dict):
            lines_payload.append(ln)

    meta = {
        "engine": "fast_ocr",
        "model": (
            f"PP-OCRv5/{settings.fast_lang_type}/{settings.fast_model_type}"
        ),
        "elapsed_sec": round(getattr(result, "elapsed_sec", 0.0), 3),
        "cache": True,
    }
    return OcrCacheEntry(
        text=getattr(result, "text", "") or "",
        confidence=round(getattr(result, "mean_score", 0.0), 4),
        metadata=meta,
        engine="fast_ocr",
        lines=lines_payload,
        min_score=round(getattr(result, "min_score", 0.0), 4),
        low_conf_ratio=round(getattr(result, "low_conf_ratio", 0.0), 4),
        n_boxes=int(getattr(result, "n_boxes", 0)),
    )


def _entry_to_fast_result(entry: OcrCacheEntry) -> Any:
    from app.services.fast_ocr import FastOCRLine, FastOCRResult

    lines: list[FastOCRLine] = []
    for item in entry.lines or []:
        bbox = item.get("bbox", (0, 0, 0, 0))
        if isinstance(bbox, list):
            bbox = tuple(int(x) for x in bbox[:4])
        lines.append(FastOCRLine(
            text=str(item.get("text", "")),
            mean_score=float(item.get("mean_score", entry.confidence or 0.0)),
            min_score=float(item.get("min_score", entry.min_score or 0.0)),
            n_boxes=int(item.get("n_boxes", 1)),
            bbox=bbox,
        ))

    meta = dict(entry.metadata)
    meta["ocr_cache_hit"] = True
    elapsed = float(meta.get("elapsed_sec", 0.0))

    return FastOCRResult(
        text=entry.text,
        lines=lines,
        mean_score=float(entry.confidence or 0.0),
        min_score=float(entry.min_score or 0.0),
        low_conf_ratio=float(entry.low_conf_ratio or 0.0),
        n_boxes=int(entry.n_boxes or len(lines)),
        elapsed_sec=elapsed,
    )


def cached_ocr_pil_fast(
    image: Image.Image,
    compute: Any,
    *,
    preprocess_variant: str = "original",
) -> Any:
    """Wrap ``ocr_pil_fast`` body with disk cache."""
    variant = fast_ocr_preprocess_variant(preprocess_variant)
    img_bytes = image_bytes_for_cache(image)

    def _compute() -> OcrCacheEntry:
        result = compute()
        return _fast_result_to_entry(result)

    entry, hit = get_or_compute(img_bytes, variant, _compute)
    return _entry_to_fast_result(entry)


def cached_ocr_image_fast(
    image_path: Path,
    compute: Any,
    *,
    preprocess_variant: str = "original",
) -> Any:
    variant = fast_ocr_preprocess_variant(preprocess_variant)
    file_bytes = file_bytes_for_cache(image_path)

    def _compute() -> OcrCacheEntry:
        return _fast_result_to_entry(compute())

    entry, _hit = get_or_compute(file_bytes, variant, _compute)
    return _entry_to_fast_result(entry)


def cached_vision_ocr(
    image: Image.Image,
    compute: Any,
    *,
    model: str,
    max_side: int,
    preprocess_variant: str | None = None,
) -> tuple[str, dict[str, Any], bool]:
    """
    Wrap vision OCR. Returns ``(text, metadata, cache_hit)``.
    """
    import io

    from app.services.vision_llm import _downscale

    variant = preprocess_variant or vision_preprocess_variant(model, max_side=max_side)
    rgb = _downscale(image.convert("RGB"), max_side)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=settings.vision_jpeg_quality, optimize=False)
    jpeg_bytes = buf.getvalue()

    def _compute() -> OcrCacheEntry:
        text, meta = compute()
        return OcrCacheEntry(
            text=text,
            confidence=None,
            metadata={**meta, "engine": "vision_llm"},
            engine="vision_llm",
        )

    entry, hit = get_or_compute(jpeg_bytes, variant, _compute)
    meta = dict(entry.metadata)
    if hit:
        meta["ocr_cache_hit"] = True
    return entry.text, meta, hit
