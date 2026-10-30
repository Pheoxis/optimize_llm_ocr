"""Structured OCR line objects (bbox, confidence, page, engine)."""

from __future__ import annotations

from typing import Any

from app.models.schemas import OcrLine
from app.services.semantic_validators.line_enrichment import enrich_ocr_line_semantic


def _bbox_list(bbox: tuple[int, ...] | list[int] | None) -> list[int]:
    if not bbox or len(bbox) < 4:
        return [0, 0, 0, 0]
    return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]


def from_fast_ocr_line(
    line: Any,
    *,
    page: int = 1,
    engine: str = "fast_ocr",
    document: int | None = None,
    line_index: int | None = None,
    y_offset: int = 0,
    x_offset: int = 0,
) -> OcrLine:
    """Convert a ``FastOCRLine`` (or compatible) to ``OcrLine``."""
    x0, y0, x1, y1 = _bbox_list(getattr(line, "bbox", None))
    if x_offset or y_offset:
        x0 += x_offset
        x1 += x_offset
        y0 += y_offset
        y1 += y_offset
    conf = float(getattr(line, "mean_score", 0.0))
    return enrich_ocr_line_semantic(OcrLine(
        text=getattr(line, "text", "") or "",
        confidence=round(conf, 4),
        bbox=[x0, y0, x1, y1],
        page=page,
        engine=engine,
        document=document,
        line_index=line_index,
        rapid_text=getattr(line, "text", None),
    ))


def lines_from_fast_result(
    fast: Any,
    *,
    page: int = 1,
    engine: str = "fast_ocr",
    document: int | None = None,
    y_offset: int = 0,
    x_offset: int = 0,
) -> list[OcrLine]:
    """All visual lines from a ``FastOCRResult``."""
    raw_lines = getattr(fast, "lines", None) or []
    return [
        from_fast_ocr_line(
            ln,
            page=page,
            engine=engine,
            document=document,
            line_index=idx,
            y_offset=y_offset,
            x_offset=x_offset,
        )
        for idx, ln in enumerate(raw_lines)
    ]


def line_dicts_from_fast_result(
    fast: Any,
    *,
    page: int = 1,
    engine: str = "fast_ocr",
    document: int | None = None,
) -> list[dict[str, Any]]:
    """Serializable line dicts for OCR metadata."""
    return [ln.model_dump() for ln in lines_from_fast_result(
        fast, page=page, engine=engine, document=document,
    )]


def _dict_to_ocr_line(item: dict[str, Any], *, default_page: int = 1) -> OcrLine:
    conf = item.get("confidence", item.get("mean_score", 0.0))
    return OcrLine(
        text=str(item.get("text", "") or ""),
        confidence=round(float(conf or 0.0), 4),
        bbox=_bbox_list(item.get("bbox")),
        page=int(item.get("page", default_page)),
        engine=str(item.get("engine", "fast_ocr")),
        document=item.get("document"),
        line_index=item.get("line_index"),
        escalated=item.get("escalated"),
        suspicious=item.get("suspicious"),
        disagreement=item.get("disagreement", item.get("disagreement_score")),
        rapid_text=item.get("rapid_text"),
        tesseract_text=item.get("tesseract_text"),
        verification_skipped=item.get("verification_skipped"),
        escalation_hint=item.get("escalation_hint"),
        validation_passed=item.get("validation_passed"),
        validation_issues=item.get("validation_issues"),
        validation_failed_kinds=item.get("validation_failed_kinds"),
    )


def parse_lines_from_meta(
    meta: dict[str, Any],
    *,
    default_page: int = 1,
) -> list[OcrLine]:
    """Parse ``lines`` embedded in OCR metadata (fast_ocr or line-smart)."""
    raw = meta.get("lines")
    if not isinstance(raw, list):
        return []
    out: list[OcrLine] = []
    for idx, item in enumerate(raw):
        if isinstance(item, OcrLine):
            out.append(item)
        elif isinstance(item, dict):
            parsed = _dict_to_ocr_line(item, default_page=default_page)
            if parsed.line_index is None:
                parsed = parsed.model_copy(update={"line_index": idx})
            out.append(parsed)
    return out


def merge_lines_text(lines: list[OcrLine]) -> str:
    """Rebuild merged text from line objects (same order as OCR join)."""
    return "\n".join(ln.text for ln in lines if ln.text)


def aggregate_lines_from_metas(
    metas: list[dict[str, Any]],
    *,
    page_key: str = "page",
) -> list[OcrLine]:
    """Collect lines from per-page / per-document OCR metadata dicts."""
    combined: list[OcrLine] = []
    for meta in metas:
        page = int(meta.get(page_key, 1) or 1)
        combined.extend(parse_lines_from_meta(meta, default_page=page))
    return combined


def line_dicts_from_line_smart_debug(
    debug_lines: list[Any],
    *,
    page: int = 1,
    document: int | None = None,
) -> list[dict[str, Any]]:
    """Canonical line dicts from ``LineDebugMeta`` entries."""
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(debug_lines):
        if isinstance(item, dict):
            parsed = _dict_to_ocr_line({**item, "page": item.get("page", page)}, default_page=page)
        else:
            parsed = OcrLine(
                text=getattr(item, "text", "") or "",
                confidence=round(float(getattr(item, "confidence", 0.0)), 4),
                bbox=_bbox_list(getattr(item, "bbox", None)),
                page=page,
                engine=str(getattr(item, "engine", "line_smart_ocr")),
                document=document,
                line_index=idx,
                escalated=getattr(item, "escalated", None),
                suspicious=getattr(item, "suspicious", None),
                disagreement=getattr(
                    item, "disagreement", getattr(item, "disagreement_score", None),
                ),
                rapid_text=getattr(item, "rapid_text", None) or None,
                tesseract_text=getattr(item, "tesseract_text", None) or None,
                verification_skipped=getattr(item, "verification_skipped", None)
                if getattr(item, "verification_skipped", None) is not None
                else getattr(item, "tess_skipped", None),
                escalation_hint=getattr(item, "escalation_hint", None)
                or getattr(item, "escalation_reason", None),
            )
        if parsed.line_index is None:
            parsed = parsed.model_copy(update={"line_index": idx})
        out.append(parsed.model_dump())
    return out


def bbox_crop_box(
    line: OcrLine,
    *,
    pad: int = 0,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[int, int, int, int]:
    """Pixel crop ``(x0, y0, x1, y1)`` for bbox-based line extraction."""
    x0, y0, x1, y1 = line.bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = x1 + pad
    y1 = y1 + pad
    if image_width is not None:
        x1 = min(image_width, x1)
    if image_height is not None:
        y1 = min(image_height, y1)
    return x0, y0, x1, y1


def attach_fast_lines_to_meta(
    meta: dict[str, Any],
    fast: Any,
    *,
    page: int = 1,
    engine: str | None = None,
    document: int | None = None,
) -> dict[str, Any]:
    """Add canonical ``lines`` list to metadata (mutates and returns *meta*)."""
    eng = engine or meta.get("engine", "fast_ocr")
    meta["lines"] = line_dicts_from_fast_result(
        fast, page=page, engine=str(eng), document=document,
    )
    meta["n_lines"] = len(meta["lines"])
    return meta
