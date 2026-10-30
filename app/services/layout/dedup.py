"""IoU + text-similarity duplicate suppression for OCR boxes."""

from __future__ import annotations

from app.services.layout.types import LayoutBox


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 <= ax0 or ay1 <= ay0 or bx1 <= bx0 or by1 <= by0:
        return 0.0
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in (text or "").split() if t}


def text_jaccard(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def suppress_duplicates(
    boxes: list[LayoutBox],
    *,
    iou_threshold: float = 0.55,
    text_threshold: float = 0.70,
    exact_text_only: bool = True,
    image_width: int | None = None,
) -> int:
    """Mark duplicate boxes as ``suppressed`` in place. Returns count suppressed.

    Default ``exact_text_only=True``: only suppress when bbox IoU is high AND
  text is identical (Unicode casefold). This preserves intentional duplicate
    labels on forms at different positions.

    Set ``exact_text_only=False`` to use token Jaccard (legacy; not recommended
    for PL forms).
    """
    from app.services.ocr_dedup import OcrDedupPolicy, is_duplicate_detection

    if len(boxes) < 2:
        return 0
    policy = OcrDedupPolicy.from_settings()
    if iou_threshold > policy.overlap_iou:
        policy = OcrDedupPolicy(
            overlap_iou=iou_threshold,
            max_adjacent_gap_px=policy.max_adjacent_gap_px,
            min_vertical_overlap=policy.min_vertical_overlap,
            containment_ratio=policy.containment_ratio,
        )

    order = sorted(
        range(len(boxes)),
        key=lambda i: (boxes[i].score, len(boxes[i].text or "")),
        reverse=True,
    )

    suppressed = 0
    for pos, i in enumerate(order):
        if boxes[i].suppressed:
            continue
        for j in order[pos + 1 :]:
            if boxes[j].suppressed:
                continue
            if exact_text_only:
                if not is_duplicate_detection(
                    boxes[i].bbox,
                    boxes[j].bbox,
                    boxes[i].text,
                    boxes[j].text,
                    policy,
                    image_width=image_width,
                ):
                    continue
            else:
                if bbox_iou(boxes[i].bbox, boxes[j].bbox) < iou_threshold:
                    continue
                sim = text_jaccard(boxes[i].text, boxes[j].text)
                if sim < text_threshold:
                    continue
            boxes[j].suppressed = True
            boxes[j].duplicate_of = boxes[i].index
            suppressed += 1
    return suppressed
