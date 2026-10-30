"""Layout reconstruction orchestrator.

Public entry point used by ``crop_escalation`` and other pipeline stages.

Flow
----

1. Build :class:`LayoutBox` list from caller's OCR results (Rapid / line-smart).
2. (optional) Detect barcode regions and mark boxes overlapping them.
3. (optional) Detect form features (checkboxes, horizontal lines, sparse).
4. Detect column boundaries, assign each box to a column.
5. Cluster lines (skew tolerant) per column.
6. Duplicate suppression (bbox IoU + text Jaccard).
7. Flag suspected handwriting boxes.
8. Reading order reconstruction.

Returns :class:`LayoutResult` with the ordered indices and detected features
so the caller can rebuild text, mark lines and emit metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from PIL import Image

from app.services.layout.barcode import detect_barcode_regions
from app.services.layout.columns import assign_columns, detect_columns
from app.services.layout.dedup import bbox_iou, suppress_duplicates
from app.services.layout.form_detect import detect_form_features
from app.services.layout.handwriting import flag_handwriting_lines
from app.services.layout.lines import cluster_lines
from app.services.layout.reading_order import (
    lines_from_reading_order,
    reading_order_indices,
)
from app.services.layout.types import LayoutBox, LayoutFeatures, LayoutResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayoutReconstructionConfig:
    enable_barcodes: bool = True
    enable_form_detection: bool = True
    enable_columns: bool = True
    enable_handwriting_flag: bool = True
    enable_duplicate_suppression: bool = True
    barcode_min_overlap: float = 0.30
    iou_threshold: float = 0.55
    text_threshold: float = 0.70
    iou_threshold_form: float = 0.40
    text_threshold_form: float = 0.55
    skew_tolerance_ratio: float = 0.45


def _to_layout_boxes(
    items: Iterable[tuple[str, tuple[int, int, int, int], float]],
) -> list[LayoutBox]:
    boxes: list[LayoutBox] = []
    for i, (text, bbox, score) in enumerate(items):
        x0, y0, x1, y1 = bbox
        boxes.append(
            LayoutBox(
                index=i,
                text=text or "",
                bbox=(int(x0), int(y0), int(x1), int(y1)),
                score=float(score or 0.0),
            ),
        )
    return boxes


def _box_overlaps_region(
    box: tuple[int, int, int, int],
    region: tuple[int, int, int, int],
    *,
    min_overlap: float,
) -> bool:
    bx0, by0, bx1, by1 = box
    rx0, ry0, rx1, ry1 = region
    ix0 = max(bx0, rx0)
    iy0 = max(by0, ry0)
    ix1 = min(bx1, rx1)
    iy1 = min(by1, ry1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    box_area = max(1, (bx1 - bx0) * (by1 - by0))
    return (inter / box_area) >= min_overlap


def reconstruct_layout(
    items: Iterable[tuple[str, tuple[int, int, int, int], float]],
    image: Image.Image | None = None,
    *,
    image_size: tuple[int, int] | None = None,
    config: LayoutReconstructionConfig | None = None,
) -> LayoutResult:
    """Run layout reconstruction.

    ``items`` is a list of ``(text, bbox, score)`` tuples in the caller's
    original order. ``bbox`` is in image coordinates ``(x0, y0, x1, y1)``.
    Either ``image`` (PIL) or ``image_size`` must be provided.
    """
    cfg = config or LayoutReconstructionConfig()
    boxes = _to_layout_boxes(items)
    result = LayoutResult(boxes=boxes)

    if not boxes:
        return result

    if image is not None:
        w, h = image.size
    elif image_size is not None:
        w, h = image_size
    else:
        w = max(b.bbox[2] for b in boxes) + 1
        h = max(b.bbox[3] for b in boxes) + 1

    # 1. Barcode regions (optional)
    if cfg.enable_barcodes and image is not None:
        try:
            regions = detect_barcode_regions(image)
        except Exception as exc:  # noqa: BLE001
            logger.debug("barcode detection skipped: %s", exc)
            regions = []
        result.barcode_regions = [r.bbox for r in regions]
        result.decoded_barcodes = [r.data for r in regions if r.data]
        result.features.barcode_region_count = len(regions)
        if regions:
            for b in boxes:
                for r in regions:
                    if _box_overlaps_region(
                        b.bbox, r.bbox, min_overlap=cfg.barcode_min_overlap,
                    ):
                        b.is_barcode = True
                        break

    # 2. Form features (optional)
    if cfg.enable_form_detection and image is not None:
        result.features = detect_form_features(image, boxes)
        # Carry barcode counts from step 1 because detect_form_features
        # resets the LayoutFeatures instance.
        result.features.barcode_region_count = len(result.barcode_regions)
    elif cfg.enable_form_detection:
        # Without an image we still derive a sparse-text estimate from boxes
        text_area = sum(
            max(0, b.bbox[2] - b.bbox[0]) * max(0, b.bbox[3] - b.bbox[1])
            for b in boxes
            if not b.suppressed and not b.is_barcode
        )
        result.features.text_area_ratio = text_area / max(1, w * h)
        result.features.sparse_text = result.features.text_area_ratio < 0.18

    form_mode = result.features.form_mode

    # 3. Columns
    if cfg.enable_columns:
        columns = detect_columns(
            [b for b in boxes if not b.is_barcode],
            image_width=w,
        )
        result.features.column_count = len(columns)
        assign_columns(boxes, columns)

    # 4. Line clustering
    skew_tol = cfg.skew_tolerance_ratio
    if form_mode:
        skew_tol = max(0.25, skew_tol * 0.7)  # stricter merging in forms
    cluster_lines(boxes, skew_tolerance_ratio=skew_tol)

    # 5. Duplicate suppression — universal geometry policy (see ocr_dedup)
    if cfg.enable_duplicate_suppression:
        suppressed = suppress_duplicates(
            boxes,
            iou_threshold=cfg.iou_threshold,
            exact_text_only=True,
            image_width=w,
        )
        result.features.suppressed_duplicate_count = suppressed
        result.suppressed_indices = [b.index for b in boxes if b.suppressed]

    # 6. Handwriting flag
    if cfg.enable_handwriting_flag:
        result.features.handwriting_line_count = flag_handwriting_lines(boxes)

    # 7. Reading order
    result.ordered_indices = reading_order_indices(boxes)
    return result


def merged_text_from_layout(
    layout: LayoutResult,
    *,
    separator: str = "    ",
) -> str:
    """Rebuild a single text block from a ``LayoutResult``."""
    lines = lines_from_reading_order(
        layout.boxes, layout.ordered_indices, separator=separator,
    )
    return "\n".join(line for line in lines if line.strip())


__all__ = [
    "LayoutReconstructionConfig",
    "reconstruct_layout",
    "merged_text_from_layout",
    "bbox_iou",
]
