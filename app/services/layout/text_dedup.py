"""Backward-compatible re-exports — use :mod:`app.services.ocr_dedup` instead."""

from __future__ import annotations

from app.services.ocr_dedup import (
    OcrDedupPolicy,
    dedupe_raw_detections,
    is_duplicate_detection,
    merge_line_detections as merge_overlapping_box_texts,
    pick_better_text,
    texts_exact_same,
)

__all__ = [
    "OcrDedupPolicy",
    "dedupe_raw_detections",
    "is_duplicate_detection",
    "merge_overlapping_box_texts",
    "pick_better_text",
    "texts_exact_same",
]
