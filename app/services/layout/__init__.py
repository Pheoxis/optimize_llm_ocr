"""
Form-aware OCR layout reconstruction.

Public entry point: :func:`reconstruct_layout` takes a list of OCR boxes and the
source image, and returns a reordered, deduplicated set of boxes plus detected
form / barcode / handwriting features.

Submodules are intentionally small and depend only on numpy / OpenCV / PIL so
the full layout pass costs ~50-150 ms per page even on dense scans.
"""

from __future__ import annotations

from app.services.layout.types import LayoutBox, LayoutFeatures, LayoutResult
from app.services.layout.barcode import detect_barcode_regions
from app.services.layout.columns import detect_columns
from app.services.layout.dedup import suppress_duplicates
from app.services.layout.form_detect import detect_form_features
from app.services.layout.handwriting import flag_handwriting_lines
from app.services.layout.lines import cluster_lines
from app.services.layout.reading_order import reading_order_indices
from app.services.layout.reconstruction import (
    LayoutReconstructionConfig,
    merged_text_from_layout,
    reconstruct_layout,
)

__all__ = [
    "LayoutBox",
    "LayoutFeatures",
    "LayoutReconstructionConfig",
    "LayoutResult",
    "cluster_lines",
    "detect_barcode_regions",
    "detect_columns",
    "detect_form_features",
    "flag_handwriting_lines",
    "merged_text_from_layout",
    "reading_order_indices",
    "reconstruct_layout",
    "suppress_duplicates",
]
