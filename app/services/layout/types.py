"""Shared dataclasses for layout reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LayoutBox:
    """Single OCR box used by layout reconstruction.

    Indices keep the link to the caller's original list so they can update
    text/order without losing metadata.
    """

    index: int
    text: str
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    score: float = 0.0
    column: int = 0
    line: int = 0
    is_barcode: bool = False
    is_handwriting: bool = False
    suppressed: bool = False
    duplicate_of: int | None = None
    reading_order: int = 0


@dataclass
class LayoutFeatures:
    has_checkboxes: bool = False
    checkbox_count: int = 0
    horizontal_line_count: int = 0
    barcode_region_count: int = 0
    column_count: int = 1
    sparse_text: bool = False
    text_area_ratio: float = 0.0
    handwriting_line_count: int = 0
    form_mode: bool = False
    suppressed_duplicate_count: int = 0

    def to_dict(self) -> dict:
        return {
            "has_checkboxes": self.has_checkboxes,
            "checkbox_count": self.checkbox_count,
            "horizontal_line_count": self.horizontal_line_count,
            "barcode_region_count": self.barcode_region_count,
            "column_count": self.column_count,
            "sparse_text": self.sparse_text,
            "text_area_ratio": round(self.text_area_ratio, 4),
            "handwriting_line_count": self.handwriting_line_count,
            "form_mode": self.form_mode,
            "suppressed_duplicate_count": self.suppressed_duplicate_count,
        }


@dataclass
class LayoutResult:
    """Output of :func:`reconstruct_layout`."""

    ordered_indices: list[int] = field(default_factory=list)
    suppressed_indices: list[int] = field(default_factory=list)
    boxes: list[LayoutBox] = field(default_factory=list)
    features: LayoutFeatures = field(default_factory=LayoutFeatures)
    barcode_regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    decoded_barcodes: list[str] = field(default_factory=list)
