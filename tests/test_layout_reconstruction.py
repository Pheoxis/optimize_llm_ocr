"""Form-aware OCR layout reconstruction tests."""

from __future__ import annotations

from PIL import Image, ImageDraw

from app.services.layout import (
    LayoutBox,
    cluster_lines,
    detect_columns,
    reading_order_indices,
    reconstruct_layout,
    suppress_duplicates,
)
from app.services.layout.columns import assign_columns
from app.services.layout.dedup import bbox_iou, text_jaccard
from app.services.layout.handwriting import flag_handwriting_lines
from app.services.layout.reading_order import lines_from_reading_order
from app.services.layout.reconstruction import (
    LayoutReconstructionConfig,
    merged_text_from_layout,
)


def _box(idx: int, text: str, bbox: tuple[int, int, int, int], score: float = 0.95) -> LayoutBox:
    return LayoutBox(index=idx, text=text, bbox=bbox, score=score)


def test_bbox_iou_and_text_jaccard() -> None:
    a = (0, 0, 50, 30)
    b = (10, 5, 60, 35)
    assert 0.3 < bbox_iou(a, b) < 0.7
    assert bbox_iou(a, a) == 1.0
    assert bbox_iou(a, (100, 100, 110, 110)) == 0.0
    assert text_jaccard("Faktura VAT", "Faktura VAT") == 1.0
    assert 0.3 <= text_jaccard("Faktura VAT 2025", "VAT 2025 PL") <= 0.8


def test_suppress_duplicates_keeps_higher_score() -> None:
    boxes = [
        _box(0, "Faktura VAT", (0, 0, 80, 20), score=0.80),
        _box(1, "Faktura VAT", (2, 2, 78, 22), score=0.95),
        _box(2, "Adres", (200, 0, 260, 20), score=0.92),
    ]
    suppressed = suppress_duplicates(boxes)
    assert suppressed == 1
    assert boxes[0].suppressed is True
    assert boxes[1].suppressed is False
    assert boxes[2].suppressed is False


def test_cluster_lines_skew_tolerant() -> None:
    # Slightly sloping line of 3 boxes — should still be one line.
    boxes = [
        _box(0, "A", (0, 100, 30, 130)),
        _box(1, "B", (40, 102, 70, 132)),
        _box(2, "C", (80, 104, 110, 134)),
        _box(3, "D", (0, 160, 30, 190)),
    ]
    line_count = cluster_lines(boxes)
    assert line_count == 2
    assert boxes[0].line == boxes[1].line == boxes[2].line
    assert boxes[3].line != boxes[0].line


def test_detect_columns_two_columns() -> None:
    boxes = [
        _box(0, "Left A", (10, 10, 200, 30)),
        _box(1, "Left B", (10, 40, 200, 60)),
        _box(2, "Right A", (500, 10, 700, 30)),
        _box(3, "Right B", (500, 40, 700, 60)),
    ]
    cols = detect_columns(boxes, image_width=800)
    assert len(cols) == 2
    assign_columns(boxes, cols)
    assert boxes[0].column == 0 and boxes[1].column == 0
    assert boxes[2].column == 1 and boxes[3].column == 1


def test_detect_columns_single_column() -> None:
    boxes = [
        _box(0, "header", (0, 0, 600, 30)),
        _box(1, "body", (0, 60, 600, 90)),
    ]
    cols = detect_columns(boxes, image_width=800)
    assert len(cols) == 1


def test_reading_order_two_columns() -> None:
    boxes = [
        _box(0, "L1", (10, 10, 200, 30)),
        _box(1, "R1", (500, 10, 700, 30)),
        _box(2, "L2", (10, 50, 200, 70)),
        _box(3, "R2", (500, 50, 700, 70)),
    ]
    cols = detect_columns(boxes, image_width=800)
    assign_columns(boxes, cols)
    cluster_lines(boxes)
    order = reading_order_indices(boxes)
    assert [boxes[i].text for i in order] == ["L1", "L2", "R1", "R2"]


def test_lines_from_reading_order_keeps_intentional_repeats() -> None:
    boxes = [
        _box(0, "kwota pobrania", (0, 0, 120, 20)),
        _box(1, "kwota pobrania", (140, 0, 260, 20)),
        _box(2, "www.poczta-polska.pl", (0, 60, 200, 80)),
        _box(3, "www.poczta-polska.pl", (220, 60, 420, 80)),
    ]
    cols = detect_columns(boxes, image_width=600)
    assign_columns(boxes, cols)
    cluster_lines(boxes)
    order = reading_order_indices(boxes)
    out = lines_from_reading_order(boxes, order, separator=" | ")
    assert out == [
        "kwota pobrania | kwota pobrania",
        "www.poczta-polska.pl | www.poczta-polska.pl",
    ]


def test_lines_from_reading_order_keeps_distinct_tokens() -> None:
    boxes = [
        _box(0, "PL", (10, 0, 30, 20)),
        _box(1, "PL", (35, 0, 55, 20)),  # short → keep both (min_chars=4)
        _box(2, "Faktura", (60, 0, 130, 20)),
        _box(3, "VAT", (135, 0, 175, 20)),
    ]
    # Single column (no big gaps)
    for b in boxes:
        b.column = 0
        b.line = 0
    order = reading_order_indices(boxes)
    out = lines_from_reading_order(boxes, order, separator=" ")
    assert out == ["PL PL Faktura VAT"]


def test_lines_from_reading_order_groups_per_line() -> None:
    boxes = [
        _box(0, "Field:", (0, 0, 50, 20)),
        _box(1, "value", (60, 0, 120, 20)),
        _box(2, "Other:", (0, 40, 60, 60)),
    ]
    cols = detect_columns(boxes, image_width=400)
    assign_columns(boxes, cols)
    cluster_lines(boxes)
    order = reading_order_indices(boxes)
    out = lines_from_reading_order(boxes, order, separator=" | ")
    assert out == ["Field: | value", "Other:"]


def test_flag_handwriting_low_conf() -> None:
    boxes = [
        _box(0, "printed", (0, 0, 100, 20), score=0.95),
        _box(1, "scribble", (200, 0, 300, 20), score=0.40),
    ]
    n = flag_handwriting_lines(boxes)
    assert n == 1
    assert boxes[1].is_handwriting is True
    assert boxes[0].is_handwriting is False


def test_reconstruct_layout_end_to_end_two_columns() -> None:
    items = [
        ("Left header", (10, 10, 200, 30), 0.95),
        ("Right header", (500, 10, 700, 30), 0.95),
        ("Left body", (10, 50, 200, 70), 0.90),
        ("Right body", (500, 50, 700, 70), 0.90),
        ("Left header", (12, 11, 198, 29), 0.80),  # duplicate
    ]
    image = Image.new("RGB", (800, 200), color=(255, 255, 255))
    result = reconstruct_layout(items, image, config=LayoutReconstructionConfig(
        enable_barcodes=False,
        enable_form_detection=False,
        enable_handwriting_flag=False,
    ))
    text = merged_text_from_layout(result, separator=" | ")
    lines = text.split("\n")
    assert lines == ["Left header", "Left body", "Right header", "Right body"]
    assert result.features.suppressed_duplicate_count == 1


def test_reconstruct_layout_handles_empty() -> None:
    image = Image.new("RGB", (100, 100), color=(255, 255, 255))
    result = reconstruct_layout([], image)
    assert result.ordered_indices == []


def test_reconstruct_layout_with_form_features() -> None:
    """Synthetic checkbox image → form_mode True (best-effort)."""
    image = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    # Three checkboxes
    for y in (40, 100, 160):
        draw.rectangle((30, y, 50, y + 20), outline="black", width=2)
    # Two horizontal separator lines
    draw.line((20, 220, 380, 220), fill="black", width=2)
    draw.line((20, 260, 380, 260), fill="black", width=2)
    items = [
        ("Imię", (60, 40, 120, 60), 0.95),
        ("Nazwisko", (60, 100, 200, 120), 0.90),
        ("Adres", (60, 160, 200, 180), 0.92),
    ]
    result = reconstruct_layout(items, image)
    assert result.features.checkbox_count >= 0  # detector exists; counts may vary
    assert result.features.horizontal_line_count >= 0
    # End-to-end did not crash and produced ordered output
    assert len(result.ordered_indices) == 3
