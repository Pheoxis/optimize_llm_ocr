"""Universal OCR dedup policy tests."""

from __future__ import annotations

from app.services.ocr_dedup import (
    OcrDedupPolicy,
    is_duplicate_detection,
    is_spurious_ocr_fragment,
    merge_line_detections,
    texts_exact_same,
)


def test_form_column_labels_not_duplicates() -> None:
    pol = OcrDedupPolicy()
    a = (10, 0, 120, 20)
    b = (200, 0, 310, 20)
    assert not is_duplicate_detection(a, b, "kwota pobrania", "kwota pobrania", pol)


def test_overlapping_tile_duplicate() -> None:
    pol = OcrDedupPolicy()
    a = (10, 0, 120, 20)
    b = (15, 2, 118, 22)
    assert is_duplicate_detection(a, b, "kwota pobrania", "kwota pobrania", pol)


def test_touching_same_row_duplicate() -> None:
    pol = OcrDedupPolicy(max_adjacent_gap_px=5)
    a = (10, 0, 100, 20)
    b = (103, 0, 190, 20)
    assert is_duplicate_detection(a, b, "test", "test", pol)


def test_polish_variants_not_duplicates() -> None:
    pol = OcrDedupPolicy()
    a = (10, 0, 200, 20)
    b = (220, 0, 420, 20)
    assert not texts_exact_same("zaptacenia", "zaplacenia")
    assert not is_duplicate_detection(a, b, "zaptacenia", "zaplacenia", pol)


def test_merge_preserves_form_repeats() -> None:
    parts = ["kwota pobrania", "kwota pobrania", "16 Z"]
    bboxes = [(10, 0, 120, 20), (200, 0, 310, 20), (320, 0, 360, 20)]
    scores = [0.9, 0.9, 0.70]
    out = merge_line_detections(parts, bboxes, scores, "    ", image_width=400)
    assert out == "kwota pobrania    kwota pobrania"


def test_merge_tile_echo_polish_variants() -> None:
    pol = OcrDedupPolicy()
    parts = [
        "inne nalezności do zaptacenia.",
        "inne naleznosci do zaplacenia",
    ]
    bboxes = [(726, 1640, 1402, 1742), (727, 1644, 1396, 1744)]
    scores = [0.96, 0.94]
    out = merge_line_detections(
        parts, bboxes, scores, "    ", pol, image_width=3024,
    )
    assert out.count("inne") == 1
    assert "zaptacenia" in out or "zaplacenia" in out


def test_spurious_form_fragments() -> None:
    assert is_spurious_ocr_fragment("16 Z", 0.70)
    assert is_spurious_ocr_fragment("FZ", 0.78)
    assert is_spurious_ocr_fragment("COO", 0.54)
    assert not is_spurious_ocr_fragment("zt", 0.92)
    assert not is_spurious_ocr_fragment("gr", 0.99)
    assert not is_spurious_ocr_fragment("…", 0.55)
    assert not is_spurious_ocr_fragment("kwota pobrania", 0.99)
    assert not is_spurious_ocr_fragment("UP", 0.95)


def test_merge_keeps_zl_dots_gr_drops_unknown() -> None:
    parts = ["kwota pobrania", "16 Z", "zt", "…", "gr"]
    bboxes = [
        (724, 1562, 1079, 1644),
        (1468, 1527, 1792, 1602),
        (1477, 1548, 1533, 1592),
        (1539, 1542, 1739, 1593),
        (2073, 1610, 2135, 1659),
    ]
    scores = [0.99, 0.70, 0.92, 0.55, 0.99]
    out = merge_line_detections(parts, bboxes, scores, "    ", image_width=3024)
    assert out == "kwota pobrania    zt    …    gr"


def test_merge_substring_echo() -> None:
    pol = OcrDedupPolicy()
    parts = ["adresowana do", "adresowana"]
    bboxes = [(654, 1784, 1017, 1857), (654, 1784, 975, 1857)]
    scores = [0.97, 0.95]
    out = merge_line_detections(parts, bboxes, scores, "    ", pol, image_width=3024)
    assert out == "adresowana do"
