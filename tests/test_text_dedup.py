"""Overlap-only OCR dedup — preserves intentional form duplicates."""

from __future__ import annotations

from app.services.layout.text_dedup import (
    merge_overlapping_box_texts,
    pick_better_text,
    texts_exact_same,
)


def test_texts_exact_same_preserves_polish_diacritics() -> None:
    assert texts_exact_same("zaptacenia", "zaptacenia")
    assert not texts_exact_same("zaptacenia", "zaplacenia")
    assert not texts_exact_same("ąę", "ae")


def test_pick_better_text_prefers_polish() -> None:
    assert pick_better_text("zaplacenia", "zapłacenia") == "zapłacenia"


def test_merge_keeps_separated_form_labels() -> None:
    """Two 'kwota pobrania' at different x — both kept."""
    parts = ["kwota pobrania", "kwota pobrania"]
    bboxes = [(10, 0, 120, 20), (200, 0, 310, 20)]
    scores = [0.9, 0.9]
    out = merge_overlapping_box_texts(parts, bboxes, scores, "    ")
    assert out == "kwota pobrania    kwota pobrania"


def test_merge_drops_overlapping_tile_duplicate() -> None:
    parts = ["kwota pobrania", "kwota pobrania"]
    bboxes = [(10, 0, 120, 20), (15, 2, 118, 22)]
    scores = [0.8, 0.95]
    out = merge_overlapping_box_texts(parts, bboxes, scores, "    ")
    assert out == "kwota pobrania"


def test_merge_keeps_different_polish_variants_on_same_line() -> None:
    parts = ["inne nalezności do zaptacenia.", "inne naleznosci do zaplacenia"]
    bboxes = [(10, 0, 200, 20), (220, 0, 420, 20)]
    scores = [0.9, 0.88]
    out = merge_overlapping_box_texts(parts, bboxes, scores, "    ")
    assert "zaptacenia" in out and "zaplacenia" in out
