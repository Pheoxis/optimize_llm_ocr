"""Crop-level vision escalation (no full-page VLM)."""

from __future__ import annotations

from unittest.mock import patch

from PIL import Image

from app.core.config import settings
from app.services.crop_escalation import (
    _select_tesseract_verification_indices,
    build_crop_batch,
    split_crop_text_for_lines,
    union_bbox,
)
from app.services.fast_ocr import FastOCRLine, FastOCRResult


def test_union_bbox_with_padding() -> None:
    bbox = union_bbox([(10, 20, 50, 40), (55, 22, 90, 42)], pad=5, image_size=(200, 100))
    assert bbox == (5, 15, 95, 47)


def test_split_crop_text_for_lines() -> None:
    assert split_crop_text_for_lines("A\nB\nC", 3) == ["A", "B", "C"]
    assert split_crop_text_for_lines("only one", 1) == ["only one"]
    assert split_crop_text_for_lines("a\nb\nc\nd", 2) == ["a", "b c d"]


def test_build_crop_batch_merges_adjacent() -> None:
    lines = [
        FastOCRLine("a", 0.5, 0.4, 1, (0, 0, 10, 10)),
        FastOCRLine("b", 0.5, 0.4, 1, (0, 12, 10, 22)),
        FastOCRLine("c", 0.99, 0.99, 1, (0, 40, 10, 50)),
    ]
    flags = [(True, "hard"), (True, "hard"), (False, "none")]
    with patch.object(settings, "ocr_crop_merge_adjacent_lines", True):
        batch = build_crop_batch(lines, flags, image_size=(100, 60))
    assert len(batch) == 1
    assert batch[0].line_indices == (0, 1)
    assert batch[0].rapid_texts == ("a", "b")


def test_select_tesseract_verification_indices_budget() -> None:
    lines = [
        FastOCRLine("a", 0.30, 0.20, 1, (0, 0, 10, 10)),
        FastOCRLine("b", 0.95, 0.90, 1, (0, 12, 10, 22)),
        FastOCRLine("c", 0.60, 0.50, 1, (0, 24, 10, 34)),
    ]
    with patch.object(settings, "ocr_line_tesseract_max_lines_per_page", 2):
        selected = _select_tesseract_verification_indices(lines)
    assert selected == {0, 2}


def test_apply_crop_escalation_preserves_line_order() -> None:
    from app.services.crop_escalation import CropBatchItem, apply_crop_escalation

    fast = FastOCRResult(
        text="GOOD\nBAD",
        lines=[
            FastOCRLine("GOOD", 0.99, 0.98, 1, (0, 0, 40, 12)),
            FastOCRLine("BAD", 0.4, 0.3, 1, (0, 14, 40, 26)),
        ],
        mean_score=0.7,
        min_score=0.3,
        low_conf_ratio=0.5,
        n_boxes=2,
        elapsed_sec=0.1,
    )
    image = Image.new("RGB", (50, 30), color=(255, 255, 255))
    batch = [CropBatchItem(line_indices=(1,), bbox=(0, 14, 40, 26), rapid_texts=("BAD",))]
    with (
        patch(
            "app.services.crop_escalation.build_crop_batch",
            return_value=batch,
        ),
        patch(
            "app.services.crop_escalation.process_crop_batch",
            return_value={1: ("FIXED", 0.95, "vision_llm", "vlm_crop", True)},
        ),
        patch.object(settings, "ocr_line_verification_enabled", False),
        patch.object(settings, "ocr_line_skip_tesseract_above", 1.0),
        patch.object(settings, "ocr_consensus_enabled", False),
    ):
        result = apply_crop_escalation(
            image, fast, "eng", allow_vlm=True, page_quality_poor=False,
        )

    assert result.text == "GOOD\nFIXED"
    assert result.line_results[0].text == "GOOD"
    assert result.line_results[1].text == "FIXED"
    assert result.line_results[1].escalated is True
