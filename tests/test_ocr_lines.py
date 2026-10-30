"""Structured OCR line output."""

from __future__ import annotations

import json

from app.models.schemas import OcrLine, ProcessingResult
from app.services.fast_ocr import FastOCRLine, FastOCRResult
from app.services.ocr_lines import (
    bbox_crop_box,
    line_dicts_from_fast_result,
    merge_lines_text,
    parse_lines_from_meta,
)
from datetime import UTC, datetime


def test_fast_result_to_ocr_lines() -> None:
    fast = FastOCRResult(
        text="Hello\nWorld",
        lines=[
            FastOCRLine(
                text="Hello",
                mean_score=0.95,
                min_score=0.9,
                n_boxes=1,
                bbox=(10, 20, 100, 40),
            ),
            FastOCRLine(
                text="World",
                mean_score=0.88,
                min_score=0.8,
                n_boxes=1,
                bbox=(10, 50, 90, 70),
            ),
        ],
        mean_score=0.91,
        min_score=0.8,
        low_conf_ratio=0.0,
        n_boxes=2,
        elapsed_sec=0.1,
    )
    lines = line_dicts_from_fast_result(fast, page=2, engine="fast_ocr")
    assert len(lines) == 2
    assert lines[0]["text"] == "Hello"
    assert lines[0]["bbox"] == [10, 20, 100, 40]
    assert lines[0]["page"] == 2
    assert lines[0]["engine"] == "fast_ocr"
    assert lines[0]["confidence"] == 0.95


def test_parse_lines_from_meta_line_smart_shape() -> None:
    meta = {
        "lines": [
            {
                "text": "Line A",
                "confidence": 0.92,
                "bbox": (5, 5, 50, 25),
                "engine": "consensus",
                "escalated": True,
            },
        ],
    }
    parsed = parse_lines_from_meta(meta, default_page=1)
    assert len(parsed) == 1
    assert parsed[0].text == "Line A"
    assert parsed[0].escalated is True
    assert parsed[0].bbox == [5, 5, 50, 25]


def test_processing_result_backward_compatible_json() -> None:
    """Legacy JSON without ``lines`` still loads; export includes lines array."""
    legacy = {
        "source": "a.pdf",
        "source_kind": "pdf",
        "processed_at": datetime.now(UTC).isoformat(),
        "page_count": 1,
        "language_detected": ["pol"],
        "text": "only merged text",
        "confidence": 0.9,
        "engine": "fast_ocr",
        "warnings": [],
        "metadata": {},
    }
    result = ProcessingResult.model_validate(legacy)
    assert result.text == "only merged text"
    assert result.lines == []

    result.lines = [
        OcrLine(
            text="row",
            confidence=0.99,
            bbox=[0, 0, 10, 10],
            page=1,
            engine="fast_ocr",
        ),
    ]
    dumped = json.loads(result.model_dump_json())
    assert dumped["text"] == "only merged text"
    assert len(dumped["lines"]) == 1
    assert dumped["lines"][0]["bbox"] == [0, 0, 10, 10]


def test_merge_lines_text_matches_fast_join() -> None:
    lines = [
        OcrLine(text="A", confidence=1.0, bbox=[0, 0, 1, 1], page=1, engine="fast_ocr"),
        OcrLine(text="B", confidence=1.0, bbox=[0, 0, 1, 1], page=1, engine="fast_ocr"),
    ]
    assert merge_lines_text(lines) == "A\nB"


def test_bbox_crop_box_with_pad() -> None:
    line = OcrLine(
        text="x",
        confidence=1.0,
        bbox=[10, 20, 30, 40],
        page=1,
        engine="fast_ocr",
    )
    assert bbox_crop_box(line, pad=5, image_width=100, image_height=100) == (
        5, 15, 35, 45,
    )
