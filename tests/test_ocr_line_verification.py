"""Line-level Rapid vs Tesseract disagreement verification."""

from __future__ import annotations

from unittest.mock import patch

from PIL import Image

from app.core.config import settings
from app.services.fast_ocr import FastOCRLine, FastOCRResult
from app.services.ocr_line_verification import (
    verify_fast_result_lines,
    verify_line_crop,
)


def test_verify_line_crop_marks_suspicious() -> None:
    line = FastOCRLine(
        text="ABC 123",
        mean_score=0.8,
        min_score=0.7,
        n_boxes=1,
        bbox=(0, 0, 50, 20),
    )
    crop = Image.new("RGB", (50, 20), color=(255, 255, 255))
    with (
        patch(
            "app.services.ocr_line_verification.tesseract_line_crop",
            return_value="XYZ 999",
        ),
        patch.object(settings, "ocr_line_verification_skip_high_conf_tesseract", False),
        patch.object(settings, "ocr_line_suspicious_threshold", 0.2),
    ):
        result = verify_line_crop(crop, line, "eng", line_index=0)
    assert result.suspicious is True
    assert result.disagreement > 0.2
    assert result.rapid_text == "ABC 123"
    assert result.tesseract_text == "XYZ 999"
    assert result.escalation_hint == "suspicious_disagreement"


def test_verify_line_crop_agreement_not_suspicious() -> None:
    line = FastOCRLine(
        text="SAME",
        mean_score=0.9,
        min_score=0.85,
        n_boxes=1,
        bbox=(0, 0, 40, 15),
    )
    crop = Image.new("RGB", (40, 15), color=(255, 255, 255))
    with (
        patch(
            "app.services.ocr_line_verification.tesseract_line_crop",
            return_value="SAME",
        ),
        patch.object(settings, "ocr_line_verification_skip_high_conf_tesseract", False),
        patch.object(settings, "ocr_line_suspicious_threshold", 0.4),
    ):
        result = verify_line_crop(crop, line, "eng", line_index=0)
    assert result.suspicious is False
    assert result.disagreement == 0.0


def test_verify_line_crop_semantic_validation_escalates() -> None:
    line = FastOCRLine(
        text="PESEL 44051401450",
        mean_score=0.95,
        min_score=0.9,
        n_boxes=1,
        bbox=(0, 0, 80, 20),
    )
    crop = Image.new("RGB", (80, 20), color=(255, 255, 255))
    with (
        patch(
            "app.services.ocr_line_verification.tesseract_line_crop",
            return_value="PESEL 44051401450",
        ),
        patch.object(settings, "ocr_line_verification_skip_high_conf_tesseract", False),
        patch.object(settings, "ocr_semantic_validation_enabled", True),
        patch.object(settings, "ocr_semantic_validation_escalate", True),
    ):
        result = verify_line_crop(crop, line, "eng", line_index=0)
    assert result.validation_passed is False
    assert result.escalation_tier == "hard"
    assert result.escalation_hint and "semantic_pesel" in result.escalation_hint


def test_verify_fast_result_lines_metrics() -> None:
    fast = FastOCRResult(
        text="A\nB",
        lines=[
            FastOCRLine("A", 0.9, 0.85, 1, (0, 0, 10, 10)),
            FastOCRLine("B", 0.9, 0.85, 1, (0, 12, 10, 22)),
        ],
        mean_score=0.9,
        min_score=0.85,
        low_conf_ratio=0.0,
        n_boxes=2,
        elapsed_sec=0.1,
    )
    image = Image.new("RGB", (20, 30), color=(255, 255, 255))
    with (
        patch(
            "app.services.ocr_line_verification.tesseract_line_crop",
            side_effect=["A", "DIFFERENT"],
        ),
        patch.object(settings, "ocr_line_verification_skip_high_conf_tesseract", False),
        patch.object(settings, "ocr_line_suspicious_threshold", 0.3),
    ):
        results, metrics = verify_fast_result_lines(image, fast, "eng")
    assert len(results) == 2
    assert metrics.total_lines == 2
    assert metrics.tess_ran == 2
    assert metrics.suspicious_count >= 1


def test_verify_line_crop_force_skip_tesseract() -> None:
    line = FastOCRLine(
        text="ABC 123",
        mean_score=0.2,
        min_score=0.1,
        n_boxes=1,
        bbox=(0, 0, 50, 20),
    )
    crop = Image.new("RGB", (50, 20), color=(255, 255, 255))
    with patch(
        "app.services.ocr_line_verification.tesseract_line_crop",
        return_value="SHOULD_NOT_BE_USED",
    ) as tess_mock:
        result = verify_line_crop(
            crop,
            line,
            "eng",
            line_index=0,
            force_skip_tesseract=True,
        )
    tess_mock.assert_not_called()
    assert result.tess_skipped is True
    assert result.tesseract_text == ""
