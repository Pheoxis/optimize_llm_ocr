"""Vision LLM prompt and OCR output sanitization."""

from app.services.vision_llm import (
    _OCR_VISION_REPEAT_PENALTY,
    _OCR_VISION_TEMPERATURE,
    _OCR_VISION_TOP_P,
    _PROMPT_OCR_STRICT,
    _ocr_vision_options,
    sanitize_ocr_output,
)


def test_prompt_is_strict_ocr_only() -> None:
    assert "DO NOT interpret meaning" in _PROMPT_OCR_STRICT
    assert "Return ONLY OCR text" in _PROMPT_OCR_STRICT
    assert "NIE poprawiaj" not in _PROMPT_OCR_STRICT


def test_ocr_vision_options_fixed_sampling() -> None:
    opts = _ocr_vision_options()
    assert opts["temperature"] == _OCR_VISION_TEMPERATURE == 0.0
    assert opts["top_p"] == _OCR_VISION_TOP_P == 0.0
    assert opts["repeat_penalty"] == _OCR_VISION_REPEAT_PENALTY == 1.0
    assert opts["top_k"] == 1
    assert opts["seed"] == 0


def test_sanitize_strips_markdown_and_prefix() -> None:
    raw = "Here is the OCR text:\n```\nLine one\nLine two\n```"
    assert sanitize_ocr_output(raw) == "Line one\nLine two"


def test_sanitize_preserves_content() -> None:
    raw = "FV/12/03/2024\nNIP 5261040828"
    assert sanitize_ocr_output(raw) == raw
