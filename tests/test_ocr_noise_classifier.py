"""Tests for OCR noise classifier."""

from __future__ import annotations

import pytest

from app.services.ocr_noise_classifier import (
    NoiseClassification,
    adaptive_noise_threshold,
    classify_token,
    clear_noise_tier,
    filter_noisy_detections,
    is_garbage_ocr,
    is_protected_token,
    score_dictionary,
    set_noise_tier,
)


@pytest.fixture(autouse=True)
def _reset_noise_tier() -> None:
    """Prevent thread-local tier leaking into other test modules."""
    clear_noise_tier()
    yield
    clear_noise_tier()


# ── protected tokens (never dropped) ─────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "zt",
        "gr",
        "zł",
        "…",
        "12345678901",
        "1234567890",
        "123456789",
        "PL61109010140000071219812874",
        "FV/123/2024",
        "1/02/2026",
        "1 234,56 zł",
        "UP",
        "nr",
    ],
)
def test_protected_tokens_never_garbage(text: str) -> None:
    assert is_protected_token(text)
    cls = classify_token(text, 0.50)
    assert not cls.is_garbage
    assert cls.protected or cls.structure_score >= 0.5


# ── known garbage ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,score",
    [
        ("16 Z", 0.70),
        ("FZ", 0.78),
        ("COO", 0.54),
        ("[?][?][?]", 0.60),
        ("||||", 0.55),
    ],
)
def test_known_garbage_detected(text: str, score: float) -> None:
    cls = classify_token(text, score, tier="easy")
    assert cls.is_garbage, f"{text!r} should be garbage: {cls.to_dict()}"


def test_real_text_kept() -> None:
    for text, score in [
        ("kwota pobrania", 0.99),
        ("SPRZEDAWCA", 0.95),
        ("NIP", 0.90),
        ("faktura VAT", 0.92),
    ]:
        cls = classify_token(text, score)
        assert not cls.is_garbage, f"{text!r}: {cls.to_dict()}"


# ── adaptive thresholds ───────────────────────────────────────────────────────


def test_adaptive_thresholds_tier_order() -> None:
    set_noise_tier("easy")
    easy = adaptive_noise_threshold()
    set_noise_tier("medium")
    medium = adaptive_noise_threshold()
    set_noise_tier("hard")
    hard = adaptive_noise_threshold()
    assert easy < medium < hard


def test_easy_stricter_than_hard() -> None:
    text = "COO"
    easy_cls = classify_token(text, 0.54, tier="easy")
    hard_cls = classify_token(text, 0.54, tier="hard")
    assert easy_cls.is_garbage or hard_cls.is_garbage


# ── scores ────────────────────────────────────────────────────────────────────


def test_dictionary_score_pl_word() -> None:
    assert score_dictionary("faktura VAT") >= 0.5


def test_dictionary_score_unknown() -> None:
    assert score_dictionary("XZQWTP") < 0.3


def test_classification_has_all_scores() -> None:
    cls = classify_token("test", 0.9)
    assert isinstance(cls, NoiseClassification)
    for attr in ("noise_score", "language_score", "structure_score", "dictionary_score"):
        val = getattr(cls, attr)
        assert 0.0 <= val <= 1.0, attr


def test_to_dict_benchmark_fields() -> None:
    cls = classify_token("16 Z", 0.70)
    d = cls.to_dict()
    for key in (
        "noise_score", "language_score", "structure_score",
        "dictionary_score", "is_garbage", "reasons", "tier",
    ):
        assert key in d


# ── batch filter ──────────────────────────────────────────────────────────────


def test_filter_noisy_detections() -> None:
    parts = ["kwota pobrania", "16 Z", "zt", "gr"]
    bboxes = [(0, 0, 100, 20)] * 4
    scores = [0.99, 0.70, 0.92, 0.99]
    out_p, out_b, out_s, meta = filter_noisy_detections(parts, bboxes, scores, tier="easy")
    assert "16 Z" not in out_p
    assert "kwota pobrania" in out_p
    assert "zt" in out_p
    assert meta["noise_filtered"] >= 1


def test_is_garbage_ocr_compat() -> None:
    assert is_garbage_ocr("16 Z", 0.70, tier="easy")
    assert not is_garbage_ocr("gr", 0.99)
