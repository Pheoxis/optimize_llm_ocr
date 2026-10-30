"""PL/EN Tesseract language routing (verification layer)."""

from __future__ import annotations

import pytest

from app.services import ocr_lang_routing as routing
from app.services.ocr_lang_routing import (
    languages_detected_from_text,
    polish_chars_detected,
    resolve_tesseract_lang,
)


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    routing.reset_ocr_lang_routing_metrics()
    yield
    routing.reset_ocr_lang_routing_metrics()


def test_polish_chars_detected() -> None:
    assert polish_chars_detected("Faktura VAT — Łódź")
    assert not polish_chars_detected("Invoice total USD 100")


def test_resolve_tesseract_lang_polish_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routing.settings, "ocr_lang_routing_enabled", True)
    monkeypatch.setattr(routing.settings, "ocr_lang_tesseract_polish", "pol+eng")
    monkeypatch.setattr(routing.settings, "ocr_lang_tesseract_english", "eng")
    monkeypatch.setattr(routing.settings, "ocr_lang_polish_char_min", 2)

    assert resolve_tesseract_lang("ąę") == "pol+eng"
    assert resolve_tesseract_lang("Hello world") == "eng"


def test_resolve_tesseract_lang_routing_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routing.settings, "ocr_lang_routing_enabled", False)
    monkeypatch.setattr(routing.settings, "ocr_lang", "pol+eng")

    assert resolve_tesseract_lang("Invoice") == "pol+eng"
    assert resolve_tesseract_lang("ąę") == "pol+eng"


def test_languages_detected_from_text() -> None:
    assert languages_detected_from_text("Warszawa, ul. Żurawia, Łódź") == ["pl", "en"]
    assert languages_detected_from_text("Receipt #1234") == ["en"]


def test_routing_metrics_increment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routing.settings, "ocr_lang_routing_enabled", True)
    resolve_tesseract_lang("ąęł")
    resolve_tesseract_lang("abc")
    m = routing.get_ocr_lang_routing_metrics()
    assert m.samples == 2
    assert m.polish_routed == 1
    assert m.english_routed == 1
