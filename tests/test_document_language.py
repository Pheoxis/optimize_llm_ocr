from __future__ import annotations

import pytest

from app.services.document_language import (
    DocumentLanguage,
    document_language_context,
    get_document_language,
    languages_for_result,
    parse_document_language,
)
from app.services.ocr_lang_routing import resolve_tesseract_lang


def test_parse_document_language_aliases() -> None:
    assert parse_document_language("auto") == DocumentLanguage.AUTO
    assert parse_document_language("polski") == DocumentLanguage.PL
    assert parse_document_language("angielski") == DocumentLanguage.EN


def test_parse_document_language_invalid() -> None:
    with pytest.raises(ValueError):
        parse_document_language("de")


def test_language_hint_forces_tesseract_lang(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.ocr_lang_routing.settings.ocr_lang_tesseract_polish", "pol+eng")
    monkeypatch.setattr("app.services.ocr_lang_routing.settings.ocr_lang_tesseract_english", "eng")

    with document_language_context("pl"):
        assert resolve_tesseract_lang("Hello") == "pol+eng"
    with document_language_context("en"):
        assert resolve_tesseract_lang("ąę") == "eng"


def test_languages_for_result_respects_hint() -> None:
    with document_language_context("pl"):
        assert languages_for_result("anything") == ["pl", "en"]
    with document_language_context("en"):
        assert languages_for_result("Łódź") == ["en"]
    with document_language_context("auto"):
        assert get_document_language() == DocumentLanguage.AUTO
