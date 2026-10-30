from __future__ import annotations

import pytest

from app.services.document_language import document_language_context
from app.services.polish_text_recovery import (
    apply_invoice_stem_fixes,
    cherrypick_tesseract_tokens,
    has_structural_polish_defect,
    is_polish_document_text,
    merge_ocr_line_polish,
    polish_ocr_needs_verification,
    should_prefer_tesseract,
)


@pytest.mark.parametrize(
    "text",
    [
        "Podgrzanie wody – Jokale užytkowe",
        "Wartošć brutto",
        "llość JM netto",
        "Faktura VAT NIP: 123",
    ],
)
def test_structural_or_domain_detection(text: str) -> None:
    with document_language_context("pl"):
        assert is_polish_document_text(text) or has_structural_polish_defect(text)


def test_invoice_stem_fixes_general() -> None:
    cases = [
        ("ustugi / towaru llość", "usługi / towaru Ilość"),
        ("Powyższa faktura speinia wymogi", "Powyższa faktura spełnia wymogi"),
        ("Data I mlejsce wystawienia", "Data I miejsce wystawienia"),
        ("nieruchomošci Wspólnej", "nieruchomości Wspólnej"),
        ("Zimna woda i šcieków", "Zimna woda i ścieków"),
        ("sześć tysięcy sledemset osiemdziesiat zlotych", "sześć tysięcy siedemset osiemdziesiąt złotych"),
        ("Data wystawienła: 2026-04-17", "Data wystawienia: 2026-04-17"),
        ("Data i miejsce wystawlenla dokumentu", "Data i miejsce wystawienia dokumentu"),
        ("Termin pfatnoścl: 2026-04-17", "Termin płatności: 2026-04-17"),
        ("Platność: Gotówka", "Płatność: Gotówka"),
        ("Kwota oplacona 150,00 PLN", "Kwota opłacona 150,00 PLN"),
        ("PHU OGRODY WLOCHY Sp. z o.o.", "PHU OGRODY WŁOCHY Sp. z o.o."),
        ("Tłumacz Przysięgty Agnieszka", "Tłumacz Przysięgły Agnieszka"),
    ]
    with document_language_context("pl"):
        for raw, expected in cases:
            assert apply_invoice_stem_fixes(raw) == expected


def test_no_client_specific_street_rules() -> None:
    """Recovery must not hard-code one client's address."""
    raw = "ul. Przykładowa 12, Kraków"
    with document_language_context("pl"):
        assert apply_invoice_stem_fixes(raw) == raw


def test_polish_ocr_needs_verification_structural() -> None:
    assert polish_ocr_needs_verification("Jokale užytkowe")
    assert not polish_ocr_needs_verification("Invoice total USD 100")


def test_should_prefer_tesseract_more_diacritics() -> None:
    rapid = "ustugi towaru llość"
    tess = "usługi towaru Ilość"
    with document_language_context("pl"):
        assert should_prefer_tesseract(rapid, tess)


def test_cherrypick_token_from_tesseract() -> None:
    rapid = "Nazwa ustugi llość"
    tess = "Nazwa usługi Ilość"
    with document_language_context("pl"):
        out = cherrypick_tesseract_tokens(rapid, tess)
    assert "usługi" in out
    assert "Ilość" in out


def test_merge_line_polish_applies_stem_fixes_via_postprocess() -> None:
    from app.services.ocr_postprocess import polish_line_output

    with document_language_context("pl"):
        out = polish_line_output("Podgrzanie wody – jokale", "")
    assert "lokale" in out
