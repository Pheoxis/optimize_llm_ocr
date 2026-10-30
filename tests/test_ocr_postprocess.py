from __future__ import annotations

import pytest

from app.services.document_language import document_language_context
from app.services.ocr_postprocess import (
    correct_polish_ocr,
    correct_polish_ocr_line,
    polish_line_output,
    polish_ocr_needs_verification,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Podgrzanie wody – Jokale užytkowe", "Podgrzanie wody – lokale użytkowe"),
        ("Powyższa faktura speinia wymogi faktury handlowej", "Powyższa faktura spełnia wymogi faktury handlowej"),
        ("Data I mlejsce wystawienia dokumentu", "Data I miejsce wystawienia dokumentu"),
        ("Lp. Symbol Nazwa ustugi / towaru    llość", "Lp. Symbol Nazwa usługi / towaru    Ilość"),
        ("Utrzymanie nieruchomošci Wspólnej", "Utrzymanie nieruchomości Wspólnej"),
        ("Zimna woda i odprowadzanie šcieków - jokale", "Zimna woda i odprowadzanie ścieków - lokale"),
        ("Wartošć brutto    91,98 PLN", "Wartość brutto    91,98 PLN"),
        ("sześć tysięcy sledemset osiemdziesiat osiem zlotych", "sześć tysięcy siedemset osiemdziesiąt osiem złotych"),
        ("Termin platnosci 10 dni", "Termin płatności 10 dni"),
        ("NIP: 5223193889 Do zaplaty 100 zt", "NIP: 5223193889 Do zapłaty 100 zł"),
        ("Data wystawienła: Mościska, 2026-04-17", "Data wystawienia: Mościska, 2026-04-17"),
        ("Termin pfatnoścl: 2026-04-17", "Termin płatności: 2026-04-17"),
        ("Platność: Gotówka", "Płatność: Gotówka"),
    ],
)
def test_polish_line_fixes(raw: str, expected: str) -> None:
    with document_language_context("pl"):
        assert correct_polish_ocr_line(raw) == expected


def test_polish_line_output_cherrypicks_from_tesseract() -> None:
    rapid = "Lp. Symbol Nazwa ustugi / towaru    llość    JM"
    tess = "Lp. Symbol Nazwa usługi / towaru Ilość JM"
    with document_language_context("pl"):
        out = polish_line_output(rapid, tess)
    assert "Ilość" in out
    assert "usługi" in out
    assert "llość" not in out


def test_polish_ocr_needs_verification_detects_bad_chars() -> None:
    assert polish_ocr_needs_verification("Podgrzanie wody – Jokale užytkowe")
    assert polish_ocr_needs_verification("Wartošć brutto")
    assert not polish_ocr_needs_verification("Invoice total USD 100")


def test_full_document_postprocess_preserves_english() -> None:
    text = "Invoice number 12345 total amount USD 100.00"
    assert correct_polish_ocr(text) == text


def test_no_hardcoded_client_address() -> None:
    raw = "02-401 Warszawa ul. Testowa 5"
    with document_language_context("pl"):
        assert correct_polish_ocr_line(raw) == raw
