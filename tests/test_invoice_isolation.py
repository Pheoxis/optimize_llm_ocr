"""Tests for first-invoice OCR text isolation."""

from __future__ import annotations

from app.services.invoice_extraction import _isolate_first_invoice_text


def test_isolate_first_invoice_cuts_paragon() -> None:
    text = (
        "Faktura 1/4\nFAKTURA\nNumer: FV/1\n"
        "Sprzedawca: ABC\nNabywca: XYZ\n"
        "SUMA PLN 100.00\n\n"
        "Paragon 1/5\nPARAGON FISKALNY\nsklep"
    )
    out = _isolate_first_invoice_text(text)
    assert "FV/1" in out
    assert "Paragon" not in out
    assert "PARAGON" not in out


def test_isolate_keeps_single_invoice() -> None:
    text = "FAKTURA\nNumer: X\nKoniec"
    assert _isolate_first_invoice_text(text) == text
