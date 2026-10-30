"""Post-processing corrections for Polish OCR output.

Domain-specific recovery (invoice stems, diacritics, engine merge) lives in
``polish_text_recovery``.  This module adds field-level fixes (NIP, IBAN,
currency) and wires the pipeline entry points.
"""
from __future__ import annotations

import re

from app.services.polish_text_recovery import (
    apply_caron_and_script_fixes,
    apply_invoice_stem_fixes,
    cherrypick_tesseract_tokens,
    is_polish_document_text,
    merge_ocr_line_polish,
    polish_ocr_needs_verification,
)

# Re-export for callers that imported from here.
__all__ = [
    "correct_polish_ocr",
    "correct_polish_ocr_line",
    "polish_line_output",
    "polish_ocr_needs_verification",
]


def _fix_nip(text: str) -> str:
    """Fix NIP (Polish tax ID) fields: NIP must be 10 digits, never letters."""
    _DIGIT_SUB: dict[str, str] = {
        "B": "8", "O": "0", "D": "0", "I": "1",
        "Q": "0", "S": "5", "Z": "2", "G": "6",
        "A": "4",
    }

    def _sub_nip(m: re.Match) -> str:
        label = m.group(1)
        digits_raw = m.group(2)
        out: list[str] = []
        for ch in digits_raw:
            if ch.isdigit() or ch in "- ":
                out.append(ch)
            elif ch.upper() in _DIGIT_SUB:
                out.append(_DIGIT_SUB[ch.upper()])
            else:
                out.append(ch)
        return label + "".join(out)

    text = re.sub(
        r"(NIP\s*[:\s]\s*)([\dA-Za-z\-]{9,14})",
        _sub_nip,
        text,
    )
    return text


def _fix_iban(text: str) -> str:
    """Fix IBAN / Polish bank account numbers."""
    _DIGIT_SUB: dict[str, str] = {
        "B": "8", "O": "0", "D": "0", "I": "1",
        "Q": "0", "S": "5", "Z": "2", "G": "6",
    }

    def _correct_account(raw: str) -> str:
        return "".join(
            _DIGIT_SUB.get(ch.upper(), ch) if not ch.isdigit() and ch not in (" ", "-") else ch
            for ch in raw
        )

    def _sub_account(m: re.Match) -> str:
        return m.group(1) + _correct_account(m.group(2))

    text = re.sub(
        r"((?:[Nn]r\s+(?:konta|rachunku)|konto|IBAN)[:\s]+)"
        r"((?:PL)?[\dA-Za-z][\dA-Za-z ]{18,34})",
        _sub_account,
        text,
    )
    text = re.sub(
        r"\b(\d{2}(?:\s?\d{4}){6})\b",
        lambda m: _correct_account(m.group(0)),
        text,
    )
    return text


def _fix_company_suffix(text: str) -> str:
    """Fix Sp. z o.o. and SPÓŁKA Z OGRANICZONĄ OCR variants."""
    text = re.sub(r"\bSp\.\s*[Zz]\s+0\.0\.", "Sp. z o.o.", text)
    text = re.sub(r"\bSp\.\s*[Zz]\s+o\.0\.", "Sp. z o.o.", text)
    text = re.sub(r"\bSp\.\s*[Zz]\s+0\.o\.", "Sp. z o.o.", text)
    text = re.sub(r"OGRANICZUNA\b", "OGRANICZONĄ", text, flags=re.IGNORECASE)
    return text


def _fix_currency(text: str) -> str:
    """Fix zł / zlotych OCR variants in amounts and table headers."""
    text = re.sub(r"(\d)\s+zt\b", r"\1 zł", text)
    text = re.sub(r"(\d)zt\b", r"\1 zł", text)
    text = re.sub(r"\bzlotych\b", "złotych", text, flags=re.IGNORECASE)
    text = re.sub(r"\bzloty\b", "złoty", text, flags=re.IGNORECASE)
    text = re.sub(r"\bzl\b", "zł", text, flags=re.IGNORECASE)
    text = re.sub(r"\(zt\)", "(zł)", text, flags=re.IGNORECASE)
    text = re.sub(r"\(zI\)", "(zł)", text)
    text = re.sub(r"\(z1\)", "(zł)", text)
    text = re.sub(r"netto\s*\(z\)", "netto (zł)", text, flags=re.IGNORECASE)
    text = re.sub(r"brutto\s*\(z\)", "brutto (zł)", text, flags=re.IGNORECASE)
    return text


def _fix_date_labels(text: str) -> str:
    """Normalise common Polish invoice date field labels."""
    text = re.sub(r"\bData\s+sprzedazy\b", "Data sprzedaży", text, flags=re.IGNORECASE)
    text = re.sub(r"\bTermin\s+platnosci\b", "Termin płatności", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSposob\s+platnosci\b", "Sposób płatności", text, flags=re.IGNORECASE)
    return text


def _fix_pl_patterns(text: str) -> str:
    """Structural invoice patterns not covered by stem list."""
    if not is_polish_document_text(text):
        return text
    text = re.sub(r"(?m)^r konta:", "Numer konta:", text)
    text = re.sub(r"\bemzeN\b", "Nazwa", text)
    text = re.sub(r"\bNazwa:\b", "Nazwa", text)
    text = re.sub(r"\bNumer\s+K[Ss]e[FfĘę]\b", "Numer KSeF", text)
    return text


def _apply_polish_corrections(text: str) -> str:
    text = apply_caron_and_script_fixes(text)
    text = _fix_nip(text)
    text = _fix_iban(text)
    text = _fix_company_suffix(text)
    text = _fix_currency(text)
    text = apply_invoice_stem_fixes(text)
    text = _fix_date_labels(text)
    text = _fix_pl_patterns(text)
    return text


def correct_polish_ocr_line(text: str) -> str:
    """Polish post-corrections for a single OCR line (no minimum length)."""
    if not text or not text.strip():
        return text
    return _apply_polish_corrections(text)


def correct_polish_ocr(text: str) -> str:
    """Apply all Polish OCR post-corrections (full document / block)."""
    if not text or len(text) < 20:
        return text
    return _apply_polish_corrections(text)


def polish_line_output(rapid: str, tess: str = "") -> str:
    """Per-line: merge engines, then apply Polish corrections."""
    base = merge_ocr_line_polish(rapid, tess)
    if tess and tess.strip():
        base = cherrypick_tesseract_tokens(base, tess.strip())
    return correct_polish_ocr_line(base)
