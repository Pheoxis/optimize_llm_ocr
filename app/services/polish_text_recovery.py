"""General Polish OCR recovery — structural patterns, not per-client strings."""

from __future__ import annotations

import re

from app.services.ocr_dedup import pick_better_text, polish_diacritic_score

# Polish accounting / invoice vocabulary (domain gate, not client-specific).
_PL_DOC_HINT = re.compile(
    r"\b("
    r"faktura|nota|paragon|rachunek|NIP|REGON|KRS|VAT|KSeF|"
    r"zł|zapłat|płatno|sprzedaż|nabywca|sprzedawca|"
    r"netto|brutto|grosz|PLN|przelew|kwota|"
    r"usług|towar|dostaw|faktur|not[ayę]"
    r")\b",
    re.IGNORECASE,
)

# Structural OCR defects (engine confusions), not named entities.
_POLISH_STRUCTURAL_DEFECT = re.compile(
    r"(?i)"
    r"([šžñň]"
    r"|llo[śs]"
    r"|\b[jJfF]okale\b"
    r"|spe[i1l]nia\b"
    r"|m[l1i]ejsce\b"
    r"|wystawien[lł][aáà]"
    r"|wystawlenl[aá]"
    r"|pfatno[sš]"
    r"|\bPlatno[sš][cć]"
    r"|\b[oO]plac"
    r"|Przysięgt"
    r"|\bWLOCHY\b"
    r"|\bImie\b"
    r"|wart[oó][sš][cć]"
    r"|nieruchomo[sš]ci\b"
    r"|\bustugi\b"
    r"|\buslugl\b"
    r"|\buslug\b"
    r"|zako[ñnň]czen"
    r"|osiemdziesiat\b"
    r"|sledemset\b"
    r"|zlotych\b"
    r"|užytkow"
    r"|w[i1l]ochy\b"
    r"|\(zt\)"
    r"|\(z[i1l]\))"
)

# High-frequency invoice stems: OCR drops ł, ś, ż, ó, ą, ę, ń.
_INVOICE_STEM_FIXES: list[tuple[str, str]] = [
    # płatność family (ł → l/t/h/1)
    (r"\bp[tlh1i]atno[sś][cć]\b", "płatność"),
    (r"\bp[tlh1i]atno[sś]ci\b", "płatności"),
  # zapłat-
    (r"\bZaplac", "Zapłac"),
    (r"\bzaplac", "zapłac"),
    (r"\bZaplata\b", "Zapłata"),
    (r"\bzaplata\b", "zapłata"),
    (r"\bZaplaty\b", "Zapłaty"),
    (r"\bzaplaty\b", "zapłaty"),
    (r"\bDo\s+zaplaty\b", "Do zapłaty"),
    (r"\bdo\s+zaplaty\b", "do zapłaty"),
    # Wartość / wartość (ś → s/š, ó → o)
    (r"\bWart[oó][sš]?[cć]\b", "Wartość"),
    (r"\bwart[oó][sš]?[cć]\b", "wartość"),
    (r"\bWartosci\b", "Wartości"),
    (r"\bwartosci\b", "wartości"),
    # Ilość (I → l)
    (r"\bllo[śs][cćé]?\b", "Ilość"),
    (r"\bIlos[cć]\b", "Ilość"),
    (r"\bilos[cć]\b", "ilość"),
    # usługi / usług
    (r"\bustugi\b", "usługi"),
    (r"\buslugl\b", "usługi"),
    (r"\buslugi\b", "usługi"),
    (r"\buslug\b", "usług"),
    (r"\buslug:", "usług:"),
    # lokale (leading J/f/l confusion)
    (r"\b[jJfF]okale\b", "lokale"),
    (r"\blokate\b", "lokale"),
    # miejsce
    (r"\bm[l1i]ejsce\b", "miejsce"),
    (r"\bmejsce\b", "miejsce"),
    # wystawienia (ł/i/l/n confusion in "wystawienia")
    (r"\bwystawienła\b", "wystawienia"),
    (r"\bwystawien[ií]a\b", "wystawienia"),
    (r"\bwystawienl[aáà]\b", "wystawienia"),
    (r"\bwystawienla\b", "wystawienia"),
    (r"\bwystawlenla\b", "wystawienia"),
    (r"\bwystawlenia\b", "wystawienia"),
    # płatność (p/f instead of ł; ści read as ścl / sci)
    (r"\bptatno[sšś][cć]t?\b", "płatności"),
    (r"\bpła[zZ]ności\b", "płatności"),
    (r"\bp[f1]atnoścl\b", "płatności"),
    (r"\bp[f1]atności\b", "płatności"),
    (r"\bp[f1]atnosci\b", "płatności"),
    (r"\bp[f1]atno[sš][cć][cl1í]+\b", "płatności"),
    (r"\bp[f1]atno[sš][cć]\b", "płatność"),
    (r"\bPlatno[śsš][cć]\b", "Płatność"),
    (r"\bplatno[śsš][cć]\b", "płatność"),
    (r"\bPlatnosc\b", "Płatność"),
    (r"\bplatnosc\b", "płatność"),
    # opłacono / opłacona
    (r"\bKwota\s+oplacona\b", "Kwota opłacona"),
    (r"\bOplacona\b", "Opłacona"),
    (r"\boplacona\b", "opłacona"),
    (r"\bOPLACONO\b", "OPŁACONO"),
    (r"\bOPLACONC\b", "OPŁACONO"),
    # spełnia / wypełnia / zapełnia (ł → i/l)
    (r"\b(sp|wy|za|do)pe[i1l]nia\b", r"\1pełnia"),
    (r"\bspe[i1l]nia\b", "spełnia"),
    # zakończenia, opłaty, ciepło
    (r"\bzako[ñnň]czenia\b", "zakończenia"),
    (r"\bOplata\b", "Opłata"),
    (r"\bOplaty\b", "Opłaty"),
    (r"\bOplata\b", "Opłata"),
    (r"\bmiesiac\b", "miesiąc"),
    (r"\bcieplo\b", "ciepło"),
    # nieruchomości / ścieków
    (r"\bnieruchomo[sš]ci\b", "nieruchomości"),
    (r"\b[sš]ciek[oó]w\b", "ścieków"),
    # sprzedaż / słownie / sposób
    (r"\bSprzedaz\b", "Sprzedaż"),
    (r"\bsprzedaz\b", "sprzedaż"),
    (r"\bS[lth1i]ownie\b", "Słownie"),
    (r"\bs[lth1i]ownie\b", "słownie"),
    (r"\bSposob\b", "Sposób"),
    (r"\bsposob\b", "sposób"),
]

_TLUMACZ_FIXES: list[tuple[str, str]] = [
    (r"\bT[lth1i]umaczenie\b", "Tłumaczenie"),
    (r"\bt[lth1i]umaczenie\b", "tłumaczenie"),
    (r"\bT[lth1i]umaczenie\b", "Tłumaczenie"),
    (r"\bt[lth1i]umaczenie\b", "tłumaczenie"),
    (r"\bT[lth1i]umacza\b", "Tłumacza"),
    (r"\bT[lth1i]umaczem\b", "Tłumaczem"),
    (r"\bT[lth1i]umacz\b", "Tłumacz"),
    (r"\bt[lth1i]umacz\b", "tłumacz"),
]

# Amount words in "Słownie:" blocks
_AMOUNT_WORD_FIXES: list[tuple[str, str]] = [
    (r"\bsledemset\b", "siedemset"),
    (r"\bosiemdziesiat\b", "osiemdziesiąt"),
    (r"\bsze[sś]ćdzlesiat\b", "sześćdziesiąt"),
    (r"\bsze[sś]ćdziesiat\b", "sześćdziesiąt"),
    (r"\bsze[sś]ćdziesi[aą]t\b", "sześćdziesiąt"),
    (r"\btyslace\b", "tysiące"),
    (r"\bzlotych\b", "złotych"),
    (r"\bzloty\b", "złoty"),
    (r"\bdziesiat\b", "dziesiąt"),
    (r"\bczternascie\b", "czternaście"),
    (r"\bdwiescie\b", "dwieście"),
]

# ł → i/l in common adjective endings (-owski/-owska family is risky; keep narrow)
_WLOCHY_FIXES: list[tuple[str, str]] = [
    (r"\bWLOCHY\b", "WŁOCHY"),
    (r"\bWtOCHY\b", "Włochy"),
    (r"\bW[i1l]ochy\b", "Włochy"),
    (r"\bw[i1l]ochy\b", "włochy"),
]

_MISC_INVOICE_FIXES: list[tuple[str, str]] = [
    (r"\bPozostalo\b", "Pozostało"),
    (r"\bpozostalo\b", "pozostało"),
    (r"\bImi[eę]q\b", "Imię"),
    (r"\bimiq\b", "imię"),
    (r"\bupowaznionej\b", "upoważnionej"),
    (r"\bupowaznion\w*\b", "upoważnion"),
    (r"\bPrzysięgt[yý]\b", "Przysięgły"),
    (r"\bprzysięgt[yý]\b", "przysięgły"),
    (r"\bPrzysieg[lt]y\b", "Przysięgły"),
    (r"\bprzysieg[lt]y\b", "przysięgły"),
    (r"\bPrzysieg[lt]a\b", "Przysięgła"),
    (r"\bImie\b", "Imię"),
    (r"\bimie\b", "imię"),
    (r"\bODPOWIEDZIALNOŚCIA\b", "ODPOWIEDZIALNOŚCIĄ"),
    (r"\bPołaczenie\b", "Połączenie"),
    (r"\bPolaczenie\b", "Połączenie"),
    (r"\bpolaczenie\b", "połączenie"),
    (r"\bZalegle\b", "Zaległe"),
    (r"\bzalegle\b", "zaległe"),
    (r"\bPolaczony\b", "Połączony"),
    (r"\bPolaczone\b", "Połączone"),
    (r"\bZelazowa\b", "Żelazowa"),
    (r"\bzelazowa\b", "żelazowa"),
]


def is_polish_document_text(text: str) -> bool:
    if has_structural_polish_defect(text):
        return True
    try:
        from app.services.document_language import DocumentLanguage, get_document_language

        hint = get_document_language()
        if hint == DocumentLanguage.PL:
            return True
        if hint == DocumentLanguage.EN:
            return False
    except Exception:  # noqa: BLE001
        pass
    from app.services.ocr_lang_routing import polish_chars_detected

    if polish_chars_detected(text):
        return True
    return bool(_PL_DOC_HINT.search(text))


def has_structural_polish_defect(text: str) -> bool:
    return bool(text and _POLISH_STRUCTURAL_DEFECT.search(text))


def polish_ocr_needs_verification(text: str) -> bool:
    if not text or not text.strip():
        return False
    if has_structural_polish_defect(text):
        return True
    if is_polish_document_text(text) and re.search(r"[šžñň]", text):
        return True
    if _looks_like_ascii_polish_keywords(text):
        return True
    return False


def _looks_like_ascii_polish_keywords(text: str) -> bool:
    return bool(
        re.search(
            r"\b(platnosc|wartosc|uslug|zaplata|sprzedaz|faktura|nabywca|sprzedawca|"
            r"wystawien|oplacona|oplacono)\b",
            text,
            re.IGNORECASE,
        ),
    )


def apply_caron_and_script_fixes(text: str) -> str:
    """Map non-Polish letters that Latin OCR models emit on Polish scans."""
    text = text.replace("ž", "ż").replace("Ž", "Ż")
    text = text.replace("už", "uż").replace("Už", "Uż")
    if not is_polish_document_text(text):
        return text
    text = text.replace("š", "ś").replace("Š", "Ś")
    text = text.replace("ñ", "ń").replace("Ñ", "Ń")
    text = text.replace("ň", "ń").replace("Ň", "Ń")
    return text


def apply_invoice_stem_fixes(text: str) -> str:
    if not is_polish_document_text(text):
        return text
    all_fixes = (
        _INVOICE_STEM_FIXES
        + _TLUMACZ_FIXES
        + _AMOUNT_WORD_FIXES
        + _WLOCHY_FIXES
        + _MISC_INVOICE_FIXES
    )
    for pattern, replacement in all_fixes:
        text = re.sub(pattern, replacement, text)
    return text


def _tesseract_looks_garbled(rapid: str, tess: str) -> bool:
    rapid, tess = rapid.strip(), tess.strip()
    if not tess:
        return True
    if len(rapid) >= 8:
        ratio = len(tess) / len(rapid)
        if ratio < 0.35 or ratio > 2.8:
            return True
    if tess.count("|") >= 3:
        return True
    if sum(1 for c in tess if ord(c) < 32) > 0:
        return True
    return False


def should_prefer_tesseract(rapid: str, tess: str) -> bool:
    if _tesseract_looks_garbled(rapid, tess):
        return False
    try:
        from app.services.document_language import DocumentLanguage, get_document_language

        if get_document_language() == DocumentLanguage.EN:
            return False
    except Exception:  # noqa: BLE001
        pass
    if not is_polish_document_text(rapid) and not is_polish_document_text(tess):
        return False
    da, db = polish_diacritic_score(rapid), polish_diacritic_score(tess)
    if db > da:
        return True
    # tess has PL diacritics where rapid has none on a similar-length line
    if da == 0 and db > 0 and len(tess) >= len(rapid) * 0.6:
        return True
    return False


def merge_ocr_line_polish(rapid: str, tess: str = "") -> str:
    """Pick better engine output, then apply structural fixes."""
    rapid = (rapid or "").strip()
    tess = (tess or "").strip()
    if not rapid:
        return ""
    base = rapid
    if tess and should_prefer_tesseract(rapid, tess):
        base = pick_better_text(rapid, tess)
    elif tess:
        base = cherrypick_tesseract_tokens(rapid, tess)
    return base


# Structural token pairs: if Tess has the canonical form, prefer it over Rapid.
_CHERRY_TOKEN_PAIRS: list[tuple[str, str, str]] = [
    (r"llość", r"Ilość", "Ilość"),
    (r"ustugi", r"usługi", "usługi"),
    (r"[jJfF]okale", r"lokale", "lokale"),
    (r"speinia", r"spełnia", "spełnia"),
    (r"mlejsce", r"miejsce", "miejsce"),
    (r"wystawien[lł][aá]", r"wystawienia", "wystawienia"),
    (r"wystawlenla", r"wystawienia", "wystawienia"),
    (r"pfatno[sš]", r"płatności", "płatności"),
    (r"Platno[sš][cć]", r"Płatność", "Płatność"),
    (r"Wart[oó][sš][cć]", r"Wartość", "Wartość"),
]


def cherrypick_tesseract_tokens(rapid: str, tess: str) -> str:
    """Borrow tokens from Tess when they match known structural corrections."""
    if not rapid.strip() or not tess.strip() or _tesseract_looks_garbled(rapid, tess):
        return rapid
    out = rapid
    for rapid_pat, tess_pat, replacement in _CHERRY_TOKEN_PAIRS:
        if re.search(rapid_pat, out, re.IGNORECASE) and re.search(
            tess_pat, tess, re.IGNORECASE,
        ):
            out = re.sub(rapid_pat, replacement, out, flags=re.IGNORECASE)
    return out
