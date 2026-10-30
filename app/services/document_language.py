"""Per-job document language hint (PL / EN / auto) for OCR routing."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from enum import StrEnum
from typing import Iterator

from app.core.config import settings

_document_language: ContextVar[DocumentLanguage | None] = ContextVar(
    "document_language",
    default=None,
)


class DocumentLanguage(StrEnum):
    AUTO = "auto"
    PL = "pl"
    EN = "en"


def parse_document_language(value: str | None) -> DocumentLanguage:
    if not value:
        return DocumentLanguage.AUTO
    normalized = value.strip().lower()
    aliases = {
        "auto": DocumentLanguage.AUTO,
        "pl": DocumentLanguage.PL,
        "pol": DocumentLanguage.PL,
        "polski": DocumentLanguage.PL,
        "polish": DocumentLanguage.PL,
        "en": DocumentLanguage.EN,
        "eng": DocumentLanguage.EN,
        "angielski": DocumentLanguage.EN,
        "english": DocumentLanguage.EN,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported document language {value!r} — use auto, pl, or en",
        ) from exc


def get_document_language() -> DocumentLanguage:
    current = _document_language.get()
    if current is not None:
        return current
    try:
        return parse_document_language(settings.ocr_document_language)
    except ValueError:
        return DocumentLanguage.AUTO


def set_document_language(lang: DocumentLanguage | str | None) -> Token:
    if lang is None:
        return _document_language.set(None)
    if isinstance(lang, DocumentLanguage):
        return _document_language.set(lang)
    return _document_language.set(parse_document_language(lang))


def reset_document_language(token: Token) -> None:
    _document_language.reset(token)


@contextmanager
def document_language_context(lang: DocumentLanguage | str | None) -> Iterator[None]:
    token = set_document_language(lang)
    try:
        yield
    finally:
        reset_document_language(token)


def effective_primary_ocr_lang() -> str:
    """Tesseract / OCR lang string for jobs without a text sample yet."""
    hint = get_document_language()
    if hint == DocumentLanguage.PL:
        return settings.ocr_lang_tesseract_polish
    if hint == DocumentLanguage.EN:
        return settings.ocr_lang_tesseract_english
    if not settings.ocr_lang_routing_enabled:
        return settings.ocr_lang
    return settings.ocr_lang_tesseract_polish


def apply_polish_postprocess(text: str) -> str:
    from app.services.ocr_postprocess import correct_polish_ocr

    if get_document_language() == DocumentLanguage.EN:
        return text
    return correct_polish_ocr(text)


def languages_for_result(text: str | None = None) -> list[str]:
    from app.services.ocr_lang_routing import languages_detected_from_text

    hint = get_document_language()
    if hint == DocumentLanguage.PL:
        return ["pl", "en"]
    if hint == DocumentLanguage.EN:
        return ["en"]
    if text and text.strip():
        return languages_detected_from_text(text)
    if not settings.ocr_lang_routing_enabled:
        return [part.strip() for part in settings.ocr_lang.split("+") if part.strip()]
    return ["en"]
