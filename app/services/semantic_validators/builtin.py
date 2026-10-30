"""Built-in semantic validators (PESEL, NIP, REGON, VIN, IBAN, dates, invoices)."""

from __future__ import annotations

import re
from datetime import datetime

from app.services.semantic_validators.base import SemanticValidator
from app.services.semantic_validators.checksums import (
    iban_checksum,
    nip_checksum,
    pesel_checksum,
    regon14_checksum,
    regon9_checksum,
    vin_check_digit_ok,
    vin_charset_ok,
)
from app.services.semantic_validators.types import FieldIssue, ValidationSeverity

PESEL_RE = re.compile(r"\b\d{11}\b")
NIP_RE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b|\b\d{10}\b")
REGON_RE = re.compile(r"\b\d{9}(?:\d{5})?\b")
VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
IBAN_RE = re.compile(
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(?:\d{4}[-/.]\d{2}[-/.]\d{2}|\d{2}[-/.]\d{2}[-/.]\d{4})\b",
)
INVOICE_RE = re.compile(
    r"\b(?:"
    r"(?:FV|FA|FP|PRO|KOR)\s*[/\-\s]?\d{1,6}(?:[/\-\s]\d{1,6}){1,3}"
    r"|(?:nr|numer)\s*(?:faktury|fv)?\s*[:.]?\s*\d{1,6}[/\-\s]\d{1,6}[/\-\s]\d{2,6}"
    r"|\d{1,4}/\d{1,4}/(?:19|20)\d{2}"
    r")\b",
    re.IGNORECASE,
)
NEAR_PESEL_RE = re.compile(r"\b[\d\s\-]{10,13}\b")


class PeselValidator(SemanticValidator):
    kind = "pesel"

    def scan(self, text: str) -> list[FieldIssue]:
        issues: list[FieldIssue] = []
        for m in PESEL_RE.finditer(text):
            value = m.group(0)
            ok = pesel_checksum(value)
            issues.append(self._issue(
                value,
                valid=ok,
                reason=None if ok else "checksum_fail",
            ))
        for m in NEAR_PESEL_RE.finditer(text):
            raw = m.group(0)
            digits = self._digits_only(raw)
            if len(digits) == 11 and digits not in {i.value for i in issues}:
                if not pesel_checksum(digits):
                    issues.append(self._issue(
                        digits,
                        valid=False,
                        reason="malformed_near_miss",
                        severity=ValidationSeverity.WARNING,
                    ))
        return issues


class NipValidator(SemanticValidator):
    kind = "nip"

    def scan(self, text: str) -> list[FieldIssue]:
        issues: list[FieldIssue] = []
        seen: set[str] = set()
        for m in NIP_RE.finditer(text):
            raw = m.group(0)
            digits = self._digits_only(raw)
            if len(digits) != 10 or digits in seen:
                if len(digits) != 10 and len(digits) >= 9:
                    issues.append(self._issue(
                        raw.strip(),
                        valid=False,
                        reason="malformed_length",
                    ))
                continue
            seen.add(digits)
            ok = nip_checksum(digits)
            issues.append(self._issue(
                digits,
                valid=ok,
                reason=None if ok else "checksum_fail",
            ))
        return issues


class RegonValidator(SemanticValidator):
    kind = "regon"

    def scan(self, text: str) -> list[FieldIssue]:
        issues: list[FieldIssue] = []
        for m in REGON_RE.finditer(text):
            digits = m.group(0)
            if len(digits) not in (9, 14):
                issues.append(self._issue(
                    digits,
                    valid=False,
                    reason="malformed_length",
                ))
                continue
            if len(digits) == 9:
                ok = regon9_checksum(digits)
            else:
                ok = regon14_checksum(digits)
            issues.append(self._issue(
                digits,
                valid=ok,
                reason=None if ok else "checksum_fail",
            ))
        return issues


class VinValidator(SemanticValidator):
    kind = "vin"

    def scan(self, text: str) -> list[FieldIssue]:
        issues: list[FieldIssue] = []
        for m in VIN_RE.finditer(text):
            value = m.group(0).upper()
            charset_ok = vin_charset_ok(value)
            if not charset_ok:
                issues.append(self._issue(
                    value,
                    valid=False,
                    reason="malformed_charset",
                ))
                continue
            check_ok = vin_check_digit_ok(value)
            issues.append(self._issue(
                value,
                valid=check_ok,
                reason=None if check_ok else "check_digit_fail",
            ))
        return issues


class IbanValidator(SemanticValidator):
    kind = "iban"

    def scan(self, text: str) -> list[FieldIssue]:
        issues: list[FieldIssue] = []
        for m in IBAN_RE.finditer(text):
            value = m.group(0).replace(" ", "").upper()
            ok = iban_checksum(value)
            issues.append(self._issue(
                value,
                valid=ok,
                reason=None if ok else "checksum_fail",
            ))
        return issues


class DateValidator(SemanticValidator):
    kind = "date"

    _FMTS = (
        "%Y-%m-%d",
        "%Y.%m.%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%d/%m/%Y",
    )

    def _parse_date(self, token: str) -> bool:
        normalized = token.replace("/", "-").replace(".", "-")
        for fmt in self._FMTS:
            try:
                datetime.strptime(normalized, fmt.replace(".", "-").replace("/", "-"))
                return True
            except ValueError:
                continue
        return False

    def scan(self, text: str) -> list[FieldIssue]:
        issues: list[FieldIssue] = []
        for m in DATE_RE.finditer(text):
            value = m.group(0)
            ok = self._parse_date(value)
            issues.append(self._issue(
                value,
                valid=ok,
                reason=None if ok else "invalid_date",
            ))
        return issues


class InvoiceNumberValidator(SemanticValidator):
    kind = "invoice_number"

    def scan(self, text: str) -> list[FieldIssue]:
        issues: list[FieldIssue] = []
        for m in INVOICE_RE.finditer(text):
            value = m.group(0).strip()
            normalized = re.sub(r"\s+", " ", value)
            parts = re.split(r"[/\-\s]+", normalized)
            digit_parts = [p for p in parts if p.isdigit()]
            if len(digit_parts) < 2:
                issues.append(self._issue(
                    value,
                    valid=False,
                    reason="malformed_structure",
                ))
                continue
            if any(len(p) > 6 for p in digit_parts):
                issues.append(self._issue(
                    value,
                    valid=False,
                    reason="segment_too_long",
                ))
                continue
            issues.append(self._issue(value, valid=True))
        return issues
