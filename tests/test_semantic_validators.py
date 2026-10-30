"""Semantic OCR field validators (checksums, formats, registry)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.config import settings
from app.services.ocr_quality import classify_line_escalation
from app.services.ocr_validators import find_sensitive_fields, regex_validation_failed
from app.services.semantic_validators import (
    FieldIssue,
    SemanticValidator,
    register_validator,
    reset_custom_validators,
    validate_text,
)
from app.services.semantic_validators.checksums import (
    iban_checksum,
    nip_checksum,
    pesel_checksum,
    regon9_checksum,
    vin_check_digit_ok,
)
from app.services.semantic_validators.types import ValidationSeverity


# Known-valid test vectors (checksums verified against helpers).
VALID_PESEL = "44051401458"
VALID_NIP = "5261040828"
VALID_REGON9 = "610188201"
VALID_IBAN_PL = "PL61109010140000071219812874"
VALID_VIN = "1HGCM82633A004352"
VALID_DATE = "2024-03-15"
VALID_INVOICE = "FV/12/03/2024"


@pytest.fixture(autouse=True)
def _enable_semantic_validation() -> None:
    with (
        patch.object(settings, "ocr_semantic_validation_enabled", True),
        patch.object(settings, "ocr_semantic_validation_escalate", True),
        patch.object(settings, "ocr_semantic_validation_escalate_warnings", False),
    ):
        reset_custom_validators()
        yield
        reset_custom_validators()


def test_checksum_helpers_accept_valid_samples() -> None:
    assert pesel_checksum(VALID_PESEL)
    assert nip_checksum(VALID_NIP)
    assert regon9_checksum(VALID_REGON9)
    assert iban_checksum(VALID_IBAN_PL)
    assert vin_check_digit_ok(VALID_VIN)


def test_pesel_invalid_checksum_escalates() -> None:
    result = validate_text(f"PESEL: {VALID_PESEL[:-1]}0")
    invalid = [i for i in result.issues if i.kind == "pesel" and not i.valid]
    assert invalid
    assert invalid[0].reason == "checksum_fail"
    assert result.should_escalate


def test_nip_valid_and_invalid() -> None:
    ok = validate_text(f"NIP {VALID_NIP}")
    assert any(i.kind == "nip" and i.valid for i in ok.issues)
    bad = validate_text("NIP 1234567890")
    nip_issues = [i for i in bad.issues if i.kind == "nip"]
    assert nip_issues
    assert not nip_issues[0].valid


def test_date_invalid() -> None:
    result = validate_text("Data: 2024-02-30")
    dates = [i for i in result.issues if i.kind == "date"]
    assert dates and not dates[0].valid
    assert dates[0].reason == "invalid_date"


def test_date_valid() -> None:
    result = validate_text(f"Wystawiono: {VALID_DATE}")
    dates = [i for i in result.issues if i.kind == "date"]
    assert dates and dates[0].valid


def test_invoice_valid() -> None:
    result = validate_text(f"Faktura {VALID_INVOICE}")
    inv = [i for i in result.issues if i.kind == "invoice_number"]
    assert inv and inv[0].valid


def test_custom_validator_registration() -> None:
    class DummyValidator(SemanticValidator):
        kind = "dummy"

        def scan(self, text: str) -> list[FieldIssue]:
            if "MAGIC_TOKEN" in text:
                return [self._issue("MAGIC_TOKEN", valid=False, reason="blocked")]
            return []

    register_validator(DummyValidator())
    result = validate_text("line with MAGIC_TOKEN here")
    assert result.should_escalate
    assert "dummy" in result.failed_kinds


def test_regex_validation_failed_delegates() -> None:
    failed, reason = regex_validation_failed(f"id {VALID_PESEL[:-1]}0")
    assert failed is True
    assert reason and reason.startswith("semantic_pesel_")


def test_classify_line_escalation_hard_on_checksum() -> None:
    tier, reason = classify_line_escalation(
        mean_conf=0.95,
        min_conf=0.9,
        rapid_text=f"PESEL {VALID_PESEL[:-1]}0",
        tess_text=f"PESEL {VALID_PESEL[:-1]}0",
        quality_score=0.9,
    )
    assert tier == "hard"
    assert reason and "semantic_pesel" in reason


def test_find_sensitive_fields_includes_semantic_kinds() -> None:
    text = f"{VALID_PESEL} {VALID_NIP}"
    fields = find_sensitive_fields(text)
    kinds = {f.kind for f in fields}
    assert "pesel" in kinds
    assert "nip" in kinds


def test_validation_disabled_returns_empty() -> None:
    with patch.object(settings, "ocr_semantic_validation_enabled", False):
        result = validate_text(f"PESEL {VALID_PESEL[:-1]}0")
    assert result.passed is True
    assert result.issues == []
