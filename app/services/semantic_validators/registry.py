"""Validator registry and orchestration."""

from __future__ import annotations

import logging

from app.core.config import settings
from app.services.semantic_validators.base import SemanticValidator
from app.services.semantic_validators.builtin import (
    DateValidator,
    IbanValidator,
    InvoiceNumberValidator,
    NipValidator,
    PeselValidator,
    RegonValidator,
    VinValidator,
)
from app.services.semantic_validators.types import (
    FieldIssue,
    SemanticValidationResult,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)

_BUILTIN: list[SemanticValidator] = [
    PeselValidator(),
    NipValidator(),
    RegonValidator(),
    VinValidator(),
    IbanValidator(),
    DateValidator(),
    InvoiceNumberValidator(),
]

_custom_validators: list[SemanticValidator] = []


def register_validator(validator: SemanticValidator) -> None:
    """Register a custom validator (runs after built-ins)."""
    _custom_validators.append(validator)
    logger.info("Registered semantic validator: %s", validator.kind)


def reset_custom_validators() -> None:
    """Clear custom validators (tests only)."""
    _custom_validators.clear()


def all_validators() -> list[SemanticValidator]:
    return [* _BUILTIN, *_custom_validators]


def semantic_validation_failed(text: str) -> tuple[bool, str | None]:
    """True when ERROR-severity semantic checks fail (for escalation scoring)."""
    if not settings.ocr_semantic_validation_enabled:
        return False, None
    result = validate_text(text)
    invalid = [
        i for i in result.issues
        if not i.valid and i.severity == ValidationSeverity.ERROR
    ]
    if not invalid:
        return False, None
    first = invalid[0]
    reason = first.reason or "invalid"
    return True, f"semantic_{first.kind}_{reason}:{first.value}"


def validate_text(text: str) -> SemanticValidationResult:
    """
    Run all registered semantic validators on OCR line text.

    ``should_escalate`` is true when any ERROR-severity invalid field is found.
    """
    if not settings.ocr_semantic_validation_enabled:
        return SemanticValidationResult()

    if not (text or "").strip():
        return SemanticValidationResult()

    issues: list[FieldIssue] = []
    for validator in all_validators():
        try:
            issues.extend(validator.scan(text))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Semantic validator %s failed: %s",
                validator.kind,
                exc,
            )

    invalid = [
        i for i in issues
        if not i.valid and i.severity == ValidationSeverity.ERROR
    ]
    warnings = [
        i for i in issues
        if not i.valid and i.severity == ValidationSeverity.WARNING
    ]

    should_escalate = False
    primary_reason: str | None = None
    failed_kinds: list[str] = []

    if invalid and settings.ocr_semantic_validation_escalate:
        should_escalate = True
        first = invalid[0]
        primary_reason = f"semantic_{first.kind}_{first.reason}:{first.value}"
        failed_kinds = sorted({i.kind for i in invalid})
    elif warnings and settings.ocr_semantic_validation_escalate_warnings:
        should_escalate = True
        first = warnings[0]
        primary_reason = f"semantic_{first.kind}_{first.reason}:{first.value}"
        failed_kinds = sorted({i.kind for i in warnings})

    passed = len(invalid) == 0 and len(warnings) == 0

    if should_escalate:
        logger.info(
            "Semantic validation failed: %s (%d issues)",
            primary_reason,
            len(invalid) + len(warnings),
        )

    return SemanticValidationResult(
        issues=issues,
        passed=passed,
        should_escalate=should_escalate,
        primary_reason=primary_reason,
        failed_kinds=failed_kinds,
    )


def log_validation_metrics(
    results: list[SemanticValidationResult],
    *,
    label: str = "",
) -> None:
    """Log aggregate semantic validation stats for a page or job."""
    if not results:
        return
    failed = sum(1 for r in results if r.should_escalate)
    total_issues = sum(
        len([i for i in r.issues if not i.valid]) for r in results
    )
    prefix = f"{label}: " if label else ""
    logger.info(
        "%ssemantic validation — lines=%d escalate=%d invalid_fields=%d",
        prefix,
        len(results),
        failed,
        total_issues,
    )
