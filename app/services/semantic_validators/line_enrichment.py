"""Apply semantic validation metadata to ``OcrLine`` objects."""

from __future__ import annotations

from app.core.config import settings
from app.models.schemas import OcrLine
from app.services.semantic_validators.registry import validate_text
from app.services.semantic_validators.types import ValidationSeverity


def enrich_ocr_line_semantic(line: OcrLine) -> OcrLine:
    """Run semantic validators on line text and attach validation fields."""
    if not settings.ocr_semantic_validation_enabled:
        return line

    text = (line.rapid_text or line.text or "").strip()
    if not text:
        return line

    result = validate_text(text)
    invalid = [
        i for i in result.issues
        if not i.valid and i.severity == ValidationSeverity.ERROR
    ]
    warnings = [
        i for i in result.issues
        if not i.valid and i.severity == ValidationSeverity.WARNING
    ]

    updates: dict[str, object] = {
        "validation_passed": result.passed,
        "validation_issues": [i.to_dict() for i in result.issues if not i.valid],
        "validation_failed_kinds": result.failed_kinds,
    }

    if result.should_escalate:
        hint = result.primary_reason or "semantic_validation_failed"
        updates["escalation_hint"] = line.escalation_hint or hint
        if line.suspicious is not True:
            updates["suspicious"] = True

    if invalid or warnings:
        updates["validation_passed"] = len(invalid) == 0 and len(warnings) == 0

    return line.model_copy(update=updates)
