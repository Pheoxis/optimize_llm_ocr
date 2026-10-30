"""Regex + checksum validators for sensitive fields (delegates to semantic_validators)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.semantic_validators.registry import (
    semantic_validation_failed,
    validate_text,
)
# Legacy plate pattern (not part of built-in semantic validators).
PLATE_RE = re.compile(r"\b[A-Z]{2,3}\s?[A-Z0-9]{4,5}\b")


@dataclass(frozen=True)
class FieldMatch:
    kind: str
    value: str
    valid: bool
    reason: str | None = None


def find_sensitive_fields(text: str) -> list[FieldMatch]:
    """Detect validated semantic fields (+ legacy plates) in a line."""
    out: list[FieldMatch] = []
    result = validate_text(text)
    for issue in result.issues:
        out.append(FieldMatch(
            kind=issue.kind,
            value=issue.value,
            valid=issue.valid,
            reason=issue.reason,
        ))
    for m in PLATE_RE.finditer(text):
        out.append(FieldMatch("plate", m.group(0).replace(" ", ""), True))
    return out


def regex_validation_failed(text: str) -> tuple[bool, str | None]:
    """
    True when a sensitive token looks almost valid but fails checksum/charset.
    """
    failed, reason = semantic_validation_failed(text)
    if failed:
        return True, reason
    return False, None
