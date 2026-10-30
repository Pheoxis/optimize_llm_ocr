"""Shared types for semantic OCR field validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationSeverity(str, Enum):
    """How strongly a failed check should influence escalation."""

    ERROR = "error"  # checksum / format failure on a detected candidate
    WARNING = "warning"  # near-miss or weak format match


@dataclass(frozen=True)
class FieldIssue:
    """One detected token and the outcome of semantic validation."""

    kind: str
    value: str
    valid: bool
    reason: str | None = None
    severity: ValidationSeverity = ValidationSeverity.ERROR

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "valid": self.valid,
            "reason": self.reason,
            "severity": self.severity.value,
        }


@dataclass
class SemanticValidationResult:
    """Aggregated semantic validation for a line or text block."""

    issues: list[FieldIssue] = field(default_factory=list)
    passed: bool = True
    should_escalate: bool = False
    primary_reason: str | None = None
    failed_kinds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "should_escalate": self.should_escalate,
            "primary_reason": self.primary_reason,
            "failed_kinds": self.failed_kinds,
            "issues": [i.to_dict() for i in self.issues],
        }
