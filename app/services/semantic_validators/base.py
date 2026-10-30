"""Base class for pluggable semantic validators."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.services.semantic_validators.types import FieldIssue, ValidationSeverity


class SemanticValidator(ABC):
    """
    Detect candidate tokens in OCR text and validate checksum / format.

    Subclass and register via ``register_validator`` for custom fields.
    """

    kind: str = "base"

    @abstractmethod
    def scan(self, text: str) -> list[FieldIssue]:
        """Return all field issues found in *text* (valid and invalid)."""

    def _issue(
        self,
        value: str,
        *,
        valid: bool,
        reason: str | None = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> FieldIssue:
        return FieldIssue(
            kind=self.kind,
            value=value,
            valid=valid,
            reason=reason,
            severity=severity,
        )

    @staticmethod
    def _digits_only(value: str) -> str:
        return "".join(c for c in value if c.isdigit())

    @staticmethod
    def _find_pattern(text: str, pattern: re.Pattern[str]) -> list[str]:
        return [m.group(0) for m in pattern.finditer(text)]
