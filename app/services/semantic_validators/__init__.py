"""Modular semantic validators for OCR output (checksums, formats)."""

from app.services.semantic_validators.base import SemanticValidator
from app.services.semantic_validators.line_enrichment import enrich_ocr_line_semantic
from app.services.semantic_validators.registry import (
    all_validators,
    log_validation_metrics,
    register_validator,
    reset_custom_validators,
    semantic_validation_failed,
    validate_text,
)
from app.services.semantic_validators.types import (
    FieldIssue,
    SemanticValidationResult,
    ValidationSeverity,
)

__all__ = [
    "FieldIssue",
    "SemanticValidationResult",
    "SemanticValidator",
    "ValidationSeverity",
    "all_validators",
    "log_validation_metrics",
    "enrich_ocr_line_semantic",
    "register_validator",
    "reset_custom_validators",
    "semantic_validation_failed",
    "validate_text",
]
