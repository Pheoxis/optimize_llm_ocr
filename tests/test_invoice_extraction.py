"""Unit tests for propOS invoice extraction helpers."""

from __future__ import annotations

from app.models.propos_invoice import FieldValue, InvoiceResult
from app.services.invoice_extraction import (
    _coerce_field_value,
    _lower_confidence_on_sum_mismatch,
    _normalize_money_field,
    _normalize_nip_field,
    _parse_fields,
    _parse_line_items,
    _parse_vat_summary,
)


def test_coerce_field_value_null() -> None:
    fv = _coerce_field_value(None)
    assert fv.value is None
    assert fv.confidence == 0.0


def test_normalize_nip_digits() -> None:
    raw = FieldValue(value="526-020-74-27", confidence=0.9)
    out = _normalize_nip_field(raw)
    assert out is not None
    assert out.value == "5260207427"
    assert out.confidence == 0.9


def test_normalize_money_to_grosze() -> None:
    fv = FieldValue(value="123.45", confidence=0.8)
    out = _normalize_money_field(fv)
    assert out is not None
    assert out.value == 12345


def test_parse_fields_and_line_items() -> None:
    fields = _parse_fields(
        {
            "invoice_number": {"value": "FV/1/2026", "confidence": 0.95},
            "total_gross": {"value": 10000, "confidence": 0.9},
            "seller": {"name": {"value": "ABC Sp. z o.o.", "confidence": 0.9}},
        },
    )
    assert fields.invoice_number is not None
    assert fields.invoice_number.value == "FV/1/2026"
    items = _parse_line_items(
        [
            {
                "name": {"value": "Usługa IT", "confidence": 0.9},
                "unit_price_net": {"value": 10000, "confidence": 0.85},
                "vat_rate": {"value": "23", "confidence": 0.9},
            },
        ],
    )
    assert len(items) == 1
    assert items[0].name is not None
    assert items[0].name.value == "Usługa IT"


def test_vat_summary_always_list() -> None:
    rows = _parse_vat_summary(
        [{"vat_rate": {"value": "23", "confidence": 0.9}, "net_amount": {"value": 10000, "confidence": 0.9}}],
    )
    assert len(rows) == 1
    assert rows[0].vat_rate is not None
    assert rows[0].vat_rate.value == "23"


def test_sum_mismatch_lowers_confidence() -> None:
    result = InvoiceResult(
        raw_text="x",
        fields=_parse_fields(
            {
                "total_net": {"value": 10000, "confidence": 0.95},
                "total_vat": {"value": 2300, "confidence": 0.95},
                "total_gross": {"value": 15000, "confidence": 0.95},
            },
        ),
        line_items=[],
        vat_summary=[],
    )
    warnings = _lower_confidence_on_sum_mismatch(result)
    assert "header_totals_inconsistent" in warnings
    assert result.fields.total_gross is not None
    assert result.fields.total_gross.confidence <= 0.35
