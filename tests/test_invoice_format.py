"""Tests for propOS wire-format mapping."""

from __future__ import annotations

from uuid import uuid4

from app.core.constants import JobStatus
from app.models.propos_invoice import (
    FieldValue,
    InvoiceExtractResponse,
    InvoiceHeaderFields,
    InvoiceLineItem,
    InvoiceProcessingInfo,
    InvoiceResult,
    PartyFields,
    VatSummaryRow,
)
from app.services.invoice_format import to_invoice_job_response


def test_done_payload_shape() -> None:
    internal = InvoiceExtractResponse(
        status="done",
        processing=InvoiceProcessingInfo(
            ocr_engine="fast_ocr+tesseract",
            ocr_confidence=0.93,
            elapsed_ms=14200,
            page_count=2,
        ),
        result=InvoiceResult(
            raw_text="TAURON\nFV/2026/05/001",
            fields=InvoiceHeaderFields(
                invoice_number=FieldValue(value="FV/2026/05/001", confidence=0.98),
                total_net=FieldValue(value=10000, confidence=0.96),
                total_vat=FieldValue(value=2300, confidence=0.96),
                total_gross=FieldValue(value=12300, confidence=0.98),
                seller=PartyFields(
                    name=FieldValue(value="Tauron", confidence=0.95),
                    nip=FieldValue(value="9542583988", confidence=0.99),
                ),
            ),
            line_items=[
                InvoiceLineItem(
                    name=FieldValue(value="Energia", confidence=0.91),
                    quantity=FieldValue(value=1450.5, confidence=0.9),
                    unit=FieldValue(value="kWh", confidence=0.9),
                    unit_price_net=FieldValue(value=68, confidence=0.9),
                    net_amount=FieldValue(value=9863, confidence=0.9),
                    vat_rate=FieldValue(value="23", confidence=0.9),
                    gross_amount=FieldValue(value=12131, confidence=0.9),
                ),
            ],
            vat_summary=[
                VatSummaryRow(
                    vat_rate=FieldValue(value="23", confidence=0.9),
                    net_amount=FieldValue(value=10000, confidence=0.9),
                    vat_amount=FieldValue(value=2300, confidence=0.9),
                    gross_amount=FieldValue(value=12300, confidence=0.9),
                ),
            ],
        ),
    )
    job_id = uuid4()
    out = to_invoice_job_response(
        job_id,
        "pi_12345",
        job_status=JobStatus.DONE,
        internal=internal,
    )
    assert out.status == "done"
    assert out.external_id == "pi_12345"
    assert out.processing.engine == "rapidocr+tesseract"
    assert out.processing.duration_ms == 14200
    assert out.processing.pages == 2
    assert out.result is not None
    assert out.result.confidence == 0.93
    assert out.result.fields.invoice_number is not None
    assert out.result.fields.invoice_number.value == "FV/2026/05/001"
    item = out.result.line_items[0]
    assert item.name == "Energia"
    assert item.total_net == 9863
    assert item.confidence > 0
    assert out.result.vat_summary[0].rate == "23"
    assert out.result.vat_summary[0].gross == 12300


def test_failed_payload_shape() -> None:
    internal = InvoiceExtractResponse(
        status="failed",
        processing=InvoiceProcessingInfo(
            ocr_engine="fast_ocr",
            elapsed_ms=3100,
            page_count=1,
        ),
        error={
            "code": "UNREADABLE",
            "message": "Dokument nieczytelny — zbyt niska rozdzielczość skanu.",
        },
    )
    out = to_invoice_job_response(
        uuid4(),
        "pi_12345",
        job_status=JobStatus.FAILED,
        internal=internal,
    )
    assert out.status == "failed"
    assert out.result is None
    assert out.error is not None
    assert out.error.code == "UNREADABLE"
    assert out.processing.pages == 1
