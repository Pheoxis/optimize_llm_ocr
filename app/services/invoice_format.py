"""Map internal invoice extraction to propOS wire JSON (§4.1 / §4.2)."""

from __future__ import annotations

from uuid import UUID

from app.core.config import settings
from app.core.constants import JobStatus
from app.models.propos_invoice import (
    FieldValue,
    InvoiceExtractResponse,
    InvoiceHeaderFields,
    InvoiceJobResponse,
    InvoiceLineItem,
    InvoiceResult,
    PartyFields,
    ProposFieldValue,
    ProposHeaderFields,
    ProposInvoiceError,
    ProposInvoiceResult,
    ProposLineItem,
    ProposPartyFields,
    ProposProcessingInfo,
    ProposVatSummaryRow,
    VatSummaryRow,
)


def _engine_label(raw: str | None) -> str | None:
    if not raw:
        return None
    label = raw.replace("fast_ocr", "rapidocr")
    if "vlm" in label or "vision" in label or "minicpm" in label:
        if "rapidocr" in label:
            return "rapidocr+vlm"
        return label
    return label


def _round_conf(value: float) -> float:
    return round(value, 2)


def _field_or_empty(fv: FieldValue | None) -> ProposFieldValue:
    if fv is None:
        return ProposFieldValue(value=None, confidence=0.0)
    return ProposFieldValue(value=fv.value, confidence=_round_conf(fv.confidence))


def _party_or_empty(party: PartyFields | None) -> ProposPartyFields:
    if party is None:
        return ProposPartyFields(
            name=ProposFieldValue(value=None, confidence=0.0),
            nip=ProposFieldValue(value=None, confidence=0.0),
            address=ProposFieldValue(value=None, confidence=0.0),
        )
    return ProposPartyFields(
        name=_field_or_empty(party.name),
        nip=_field_or_empty(party.nip),
        address=_field_or_empty(party.address),
    )


def _to_propos_header(fields: InvoiceHeaderFields) -> ProposHeaderFields:
    """§4.1 — fixed field set (no sale_date); always emit all keys."""
    return ProposHeaderFields(
        invoice_number=_field_or_empty(fields.invoice_number),
        issue_date=_field_or_empty(fields.issue_date),
        due_date=_field_or_empty(fields.due_date),
        currency=_field_or_empty(fields.currency),
        total_net=_field_or_empty(fields.total_net),
        total_vat=_field_or_empty(fields.total_vat),
        total_gross=_field_or_empty(fields.total_gross),
        seller=_party_or_empty(fields.seller),
        buyer=_party_or_empty(fields.buyer),
        payment_method=_field_or_empty(fields.payment_method),
        bank_account=_field_or_empty(fields.bank_account),
        description=_field_or_empty(fields.description),
    )


def _scalar(fv: FieldValue | None) -> str | int | float | None:
    return None if fv is None else fv.value


def _scalar_int(fv: FieldValue | None) -> int | None:
    if fv is None or fv.value is None:
        return None
    try:
        return int(fv.value)
    except (TypeError, ValueError):
        return None


def _row_confidence(row: InvoiceLineItem) -> float:
    confs = [
        f.confidence
        for f in (
            row.name,
            row.quantity,
            row.unit,
            row.unit_price_net,
            row.vat_rate,
            row.net_amount,
            row.gross_amount,
        )
        if f is not None and f.value is not None
    ]
    if not confs:
        return 0.0
    return _round_conf(sum(confs) / len(confs))


def _to_propos_line_item(row: InvoiceLineItem) -> ProposLineItem:
    return ProposLineItem(
        name=_scalar(row.name),
        quantity=_scalar(row.quantity),
        unit=_scalar(row.unit) if _scalar(row.unit) is not None else None,
        unit_price_net=_scalar_int(row.unit_price_net),
        total_net=_scalar_int(row.net_amount),
        vat_rate=_scalar(row.vat_rate) if _scalar(row.vat_rate) is not None else None,
        total_gross=_scalar_int(row.gross_amount),
        confidence=_row_confidence(row),
    )


def _to_propos_vat_row(row: VatSummaryRow) -> ProposVatSummaryRow:
    rate = _scalar(row.vat_rate)
    return ProposVatSummaryRow(
        rate=str(rate) if rate is not None else None,
        net=_scalar_int(row.net_amount),
        vat=_scalar_int(row.vat_amount),
        gross=_scalar_int(row.gross_amount),
    )


def _result_confidence(result: InvoiceResult, ocr_confidence: float | None) -> float | None:
    if ocr_confidence is not None:
        return _round_conf(ocr_confidence)
    confs: list[float] = []
    for attr in (
        "invoice_number",
        "issue_date",
        "total_gross",
        "total_net",
        "total_vat",
    ):
        fv = getattr(result.fields, attr, None)
        if isinstance(fv, FieldValue) and fv.value is not None:
            confs.append(fv.confidence)
    if not confs:
        return None
    return _round_conf(sum(confs) / len(confs))


def _wire_raw_text(text: str) -> str:
    limit = settings.invoice_raw_text_max_chars
    if len(text) <= limit:
        return text
    return text[:limit] + "\n..."


def _to_propos_result(
    result: InvoiceResult,
    ocr_confidence: float | None,
) -> ProposInvoiceResult:
    return ProposInvoiceResult(
        confidence=_result_confidence(result, ocr_confidence),
        fields=_to_propos_header(result.fields),
        line_items=[_to_propos_line_item(row) for row in result.line_items],
        vat_summary=[_to_propos_vat_row(row) for row in result.vat_summary],
        raw_text=_wire_raw_text(result.raw_text),
    )


def to_invoice_job_response(
    job_id: UUID,
    external_id: str | None,
    *,
    job_status: JobStatus,
    internal: InvoiceExtractResponse | None = None,
) -> InvoiceJobResponse:
    """Build §4.1 / §4.2 payload from internal extraction output."""
    if job_status in (JobStatus.QUEUED, JobStatus.PROCESSING):
        return InvoiceJobResponse(
            job_id=job_id,
            external_id=external_id,
            status=job_status.value,  # type: ignore[arg-type]
            processing=ProposProcessingInfo(),
            result=None,
            error=None,
        )

    if internal is None:
        return InvoiceJobResponse(
            job_id=job_id,
            external_id=external_id,
            status="failed",
            processing=ProposProcessingInfo(),
            result=None,
            error=ProposInvoiceError(code="INTERNAL", message="Missing extraction result"),
        )

    proc = ProposProcessingInfo(
        engine=_engine_label(internal.processing.ocr_engine),
        duration_ms=internal.processing.elapsed_ms,
        pages=internal.processing.page_count or 0,
    )

    if internal.status == "failed":
        err_raw = internal.error or {}
        return InvoiceJobResponse(
            job_id=job_id,
            external_id=external_id,
            status="failed",
            error=ProposInvoiceError(
                code=str(err_raw.get("code", "FAILED")),
                message=str(err_raw.get("message", "Processing failed")),
            ),
            processing=proc,
            result=None,
        )

    assert internal.result is not None
    return InvoiceJobResponse(
        job_id=job_id,
        external_id=external_id,
        status="done",
        error=None,
        processing=proc,
        result=_to_propos_result(internal.result, internal.processing.ocr_confidence),
    )
