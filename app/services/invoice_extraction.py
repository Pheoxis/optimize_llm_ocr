"""OCR text → propOS invoice JSON (LLM + normalization)."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.core.config import settings
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
from app.services.document_sync import ocr_document_sync
from app.services.text_llm import TextLLMError, chat_json

logger = logging.getLogger(__name__)

_NIP_RE = re.compile(r"\D")
# Next document after first invoice in multi-page scans (receipt / invoice 2+).
_INVOICE_BOUNDARY_RE = re.compile(
    r"\n\s*(?:Paragon\s+\d|PARAGON\b|Faktura\s+[2-9]\d*/|FAKTURA\s*\n\s*Nu?Mer\s*:)",
    re.IGNORECASE,
)


def _system_prompt() -> str:
    return """You are an invoice data extractor for Polish accounting (propOS).
Extract structured data ONLY from the OCR text provided.
Return a single JSON object with keys: fields, line_items, vat_summary.

Schema:
- fields: object with invoice_number, issue_date, sale_date, due_date, currency,
  total_net, total_vat, total_gross, payment_method, bank_account, description,
  seller {name,nip,address}, buyer {name,nip,address}
- Each scalar field: {"value": <string|number|null>, "confidence": 0.0-1.0}
- Party subfields use the same {value, confidence} shape
- line_items: array of rows with name, quantity, unit, unit_price_net, vat_rate,
  net_amount, vat_amount, gross_amount (each as {value, confidence})
- vat_summary: array of rows with vat_rate, net_amount, vat_amount, gross_amount

Rules:
- Use ONLY information present in the OCR text. Do NOT invent values.
- If unknown or unreadable: value=null, confidence=0
- description: short invoice subject/title (e.g. service period), NOT a line-item product name
- Do NOT swap seller and buyer
- Dates: ISO YYYY-MM-DD strings
- Currency: 3-letter code (PLN)
- Money amounts in GROSZE (integer): 1 PLN = 100, 123.45 PLN = 12345
- NIP: digits only (10 digits), no spaces or dashes in value
- vat_rate: string e.g. "23", "8", "5", "0", "zw"
- line_items and vat_summary MUST always be arrays (use [] when absent)
- Lower confidence when OCR text is ambiguous
"""


def _isolate_first_invoice_text(text: str) -> str:
    """
    Multi-page uploads often contain several invoices/receipts.
    propOS expects one invoice — keep the first FAKTURA block only.
    """
    t = text.strip()
    if not t:
        return t
    # Start at standalone FAKTURA header — skip page index lines like "Faktura 1/4".
    m = re.search(r"(?im)^FAKTURA\s*$", t)
    if not m:
        m = re.search(r"(?i)\bFAKTURA\b", t)
    if m:
        t = t[m.start() :].lstrip()
    for boundary in _INVOICE_BOUNDARY_RE.finditer(t):
        if boundary.start() >= 50:
            t = t[: boundary.start()].rstrip()
            break
    return t


def _truncate_text(text: str) -> str:
    limit = settings.invoice_max_ocr_chars
    if len(text) <= limit:
        return text
    head = int(limit * 0.75)
    tail = limit - head
    return (
        text[:head]
        + "\n\n[... OCR text truncated for model context ...]\n\n"
        + text[-tail:]
    )


def _coerce_field_value(raw: Any) -> FieldValue:
    if raw is None:
        return FieldValue(value=None, confidence=0.0)
    if isinstance(raw, FieldValue):
        return raw
    if isinstance(raw, dict):
        conf = raw.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            conf_f = 0.0
        return FieldValue(value=raw.get("value"), confidence=max(0.0, min(1.0, conf_f)))
    return FieldValue(value=raw, confidence=0.5)


def _coerce_party(raw: Any) -> PartyFields | None:
    if not isinstance(raw, dict):
        return None
    return PartyFields(
        name=_coerce_field_value(raw.get("name")),
        nip=_coerce_field_value(raw.get("nip")),
        address=_coerce_field_value(raw.get("address")),
    )


def _normalize_nip_field(fv: FieldValue | None) -> FieldValue | None:
    if fv is None or fv.value is None:
        return fv
    digits = _NIP_RE.sub("", str(fv.value))
    if len(digits) != 10:
        return FieldValue(value=fv.value, confidence=min(fv.confidence, 0.4))
    return FieldValue(value=digits, confidence=fv.confidence)


def _normalize_money_field(fv: FieldValue | None) -> FieldValue | None:
    if fv is None or fv.value is None:
        return fv
    try:
        if isinstance(fv.value, bool):
            raise ValueError("bool")
        if isinstance(fv.value, int):
            return fv
        if isinstance(fv.value, float):
            return FieldValue(value=int(round(fv.value)), confidence=fv.confidence)
        s = str(fv.value).strip().replace(",", ".")
        if "." in s:
            return FieldValue(value=int(round(float(s) * 100)), confidence=fv.confidence)
        return FieldValue(value=int(s), confidence=fv.confidence)
    except (TypeError, ValueError):
        return FieldValue(value=None, confidence=0.0)


def _parse_fields(raw: dict[str, Any]) -> InvoiceHeaderFields:
    seller = _coerce_party(raw.get("seller"))
    buyer = _coerce_party(raw.get("buyer"))
    if seller and seller.nip:
        seller.nip = _normalize_nip_field(seller.nip)
    if buyer and buyer.nip:
        buyer.nip = _normalize_nip_field(buyer.nip)

    fields = InvoiceHeaderFields(
        invoice_number=_coerce_field_value(raw.get("invoice_number")),
        issue_date=_coerce_field_value(raw.get("issue_date")),
        sale_date=_coerce_field_value(raw.get("sale_date")),
        due_date=_coerce_field_value(raw.get("due_date")),
        currency=_coerce_field_value(raw.get("currency")),
        total_net=_normalize_money_field(_coerce_field_value(raw.get("total_net"))),
        total_vat=_normalize_money_field(_coerce_field_value(raw.get("total_vat"))),
        total_gross=_normalize_money_field(_coerce_field_value(raw.get("total_gross"))),
        seller=seller,
        buyer=buyer,
        payment_method=_coerce_field_value(raw.get("payment_method")),
        bank_account=_coerce_field_value(raw.get("bank_account")),
        description=_coerce_field_value(raw.get("description")),
    )
    return fields


def _parse_line_items(raw: list[Any]) -> list[InvoiceLineItem]:
    items: list[InvoiceLineItem] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        items.append(
            InvoiceLineItem(
                name=_coerce_field_value(row.get("name")),
                quantity=_coerce_field_value(row.get("quantity")),
                unit=_coerce_field_value(row.get("unit")),
                unit_price_net=_normalize_money_field(
                    _coerce_field_value(row.get("unit_price_net")),
                ),
                vat_rate=_coerce_field_value(row.get("vat_rate")),
                net_amount=_normalize_money_field(_coerce_field_value(row.get("net_amount"))),
                vat_amount=_normalize_money_field(_coerce_field_value(row.get("vat_amount"))),
                gross_amount=_normalize_money_field(
                    _coerce_field_value(row.get("gross_amount")),
                ),
            ),
        )
    return items


def _parse_vat_summary(raw: list[Any]) -> list[VatSummaryRow]:
    rows: list[VatSummaryRow] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        rows.append(
            VatSummaryRow(
                vat_rate=_coerce_field_value(row.get("vat_rate")),
                net_amount=_normalize_money_field(_coerce_field_value(row.get("net_amount"))),
                vat_amount=_normalize_money_field(_coerce_field_value(row.get("vat_amount"))),
                gross_amount=_normalize_money_field(
                    _coerce_field_value(row.get("gross_amount")),
                ),
            ),
        )
    return rows


def _lower_confidence_on_sum_mismatch(result: InvoiceResult) -> list[str]:
    """propOS rule: do not fix sums — only lower confidence."""
    warnings: list[str] = []
    f = result.fields
    if not f.total_net or not f.total_vat or not f.total_gross:
        return warnings
    if (
        f.total_net.value is None
        or f.total_vat.value is None
        or f.total_gross.value is None
    ):
        return warnings
    try:
        net = int(f.total_net.value)
        vat = int(f.total_vat.value)
        gross = int(f.total_gross.value)
    except (TypeError, ValueError):
        return warnings
    if net + vat != gross:
        warnings.append("header_totals_inconsistent")
        low = 0.35
        f.total_net.confidence = min(f.total_net.confidence, low)
        f.total_vat.confidence = min(f.total_vat.confidence, low)
        f.total_gross.confidence = min(f.total_gross.confidence, low)
    return warnings


def _llm_extract_from_text(text: str) -> InvoiceResult:
    user = f"OCR TEXT:\n\n{_truncate_text(text)}"
    raw = chat_json(system=_system_prompt(), user=user)
    fields_raw = raw.get("fields") if isinstance(raw.get("fields"), dict) else {}
    line_items_raw = raw.get("line_items") if isinstance(raw.get("line_items"), list) else []
    vat_raw = raw.get("vat_summary") if isinstance(raw.get("vat_summary"), list) else []

    result = InvoiceResult(
        raw_text=text,
        fields=_parse_fields(fields_raw),
        line_items=_parse_line_items(line_items_raw),
        vat_summary=_parse_vat_summary(vat_raw),
    )
    return result


def extract_invoice_from_file(
    input_path: Path,
    *,
    source_name: str | None = None,
    allow_vision_escalation: bool | None = None,
) -> InvoiceExtractResponse:
    """
    Full sync pipeline: file → OCR → LLM → propOS JSON.

    Raises on OCR failure. Returns status=failed on LLM/validation errors.
    """
    t0 = time.time()
    allow_vlm = (
        settings.invoice_allow_vision_escalation
        if allow_vision_escalation is None
        else allow_vision_escalation
    )
    warnings: list[str] = []

    try:
        ocr_result = ocr_document_sync(
            input_path,
            source_name=source_name,
            allow_vision_escalation=allow_vlm,
        )
    except Exception as exc:
        logger.exception("Invoice OCR failed for %s", input_path.name)
        return InvoiceExtractResponse(
            status="failed",
            processing=InvoiceProcessingInfo(
                elapsed_ms=int((time.time() - t0) * 1000),
                page_count=0,
                warnings=[str(exc)],
            ),
            error={"code": "OCR_FAILED", "message": str(exc)},
        )

    text = (ocr_result.text or "").strip()
    if not text:
        return InvoiceExtractResponse(
            status="failed",
            processing=InvoiceProcessingInfo(
                ocr_engine=ocr_result.engine,
                ocr_confidence=ocr_result.confidence,
                elapsed_ms=int((time.time() - t0) * 1000),
                page_count=ocr_result.page_count,
                warnings=["empty_ocr_text"],
            ),
            error={
                "code": "UNREADABLE",
                "message": "Dokument nieczytelny — brak tekstu OCR.",
            },
        )

    invoice_text = _isolate_first_invoice_text(text)
    if not invoice_text.strip():
        invoice_text = text

    try:
        invoice = _llm_extract_from_text(invoice_text)
    except (TextLLMError, ValidationError) as exc:
        logger.warning("Invoice LLM extraction failed: %s", exc)
        return InvoiceExtractResponse(
            status="failed",
            processing=InvoiceProcessingInfo(
                ocr_engine=ocr_result.engine,
                ocr_confidence=ocr_result.confidence,
                elapsed_ms=int((time.time() - t0) * 1000),
                page_count=ocr_result.page_count,
                warnings=ocr_result.warnings,
            ),
            error={"code": "EXTRACTION_FAILED", "message": str(exc)},
        )

    warnings.extend(ocr_result.warnings)
    warnings.extend(_lower_confidence_on_sum_mismatch(invoice))

    elapsed_ms = int((time.time() - t0) * 1000)
    return InvoiceExtractResponse(
        status="done",
        processing=InvoiceProcessingInfo(
            ocr_engine=ocr_result.engine,
            ocr_confidence=ocr_result.confidence,
            elapsed_ms=elapsed_ms,
            page_count=ocr_result.page_count,
            warnings=warnings,
        ),
        result=invoice,
    )
