"""propOS invoice extraction API — internal + wire schemas."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


# ── Internal extraction models (LLM pipeline) ────────────────────────────────


class FieldValue(BaseModel):
    """Single extracted field with confidence."""

    value: str | int | float | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class PartyFields(BaseModel):
    name: FieldValue | None = None
    nip: FieldValue | None = None
    address: FieldValue | None = None


class InvoiceHeaderFields(BaseModel):
    invoice_number: FieldValue | None = None
    issue_date: FieldValue | None = None
    sale_date: FieldValue | None = None
    due_date: FieldValue | None = None
    currency: FieldValue | None = None
    total_net: FieldValue | None = None
    total_vat: FieldValue | None = None
    total_gross: FieldValue | None = None
    seller: PartyFields | None = None
    buyer: PartyFields | None = None
    payment_method: FieldValue | None = None
    bank_account: FieldValue | None = None
    description: FieldValue | None = None


class InvoiceLineItem(BaseModel):
    name: FieldValue | None = None
    quantity: FieldValue | None = None
    unit: FieldValue | None = None
    unit_price_net: FieldValue | None = None
    vat_rate: FieldValue | None = None
    net_amount: FieldValue | None = None
    vat_amount: FieldValue | None = None
    gross_amount: FieldValue | None = None


class VatSummaryRow(BaseModel):
    vat_rate: FieldValue | None = None
    net_amount: FieldValue | None = None
    vat_amount: FieldValue | None = None
    gross_amount: FieldValue | None = None


class InvoiceProcessingInfo(BaseModel):
    ocr_engine: str | None = None
    extraction_engine: str = "ollama_text"
    ocr_confidence: float | None = Field(None, ge=0.0, le=1.0)
    elapsed_ms: int | None = None
    page_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class InvoiceResult(BaseModel):
    raw_text: str
    fields: InvoiceHeaderFields
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
    vat_summary: list[VatSummaryRow] = Field(default_factory=list)


class InvoiceExtractResponse(BaseModel):
    """Internal pipeline result before propOS wire mapping."""

    status: Literal["done", "failed"] = "done"
    document_type: Literal["invoice"] = "invoice"
    processing: InvoiceProcessingInfo
    result: InvoiceResult | None = None
    error: dict[str, Any] | None = None


# ── propOS wire contract (sections 3.2, 4.1, 4.2) ───────────────────────────


class ProposFieldValue(BaseModel):
    value: str | int | float | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ProposPartyFields(BaseModel):
    name: ProposFieldValue | None = None
    nip: ProposFieldValue | None = None
    address: ProposFieldValue | None = None


class ProposHeaderFields(BaseModel):
    invoice_number: ProposFieldValue | None = None
    issue_date: ProposFieldValue | None = None
    due_date: ProposFieldValue | None = None
    currency: ProposFieldValue | None = None
    total_net: ProposFieldValue | None = None
    total_vat: ProposFieldValue | None = None
    total_gross: ProposFieldValue | None = None
    seller: ProposPartyFields | None = None
    buyer: ProposPartyFields | None = None
    payment_method: ProposFieldValue | None = None
    bank_account: ProposFieldValue | None = None
    description: ProposFieldValue | None = None


class ProposLineItem(BaseModel):
    name: str | int | float | None = None
    quantity: str | int | float | None = None
    unit: str | None = None
    unit_price_net: int | None = None
    total_net: int | None = None
    vat_rate: str | None = None
    total_gross: int | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ProposVatSummaryRow(BaseModel):
    rate: str | None = None
    net: int | None = None
    vat: int | None = None
    gross: int | None = None


class ProposProcessingInfo(BaseModel):
    engine: str | None = None
    duration_ms: int | None = None
    pages: int = 0


class ProposInvoiceResult(BaseModel):
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    fields: ProposHeaderFields
    line_items: list[ProposLineItem] = Field(default_factory=list)
    vat_summary: list[ProposVatSummaryRow] = Field(default_factory=list)
    raw_text: str = ""


class ProposInvoiceError(BaseModel):
    code: str
    message: str


class InvoiceUploadResponse(BaseModel):
    """§3.2 — immediate response on POST upload."""

    job_id: UUID
    status: Literal["queued"] = "queued"


class InvoiceJobResponse(BaseModel):
    """§4.1 / §4.2 — poll result for invoice job."""

    job_id: UUID
    external_id: str | None = None
    document_type: Literal["invoice"] = "invoice"
    status: Literal["queued", "processing", "done", "failed"]
    error: ProposInvoiceError | None = None
    processing: ProposProcessingInfo
    result: ProposInvoiceResult | None = None
