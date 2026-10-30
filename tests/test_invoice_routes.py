"""HTTP tests for propOS invoice API."""

from __future__ import annotations

import time
from io import BytesIO
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.propos_invoice import (
    FieldValue,
    InvoiceExtractResponse,
    InvoiceHeaderFields,
    InvoiceProcessingInfo,
    InvoiceResult,
)
from app.services.invoice_jobs import invoice_job_store


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_sync_returns_200_with_json(client: TestClient) -> None:
    mock_internal = InvoiceExtractResponse(
        status="done",
        processing=InvoiceProcessingInfo(elapsed_ms=100, page_count=1),
        result=InvoiceResult(
            raw_text="test",
            fields=InvoiceHeaderFields(
                invoice_number=FieldValue(value="FV/1", confidence=0.9),
            ),
        ),
    )
    with patch(
        "app.services.invoice_jobs._extract_internal",
        return_value=mock_internal,
    ):
        resp = client.post(
            "/api/v1/invoice/extract",
            data={"external_id": "pi_sync"},
            files={"file": ("faktura.pdf", BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["external_id"] == "pi_sync"
    assert body["result"]["fields"]["invoice_number"]["value"] == "FV/1"
    assert "job_id" in body


def test_upload_returns_202_queued(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/invoice/extract?async=true",
        data={"external_id": "pi_12345"},
        files={"file": ("faktura.pdf", BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert "job_id" in body


def test_poll_done_job(client: TestClient) -> None:
    job_id = invoice_job_store.create(external_id="pi_999")
    internal = InvoiceExtractResponse(
        status="done",
        processing=InvoiceProcessingInfo(elapsed_ms=100, page_count=1),
        result=InvoiceResult(
            raw_text="x",
            fields=InvoiceHeaderFields(
                invoice_number=FieldValue(value="FV/1", confidence=0.9),
            ),
        ),
    )
    invoice_job_store._complete(job_id, internal)  # noqa: SLF001

    resp = client.get(f"/api/v1/invoice/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["external_id"] == "pi_999"
    assert body["result"]["fields"]["invoice_number"]["value"] == "FV/1"


def test_poll_unknown_job_404(client: TestClient) -> None:
    resp = client.get(f"/api/v1/invoice/jobs/{uuid4()}")
    assert resp.status_code == 404


def test_upload_rejects_oversized(client: TestClient) -> None:
    big = b"x" * (21 * 1024 * 1024)
    resp = client.post(
        "/api/v1/invoice/extract",
        files={"file": ("big.pdf", BytesIO(big), "application/pdf")},
    )
    assert resp.status_code == 413


def test_async_flow_with_mocked_extraction(client: TestClient) -> None:
    mock_internal = InvoiceExtractResponse(
        status="done",
        processing=InvoiceProcessingInfo(
            ocr_engine="fast_ocr",
            elapsed_ms=50,
            page_count=1,
            ocr_confidence=0.9,
        ),
        result=InvoiceResult(
            raw_text="test",
            fields=InvoiceHeaderFields(
                invoice_number=FieldValue(value="FV/2", confidence=0.95),
            ),
        ),
    )

    async def fake_run(job_id, upload_path, *, source_name):  # noqa: ANN001
        invoice_job_store._complete(job_id, mock_internal)  # noqa: SLF001
        upload_path.unlink(missing_ok=True)

    with patch.object(invoice_job_store, "schedule") as schedule_mock:
        def schedule(job_id, path, *, source_name):  # noqa: ANN001
            import asyncio

            asyncio.get_event_loop().create_task(
                fake_run(job_id, path, source_name=source_name),
            )

        schedule_mock.side_effect = schedule

        upload = client.post(
            "/api/v1/invoice/extract?async=true",
            data={"external_id": "pi_async"},
            files={"file": ("f.pdf", BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
        assert upload.status_code == 202
        job_id = upload.json()["job_id"]

        for _ in range(50):
            poll = client.get(f"/api/v1/invoice/jobs/{job_id}")
            if poll.json().get("status") == "done":
                break
            time.sleep(0.05)

    assert poll.status_code == 200
    assert poll.json()["result"]["fields"]["invoice_number"]["value"] == "FV/2"
