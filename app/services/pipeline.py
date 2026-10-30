"""End-to-end: file → ProcessingResult → JSON + TXT on disk."""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from PIL import Image

from app.core.config import settings
from app.core.constants import JobStatus, SourceKind
from app.models.schemas import JobProgress, ProcessingResult
from app.services.ocr_lines import aggregate_lines_from_metas, parse_lines_from_meta
from app.services.document_segmentation import (
    DocumentRegion,
    _prefer_rotation_by_reading_order,
    detect_orientation,
    extract_documents,
    fix_inverted_reading_order,
    looks_garbled_ocr,
    looks_inverted_reading_order,
    looks_like_ksef_merged_text,
    looks_like_merged_receipts,
    resplit_bbox,
    extract_ksef_column_regions,
)
from app.services.document_classifier import (
    DocumentKind,
    classify_document,
    consolidate_page_blocks,
    format_document_stream,
    is_invoice_continuation,
    split_multi_document_block,
)
from app.services.document_language import (
    apply_polish_postprocess,
    effective_primary_ocr_lang,
    languages_for_result,
)
from app.services.result_paths import allocate_result_dir, get_result_dir


def _looks_like_real_ksef_invoice(text: str) -> bool:
    """True only when full-page text really is a 2-column KSeF invoice visualization.

    KSeF invoices (e-Urząd Skarbowy PDF visualization) always carry an
    explicit KSeF identifier or the word "WIZUALIZACJA".  Plain fuel-station
    receipts/invoices that happen to have "SPRZEDAWCA" and "NABYWCA" fields
    are NOT KSeF invoices, so we require the explicit KSeF marker first.
    The broader SPRZEDAWCA+NABYWCA heuristic caused false positives on
    side-by-side receipt scans.
    """
    if not text or len(text) < 80:
        return False
    u = text.upper().replace(" ", "")
    # Primary: explicit KSeF markers always present in genuine KSeF PDFs.
    if "KSEF" in u or "WIZUALIZACJA" in text.upper() or "QR.KSEF" in u:
        return True
    # Secondary: interleaved SPRZEDAWCA / NABYWCA on the SAME LINE is a very
    # strong signal (cannot happen in a single-column receipt).
    for ln in text.splitlines():
        ln_u = ln.upper()
        if "SPRZEDAWCA" in ln_u and "NABYWCA" in ln_u:
            return True
    return False


def _looks_like_letter_or_singledoc(text: str) -> bool:
    """True when a page reads like a continuous letter / one-doc layout."""
    if not text:
        return False
    u = text.upper()
    if any(
        m in u
        for m in (
            "SZANOWNI PAŃSTWO",
            "SZANOWNI PANSTWO",
            "SZANOWNI",
            "Z POWAŻANIEM",
            "Z POWAZANIEM",
            "DYREKTOR ZARZ",
            "BIURO SPRZEDA",
            "TAXE PERCUE",
            "TAXE PERÇUE",
            "OPŁATA POBRANA",
            "OPLATA POBRANA",
            "OBOWIĄZKOWEGO UBEZPIECZE",
            "OBOWIAZKOWEGO UBEZPIECZE",
            "UMOWA UBEZPIECZE",
            "VIENNA INSURANCE",
            "COMPENSA",
            # Government / official forms that physically span two columns
            "DOKUMENT OBLICZENIA",
            "DOKUM OBLICZENIA",
            "ZASOB GEODEZYJNY",
            "ZASÓB GEODEZYJNY",
            "KARTOGRAFICZNY",
            "OBLICZENIA OPLATY",
            "OBLICZENIA OPŁATY",
        )
    ):
        return True
    # Numbered multi-section documents (e.g. numbered forms 1. 2. 3. …) are
    # typically single documents.  Require at least 5 numbered sections and no
    # receipt/invoice markers to avoid false-positives on itemised invoices.
    if "PARAGON FISKALNY" not in u and "FAKTURA" not in u:
        numbered = sum(
            1 for i in range(1, 12)
            if f"\n{i}." in text or f"\n{i} ." in text or f"\n{i})" in text
        )
        if numbered >= 5:
            return True
    return False


def _looks_like_two_column_invoice_pair(left_text: str, right_text: str) -> bool:
    """Heuristic: two column crops are halves of one invoice, not two docs."""
    left = left_text.strip()
    right = right_text.strip()
    if not left or not right:
        return False
    lk = classify_document(left)
    rk = classify_document(right)
    if lk == DocumentKind.FAKTURA and rk == DocumentKind.FAKTURA:
        return True
    if is_invoice_continuation(left, right) or is_invoice_continuation(right, left):
        return True
    lu = left.upper()
    ru = right.upper()
    # Typical split invoice: one column has seller header, second has buyer/totals.
    if (
        ("SPRZEDAWCA" in lu and "NABYWCA" in ru)
        or ("SPRZEDAWCA" in ru and "NABYWCA" in lu)
    ):
        if "PARAGON FISKALNY" not in lu and "PARAGON FISKALNY" not in ru:
            return True
    return False


def _split_mixed_invoice_receipt_block(text: str) -> list[str]:
    """Split one OCR block when invoice and receipt headers are mixed together."""
    t = text.strip()
    if not t:
        return []
    u = t.upper()
    if "FAKTURA" not in u:
        return [t]
    # OCR may emit PARAGON as PARIAGON; cover both.
    markers = ("PARAGON FISKALNY", "PARIAGON FISKALNY", "PARAGON")
    cut_positions = [u.find(m, 120) for m in markers]
    cut = min((p for p in cut_positions if p >= 120), default=-1)
    if cut < 0:
        return [t]
    left = t[:cut].strip()
    right = t[cut:].strip()
    if left and right:
        return [left, right]
    return [t]


def _reconstruct_ksef_header(header_lines: list[str]) -> str:
    """Reconstruct an interleaved KSeF header into correct reading order.

    The full-page OCR alternates between left and right header columns, producing
    lines like: OFFLINE, Nr 26/05/2026, 2026-05-04, Faktura, Data wystawienia, …

    This function recognises and reassembles the canonical KSeF header structure:
        OFFLINE (if present)
        Faktura Nr X/Y/ZZZZ
        Data wystawienia: YYYY-MM-DD
        Data sprzedaży: YYYY-MM-DD
    """
    # Collect pieces
    has_offline = any("OFFLINE" in ln.upper() for ln in header_lines)
    faktura_line = next(
        (ln for ln in header_lines if "FAKTURA" in ln.upper()), None
    )
    nr_line = next(
        (ln for ln in header_lines if re.match(r"\s*Nr\s+\S", ln, re.IGNORECASE)), None
    )
    dates = [ln.strip() for ln in header_lines if re.search(r"\d{4}-\d{2}-\d{2}", ln)]

    date_issue_label = next(
        (ln.strip() for ln in header_lines if "WYSTAWIEN" in ln.upper()), "Data wystawienia"
    )
    date_sale_label = next(
        (ln.strip() for ln in header_lines if "SPRZED" in ln.upper() and ln.strip() != faktura_line), "Data sprzedaży"
    )

    lines: list[str] = []
    if has_offline:
        lines.append("OFFLINE")

    # Invoice identifier line
    if faktura_line and nr_line:
        faktura_clean = faktura_line.strip().rstrip(":")
        lines.append(f"{faktura_clean} {nr_line.strip()}")
    elif faktura_line:
        lines.append(faktura_line.strip())

    # Date lines
    for idx, (label, date) in enumerate(
        zip([date_issue_label, date_sale_label], dates[:2])
    ):
        label_clean = label.rstrip(":")
        lines.append(f"{label_clean}: {date}")

    return "\n".join(lines).strip() if lines else ""


def _ksef_smart_join(left: str, right: str, full_page: str) -> str:
    """Join two KSeF column crops intelligently using the full-page OCR for the header.

    Problem: the column split cuts full-width headings (e.g. "Faktura" becomes "Fa"
    in the left crop and "ktura" in the right crop).  The full-page OCR reads heading
    text correctly but interleaves seller and buyer rows (which appear side-by-side).

    Strategy:
    1. Extract header lines from full-page OCR (everything before the first
       "Sprzedawca" / "Nabywca" row where interleaving begins).
    2. If full-page header is short/empty or looks interleaved, reconstruct it
       using the KSeF-specific header pattern detection.
    3. Extract seller body from the left column crop (from "Sprzedawca" onwards).
    4. Extract buyer body from the right column crop (from "Nabywca" onwards).
    5. Assemble: header + seller_body + buyer_body.
    """
    # ── Step 1: seller/buyer split points ────────────────────────────────────
    left_lines = left.splitlines()
    left_body_start = 0
    for i, ln in enumerate(left_lines):
        if "SPRZEDAWCA" in ln.upper():
            left_body_start = i
            break

    right_lines = right.splitlines()
    right_body_start = 0
    for i, ln in enumerate(right_lines):
        if "NABYWCA" in ln.upper():
            right_body_start = i
            break

    left_body = "\n".join(left_lines[left_body_start:]).strip()
    right_body = "\n".join(right_lines[right_body_start:]).strip()

    # ── Step 2: header from full-page OCR ────────────────────────────────────
    header_lines: list[str] = []
    for ln in full_page.splitlines():
        lu = ln.upper()
        if "SPRZEDAWCA" in lu or "NABYWCA" in lu:
            break
        header_lines.append(ln)
    header = "\n".join(header_lines).strip()

    # ── Step 3: detect whether the full-page header is garbled / interleaved ──
    # Signs of interleaving: a short alpha fragment (word split at column boundary),
    # or duplicate date patterns (right-col dates appear before their left-col labels),
    # or "Faktura" appearing after the first two lines (column-alternating reading).
    short_frag_count = sum(
        1 for ln in header_lines
        if ln.strip() and len(ln.strip()) < 5 and ln.strip().isalpha()
    )
    date_count = sum(
        1 for ln in header_lines if re.search(r"\d{4}-\d{2}-\d{2}", ln)
    )
    faktura_idx = next(
        (i for i, ln in enumerate(header_lines) if "FAKTURA" in ln.upper()),
        len(header_lines),
    )
    header_garbled = short_frag_count >= 1 or date_count >= 2 or faktura_idx > 2

    if not header or len(header) < 15 or header_garbled:
        # ── Step 3a: try to reconstruct KSeF header from full-page lines ────
        if header_lines:
            reconstructed = _reconstruct_ksef_header(header_lines)
            if reconstructed and len(reconstructed) >= 20:
                header = reconstructed
        # ── Step 3b: if reconstruction failed, fall back to column crops ────
        if not header or len(header) < 15:
            left_header = "\n".join(left_lines[:left_body_start]).strip()
            right_header = "\n".join(right_lines[:right_body_start]).strip()
            left_hdr_lines = [ln for ln in left_header.splitlines() if ln.strip()]
            right_hdr_lines = [ln for ln in right_header.splitlines() if ln.strip()]

            # Join short alpha fragment from end of left column with lowercase
            # continuation found anywhere in the right column header.
            if left_hdr_lines and len(left_hdr_lines[-1].strip()) < 5 and left_hdr_lines[-1].strip().isalpha():
                for i, rln in enumerate(right_hdr_lines):
                    if rln.strip() and rln.strip()[:1].islower():
                        joined = left_hdr_lines[-1].strip() + rln.strip()
                        left_hdr_lines[-1] = joined
                        right_hdr_lines.pop(i)
                        break

            # Combine unique lines
            seen: set[str] = set()
            merged_hdr: list[str] = []
            for ln in left_hdr_lines + right_hdr_lines:
                key = ln.strip().upper()
                if key and key not in seen:
                    seen.add(key)
                    merged_hdr.append(ln)
            header = "\n".join(merged_hdr).strip()

    # ── Step 4: assemble ──────────────────────────────────────────────────────
    parts = [p for p in [header, left_body, right_body] if p]
    if len(parts) < 2:
        # Fallback: raw join if we couldn't extract useful sections
        return (left.strip() + "\n\n" + right.strip()).strip()
    return "\n\n".join(parts)


def _normalize_page_document_blocks(
    doc_texts: list[str],
    doc_metas: list[dict],
    *,
    page_idx: int,
) -> tuple[list[str], list[dict]]:
    """
    Generic page-level normalization:
    1) split accidentally glued docs inside one OCR block,
    2) merge multi-column invoice-only pages back to one document.
    """
    split_texts: list[str] = []
    split_metas: list[dict] = []
    for txt, meta in zip(doc_texts, doc_metas):
        parts = split_multi_document_block(txt)
        if len(parts) == 1:
            parts = _split_mixed_invoice_receipt_block(txt)
        if len(parts) > 1:
            for i, part in enumerate(parts, 1):
                m = dict(meta)
                m["split_part"] = i
                m["split_of"] = len(parts)
                m["text_chars"] = len(part)
                m["_no_auto_merge"] = True
                split_texts.append(part)
                split_metas.append(m)
        else:
            split_texts.append(txt)
            split_metas.append(meta)

    if len(split_texts) <= 1:
        return split_texts, split_metas

    kinds = [classify_document(t) for t in split_texts]
    has_receipt = any(k == DocumentKind.PARAGON for k in kinds)
    joined_upper = "\n".join(split_texts).upper()
    invoiceish = (
        "FAKTURA" in joined_upper
        or "KSEF" in joined_upper
        or ("SPRZEDAWCA" in joined_upper and "NABYWCA" in joined_upper)
        or "DO ZAPLATY" in joined_upper
        or "DO ZAPŁATY" in joined_upper
    )
    if invoiceish and not has_receipt:
        # Multi-region invoice-like page: merge into one page-level document.
        merged = "\n\n---\n\n".join(t.strip() for t in split_texts if t.strip())
        m0 = dict(split_metas[0]) if split_metas else {"page": page_idx}
        m0["merged_from"] = len(split_texts)
        m0["layout"] = "invoice_page_merged"
        m0["text_chars"] = len(merged)
        return [merged], [m0]
    return split_texts, split_metas
from app.services.export import write_results
from app.services.scan_enhance import prepare_crop_for_ocr, prepare_page_for_ocr
from app.services.gpu import (
    allocate_gpu,
    get_bulk_ocr_gpu_ids,
    get_primary_ocr_gpu_id,
    should_use_page_parallel,
    should_use_parallel_bulk,
)
from app.services.gpu_pool import ocr_gpu_semaphores
from app.services.ingest import IngestError
from app.services.zip_streaming import (
    ensure_zip_member_extracted,
    plan_zip_members_from_archive,
    release_zip_member_temp,
)
from app.services.jobs import job_store
from app.services.ocr import _smart_pdf_crop_pipeline_pil, ocr_image
from app.services.pipeline_profile import (
    IMAGE_PROFILE,
    PDF_CROP_PROFILE,
    PDF_LINE_PROFILE,
    ZIP_MEMBER_PROFILE,
    use_profile,
)
from app.services.oom import with_oom_retry
from app.services.parallel_ocr import OcrJob, run_ocr_jobs
from app.services.pdf_render import pdf_page_count, render_pages, render_pdf_page_pil
from app.services.zip_checkpoint import (
    ZipCheckpointStore,
    checkpoint_path,
    member_has_results,
)
from app.services.zip_queue import ZipMember, ZipProcessingQueue, plan_zip_members
from app.services.zip_adaptive_queue import AdaptiveZipQueue

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"})

_bulk_warmed = False


def _source_kind(path: Path) -> SourceKind:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return SourceKind.PDF
    if suffix in IMAGE_EXTENSIONS:
        return SourceKind.IMAGE
    if suffix == ".zip":
        return SourceKind.ZIP
    raise ValueError(f"Unsupported source: {suffix}")


def _collect_page_images(
    input_path: Path,
    work_dir: Path,
) -> tuple[list[Path], dict | None]:
    kind = _source_kind(input_path)
    if kind == SourceKind.IMAGE:
        return [input_path], None
    if kind == SourceKind.PDF:
        pages_dir = work_dir / "pages"
        if settings.pdf_dpi_routing_enabled:
            from app.services.pdf_render import render_pages_routed

            paths, routing_meta = render_pages_routed(input_path, pages_dir)
            return paths, routing_meta
        return render_pages(input_path, pages_dir, dpi=settings.pdf_dpi), None
    raise ValueError(f"ZIP must be unpacked before page collection: {input_path.name}")


def _languages_from_config(text: str | None = None) -> list[str]:
    return languages_for_result(text)


def should_use_zip_simple_receipt(pdf_path: Path, work_dir: Path) -> bool:
    """
  ZIP-only fast path: 1–N page receipt PDFs (default N=2) without bulk segmentation.

  Multi-page invoices (e.g. SKMBT, 5+ pages) use bulk. Optional region probe
  catches dense single-page scans with several side-by-side receipts.
    """
    if not settings.zip_simple_receipt_enabled:
        return False
    try:
        pages = pdf_page_count(pdf_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("zip simple: page count failed for %s: %s", pdf_path.name, exc)
        return False
    if pages < 1 or pages > settings.zip_simple_receipt_max_pages:
        return False
    if not settings.zip_simple_receipt_probe_regions:
        return True
    for page_index in range(pages):
        if _zip_simple_page_region_count(pdf_path, work_dir, page_index) > (
            settings.zip_simple_receipt_max_regions
        ):
            return False
    return True


def _zip_simple_page_region_count(
    pdf_path: Path, work_dir: Path, page_index: int,
) -> int:
    """Light render + segmentation count (only when probe_regions is enabled)."""
    probe_dpi = (
        settings.pdf_dpi_probe
        if settings.pdf_dpi_routing_enabled
        else settings.zip_simple_pdf_dpi
    )
    page_pil = render_pdf_page_pil(
        pdf_path, page_index=page_index, dpi=probe_dpi,
    )
    try:
        page_for_ocr = prepare_page_for_ocr(page_pil)
        regions = extract_documents(
            page_pil,
            ocr_page_image=page_for_ocr,
            enable_segmentation=True,
            enable_auto_rotate=settings.pdf_auto_rotate,
            min_area_pct=settings.pdf_min_region_area_pct,
            min_side_pct=settings.pdf_min_region_side_pct,
            max_regions=settings.pdf_max_regions_per_page,
            osd_min_confidence=settings.pdf_osd_min_confidence,
        )
        return len(regions)
    finally:
        page_pil.close()


def _zip_simple_receipt_region_count(pdf_path: Path, work_dir: Path) -> int:
    """Backward-compatible: region count for page 0 only."""
    return _zip_simple_page_region_count(pdf_path, work_dir, 0)


def _zip_simple_ocr_one_page(
    input_path: Path,
    page_index: int,
    *,
    source_label: str,
    allow_vision_escalation: bool,
    gpu_id: int,
    sem,
    dpi_analysis: object | None = None,
) -> tuple[str, float | None, dict, bool, int, bool, bool]:
    """OCR one PDF page; returns text, conf, meta, empty_retry, passes, escalated, empty_detection."""
    from app.services.pdf_dpi_routing import resolve_page_dpi

    dpi_decision = resolve_page_dpi(
        input_path, page_index, analysis=dpi_analysis,
    )
    page_pil = render_pdf_page_pil(
        input_path, page_index=page_index, dpi=dpi_decision.dpi,
    )
    page_for_ocr = prepare_page_for_ocr(page_pil)
    page_pil.close()

    ocr_meta_base = {"dpi_routing": dpi_decision.to_dict()}
    text, confidence, ocr_meta = _zip_simple_ocr_pil(
        page_for_ocr,
        source_label=source_label,
        allow_vision_escalation=allow_vision_escalation,
        gpu_id=gpu_id,
        sem=sem,
    )
    ocr_passes = int(ocr_meta.get("ocr_passes", 1))
    empty_detection = bool(ocr_meta.get("empty_detection", False))
    escalated = bool(ocr_meta.get("escalated", False))
    empty_retry = False

    if (
        settings.zip_simple_empty_retry
        and not text.strip()
        and ocr_meta.get("pipeline") == "pdf_crop_fast"
        and ocr_meta.get("n_boxes", 0) == 0
    ):
        empty_detection = True
        strong_page = prepare_crop_for_ocr(page_for_ocr.copy(), force_strong=True)
        text2, conf2, meta2 = _zip_simple_ocr_pil(
            strong_page,
            source_label=f"{source_label}:empty_retry",
            allow_vision_escalation=allow_vision_escalation,
            gpu_id=gpu_id,
            sem=sem,
        )
        strong_page.close()
        if text2.strip() or meta2.get("n_boxes", 0) > 0:
            ocr_passes += int(meta2.get("ocr_passes", 1))
            escalated = escalated or bool(meta2.get("escalated", False))
            text, confidence, ocr_meta = text2, conf2, meta2
            empty_retry = True
            ocr_meta["empty_retry"] = True

    page_for_ocr.close()
    if isinstance(ocr_meta, dict):
        ocr_meta = {**ocr_meta, **ocr_meta_base}
    return text, confidence, ocr_meta, empty_retry, ocr_passes, escalated, empty_detection


def _zip_simple_ocr_pil(
    page_for_ocr: Image.Image,
    *,
    source_label: str,
    allow_vision_escalation: bool,
    gpu_id: int,
    sem,
) -> tuple[str, float | None, dict]:
    def _call() -> tuple[str, float | None, dict]:
        with sem, use_profile(ZIP_MEMBER_PROFILE):
            return _smart_pdf_crop_pipeline_pil(
                page_for_ocr,
                settings.ocr_lang,
                allow_vision_escalation=allow_vision_escalation,
                gpu_id=gpu_id,
            )

    if settings.ocr_oom_retry:
        return with_oom_retry(
            _call,
            gpu_id=gpu_id,
            max_retries=settings.ocr_oom_retry_max,
            label=f"zip_simple[{source_label}]",
        )
    return _call()


def _process_zip_simple_receipt_pdf(
    input_path: Path,
    work_dir: Path,
    *,
    source_name: str | None = None,
    allow_vision_escalation: bool | None = None,
    pinned_gpu: int | None = None,
) -> ProcessingResult:
    """
    ZIP receipt PDF (1–N pages): per-page PIL → RapidOCR in RAM, no bulk segmentation.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    total_pages = pdf_page_count(input_path)
    max_pages = settings.zip_simple_receipt_max_pages
    if total_pages < 1 or total_pages > max_pages:
        raise ValueError(
            f"zip_simple_receipt expects 1–{max_pages} pages, got {total_pages}: "
            f"{input_path.name}",
        )

    gpu_id = pinned_gpu if pinned_gpu is not None else get_primary_ocr_gpu_id()
    sem = ocr_gpu_semaphores([gpu_id])[gpu_id]
    t0 = time.perf_counter()
    vision_ok = False if allow_vision_escalation is None else allow_vision_escalation
    base_label = source_name or input_path.name

    page_texts: list[str] = []
    documents: list[dict] = []
    confidences: list[float] = []
    engines_used: set[str] = set()
    ocr_passes = 0
    empty_retry = False
    empty_detection = False
    escalated = False
    dpi_analysis = None
    if settings.pdf_dpi_routing_enabled:
        from app.services.pdf_dpi_routing import analyze_pdf_dpi

        dpi_analysis = analyze_pdf_dpi(input_path)

    for page_index in range(total_pages):
        page_label = (
            base_label if total_pages == 1 else f"{base_label}:p{page_index + 1}"
        )
        text, confidence, ocr_meta, page_retry, passes, page_esc, page_empty = (
            _zip_simple_ocr_one_page(
                input_path,
                page_index,
                source_label=page_label,
                allow_vision_escalation=vision_ok,
                gpu_id=gpu_id,
                sem=sem,
                dpi_analysis=dpi_analysis,
            )
        )
        text = text.strip()
        page_texts.append(text)
        ocr_passes += passes
        empty_retry = empty_retry or page_retry
        empty_detection = empty_detection or page_empty
        escalated = escalated or page_esc
        if confidence is not None:
            confidences.append(confidence)
        engines_used.add(ocr_meta.get("engine", "fast_ocr"))
        documents.append({
            "page": page_index + 1,
            "document": 1,
            "text_chars": len(text),
            **ocr_meta,
        })

    if total_pages == 1:
        merged_text = page_texts[0] if page_texts else ""
    else:
        merged_text = _merge_documents(page_texts)
        if not merged_text.strip():
            merged_text = "\n\n".join(t for t in page_texts if t.strip())

    processing_ms = int((time.perf_counter() - t0) * 1000)
    elapsed = round(processing_ms / 1000.0, 3)
    mean_confidence = (
        sum(confidences) / len(confidences) if confidences else None
    )
    engine = next(iter(engines_used), "fast_ocr")
    warnings: list[str] = []
    if not merged_text.strip():
        warnings.append("No text extracted from document")

    structured_lines = aggregate_lines_from_metas(documents, page_key="page")

    return ProcessingResult(
        source=source_name or input_path.name,
        source_kind=SourceKind.PDF,
        processed_at=datetime.now(UTC),
        page_count=total_pages,
        language_detected=_languages_from_config(merged_text),
        text=merged_text.strip(),
        lines=structured_lines,
        confidence=round(mean_confidence, 4) if mean_confidence is not None else None,
        engine=engine,
        warnings=warnings,
        metadata={
            "pipeline": "zip_simple_receipt",
            "ocr_mode": "full_page_fast",
            "ocr_in_memory": True,
            "processing_ms": processing_ms,
            "escalated": escalated,
            "empty_detection": empty_detection,
            "ocr_passes": ocr_passes,
            "empty_retry": empty_retry,
            "ocr_gpu_ids": [gpu_id],
            "ocr_pinned_gpu": pinned_gpu,
            "ocr_parallel": False,
            "ocr_page_parallel": False,
            "ocr_batch_sec": elapsed,
            "pdf_dpi": (
                dpi_analysis.document_decision.dpi
                if dpi_analysis is not None
                else settings.zip_simple_pdf_dpi
            ),
            "dpi_routing": (
                dpi_analysis.to_dict() if dpi_analysis is not None else None
            ),
            "documents": documents,
            "documents_total": len(documents),
        },
    )


def _persist_region_image(
    image: Image.Image, work_dir: Path, page_idx: int, doc_idx: int,
) -> Path:
    docs_dir = work_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / f"page_{page_idx:04d}_doc_{doc_idx:02d}.png"
    # Crop comes from an already-enhanced page render; avoid double darkening.
    image.save(out, format="PNG")
    return out


def _receipt_has_footer(text: str) -> bool:
    u = text.upper()
    return any(
        m in u
        for m in (
            "SUMA PLN", "SUMA:", "SUMA :", "NR TRANSAKCJI", "ROZLICZENIE",
            "ZAPLACONO", "ZAPŁACONO", "DO ZAPLATY", "ZAPLACNO",
        )
    )


def _receipt_issuer_tag(text: str) -> str | None:
    """Coarse vendor label — used to avoid merging different stacked receipts."""
    u = text.upper()
    for tag in (
        "EMPIK", "AMIC", "CARREFOUR", "CHINKALI", "NORD", "SKLAD",
        "MU OIL", "MJ OIL", "M OIL", "BALTICA", "OGRODY",
    ):
        if tag in u:
            return tag
    if "PARAGON FISKALNY" in u:
        return "PARAGON"
    if "FAKTURA" in u:
        return "FAKTURA"
    return None


def _receipt_has_header(text: str) -> bool:
    u = text.upper()
    return any(m in u for m in ("FAKTURA", "PARAGON FISKALNY", "SKLAD", "EMPIK", "CARREFOUR"))


def _looks_incomplete_receipt(text: str) -> bool:
    """Fragment of a receipt/invoice — should be merged with neighbours."""
    t = text.strip()
    if len(t) < 30:
        return True
    has_header = _receipt_has_header(t)
    has_footer = _receipt_has_footer(t)
    # Beginning of receipt without totals
    if has_header and not has_footer:
        return True
    # Middle/end slice without shop header (continuation of previous crop)
    if not has_header and has_footer and len(t) < 850:
        return True
    if not has_header and not has_footer and len(t) < 450:
        return True
    return False


def _is_same_receipt_continuation(prev: str, nxt: str) -> bool:
    """Detect two OCR blocks that are one physical receipt split by segmentation."""
    ua, ub = prev.upper(), nxt.upper()

    # If either block looks inverted (upside-down), do NOT merge — let
    # _retry_inverted_180 fix the rotation first.
    if looks_inverted_reading_order(prev) or looks_inverted_reading_order(nxt):
        return False

    # Same-store continuation via shared store name markers
    store_pairs = [
        ("CHINKALI", ("CHINKALI", "NORD", "PELME", "ROZLICZENIE", "PELMENI", "RODZINKI")),
        ("PELMENI", ("PELMENI", "CHINKALI", "ROZLICZENIE", "RODZINKI")),
        ("DLA", ("PARAGON", "CHINKALI", "NORD")),
        ("AMIC", ("AMIC",)),
        ("CARREFOUR", ("CARREFOUR",)),
        ("EMPIK", ("EMPIK",)),
        ("LIDL", ("LIDL",)),
        ("BIEDRONKA", ("BIEDRONKA",)),
        ("ZABKA", ("ZABKA",)),
        ("ROSSMANN", ("ROSSMANN",)),
    ]
    for anchor, companions in store_pairs:
        if anchor in ua and any(c in ub for c in companions):
            # Extra guard: if nxt has its OWN complete receipt header+footer,
            # it is a separate document, not a continuation.
            if _receipt_has_header(nxt) and _receipt_has_footer(nxt) and len(nxt) >= 300:
                return False
            return True

    prev_tag = _receipt_issuer_tag(prev)
    nxt_tag = _receipt_issuer_tag(nxt)
    if prev_tag and nxt_tag and prev_tag != nxt_tag:
        return False
    # Head + tail of one receipt (next slice has no new shop header)
    if _receipt_has_header(prev) and not _receipt_has_footer(prev):
        if _receipt_has_footer(nxt) or "SUMA" in ub or "ZAPLACONO" in ub:
            if _receipt_has_header(nxt) and nxt_tag and prev_tag and nxt_tag != prev_tag:
                return False
            if _receipt_has_header(nxt) and nxt_tag and not prev_tag:
                return False
            return True
    return False


def _merge_receipt_continuations(
    texts: list[str], metas: list[dict],
) -> tuple[list[str], list[dict]]:
    """Join consecutive crops that are one receipt (e.g. greeting + items)."""
    if len(texts) <= 1:
        return texts, metas

    merged_texts: list[str] = [texts[0].strip()]
    merged_metas: list[dict] = [dict(metas[0])]
    for text, meta in zip(texts[1:], metas[1:]):
        t = text.strip()
        if not t:
            continue
        prev_meta = merged_metas[-1]
        # Never re-merge docs that were explicitly split (e.g. ksef columns).
        if prev_meta.get("_no_auto_merge") or meta.get("_no_auto_merge"):
            merged_texts.append(t)
            merged_metas.append(dict(meta))
            continue
        same_doc_geom = _likely_same_physical_document(prev_meta, meta)
        if same_doc_geom and _is_same_receipt_continuation(merged_texts[-1], t):
            merged_texts[-1] = merged_texts[-1] + "\n" + t
            merged_metas[-1]["text_chars"] = len(merged_texts[-1])
            merged_metas[-1]["continuations_merged"] = (
                merged_metas[-1].get("continuations_merged", 1) + 1
            )
        else:
            merged_texts.append(t)
            merged_metas.append(dict(meta))
    return merged_texts, merged_metas


def _merge_incomplete_fragments(
    texts: list[str], metas: list[dict],
) -> tuple[list[str], list[dict]]:
    """Join consecutive OCR fragments that belong to one receipt."""
    if len(texts) <= 1:
        return texts, metas

    merged_texts: list[str] = []
    merged_metas: list[dict] = []
    buffer: list[str] = []
    buffer_meta: dict | None = None

    def flush() -> None:
        nonlocal buffer, buffer_meta
        if not buffer:
            return
        combined = "\n".join(buffer)
        if buffer_meta is None:
            buffer = []
            return
        entry = {**buffer_meta, "text_chars": len(combined), "fragments_merged": len(buffer)}
        if _is_meaningful_document_text(combined):
            merged_texts.append(combined)
            merged_metas.append(entry)
        buffer = []
        buffer_meta = None

    for text, meta in zip(texts, metas):
        t = text.strip()
        if not t:
            continue
        # Documents marked _no_auto_merge bypass fragment buffering entirely.
        if meta.get("_no_auto_merge"):
            flush()
            if _is_meaningful_document_text(t):
                merged_texts.append(t)
                merged_metas.append(meta)
            continue
        # Keep fragments from different physical regions separated.
        if buffer and buffer_meta is not None and not _likely_same_physical_document(buffer_meta, meta):
            flush()
        # New complete document → flush buffered tail/head fragments first.
        if (
            buffer
            and _receipt_has_header(t)
            and _receipt_has_footer(t)
            and len(t) >= 500
        ):
            flush()
        if _looks_incomplete_receipt(t):
            buffer.append(t)
            buffer_meta = meta if buffer_meta is None else buffer_meta
        else:
            flush()
            if _is_meaningful_document_text(t):
                merged_texts.append(t)
                merged_metas.append(meta)

    flush()
    return merged_texts, merged_metas


def _is_meaningful_document_text(text: str) -> bool:
    """Drop OCR noise fragments between real receipts."""
    t = text.strip()
    if len(t) < 40:
        return False
    if len(t) >= 100:
        return True
    upper = t.upper()
    markers = ("PARAGON", "FAKTURA", "SKLAD", "NIP", "SUMA PLN", "EMPIK", "CARREFOUR")
    return any(m in upper for m in markers)


def _merge_documents(texts: list[str], metas: list[dict] | None = None) -> str:
    # Let the document_classifier handle:
    # - łączenie fragmentów tej samej faktury/paragonu (w obrębie strony),
    # - podział jednego bloku zawierającego kilka faktur,
    # - etykiety „Faktura i/N”, „Paragon i/N”, „List i/N”,
    # - usuwanie wizualnych separatorów (---, ___) bez ingerencji w treść.
    # We still keep a light guard against pure noise by requiring some length.
    base = [t for t in texts if t and t.strip() and len(t.strip()) >= 40]
    if not base:
        return ""
    safe_metas = metas or []
    if safe_metas:
        pairs = [
            (t.strip(), m)
            for t, m in zip(texts, safe_metas)
            if t and t.strip() and len(t.strip()) >= 40
        ]
        base = [t for t, _ in pairs]
        safe_metas = [m for _, m in pairs]
    page_joined = consolidate_page_blocks(base, safe_metas, merge_same_page=True)
    if not page_joined:
        return ""
    result = format_document_stream(page_joined, separator=settings.pdf_document_separator)
    return apply_polish_postprocess(result)


def _meta_bbox(meta: dict) -> tuple[int, int, int, int] | None:
    bb = meta.get("bbox")
    if not isinstance(bb, list) or len(bb) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(v) for v in bb]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _likely_same_physical_document(prev_meta: dict, next_meta: dict) -> bool:
    """
    Heuristic gate to prevent over-merging neighbouring receipts.

    We only merge when both fragments belong to the same page and their bboxes
    strongly overlap in X (same column/receipt strip) with a small vertical gap.
    """
    if int(prev_meta.get("page", 0) or 0) != int(next_meta.get("page", 0) or 0):
        return False

    prev_bb = _meta_bbox(prev_meta)
    next_bb = _meta_bbox(next_meta)
    if prev_bb is None or next_bb is None:
        # Without geometry, keep old behavior.
        return True

    px0, py0, px1, py1 = prev_bb
    nx0, ny0, nx1, ny1 = next_bb
    p_w = max(1, px1 - px0)
    n_w = max(1, nx1 - nx0)
    x_overlap = max(0, min(px1, nx1) - max(px0, nx0))
    x_overlap_ratio = x_overlap / max(1, min(p_w, n_w))

    # Similar column and stacked vertically.
    if x_overlap_ratio >= 0.55:
        y_gap = ny0 - py1
        if -80 <= y_gap <= 240:
            return True

    # Nested bbox (fragment inside larger region) -> continuation.
    if (
        nx0 >= px0 - 8 and nx1 <= px1 + 8 and ny0 >= py0 - 8 and ny1 <= py1 + 8
    ) or (
        px0 >= nx0 - 8 and px1 <= nx1 + 8 and py0 >= ny0 - 8 and py1 <= ny1 + 8
    ):
        return True

    return False


_bulk_warmup_lock = threading.Lock()


def _ensure_bulk_gpu_warmup() -> None:
    """Lazy-init RapidOCR on secondary GPU(s) used only for bulk PDF/ZIP."""
    global _bulk_warmed
    if _bulk_warmed:
        return
    with _bulk_warmup_lock:
        if _bulk_warmed:
            return
        from app.services.fast_ocr import warmup_gpus

        primary = get_primary_ocr_gpu_id()
        extra = [g for g in get_bulk_ocr_gpu_ids() if g != primary]
        if extra:
            warmup_gpus(extra)
        _bulk_warmed = True


def _warmup_zip_ocr_pool() -> None:
    """Pre-load RapidOCR on every bulk GPU before parallel ZIP workers start."""
    from app.services.fast_ocr import warmup_gpus

    if not settings.fast_use_cuda:
        return
    ids = get_bulk_ocr_gpu_ids() if settings.zip_pin_member_to_gpu else []
    if ids:
        logger.info("ZIP pre-warmup: RapidOCR on GPUs %s", ids)
        warmup_gpus(ids)
    global _bulk_warmed
    with _bulk_warmup_lock:
        _bulk_warmed = True


@dataclass
class _RegionPlan:
    region: DocumentRegion
    path: Path


@dataclass
class _PagePlan:
    page_idx: int
    image_path: Path
    # Original unenhanced page image (used for gap detection in resplit_bbox).
    # Enhancement fills white gaps between documents, breaking gap detection.
    orig_image_path: Path | None = None
    regions: list[_RegionPlan] = field(default_factory=list)
    need_full_page: bool = False
    # doc index → OCR keys for resplit sub-regions (garbled merge fix)
    resplit_keys: dict[int, list[str]] = field(default_factory=dict)
    # doc index → actual bboxes for each resplit sub-region (page coordinates)
    resplit_bboxes: dict[int, list[tuple[int, int, int, int]]] = field(default_factory=dict)


def _process_single_image(
    job_id: UUID | None,
    input_path: Path,
    *,
    source_name: str | None,
    update_job_progress: bool = True,
    allow_vision_escalation: bool | None = None,
    gpu_id: int | None = None,
) -> ProcessingResult:
    """Fast path: direct OCR on the upload — no segmentation / temp PNG."""
    primary_gpu = gpu_id if gpu_id is not None else get_primary_ocr_gpu_id()
    sem = ocr_gpu_semaphores([primary_gpu])[primary_gpu]
    t0 = time.time()

    # ── difficulty classification (<20 ms) ─────────────────────────────────────
    # Classify the document *before* OCR to pick the cheapest adequate pipeline.
    # EASY  → RapidOCR only, skip Tesseract / VLM / heavy segmentation.
    # MEDIUM → lightweight segmentation + optional Tesseract.
    # HARD  → full pipeline with VLM crop escalation and validators.
    _difficulty = None
    if settings.ocr_difficulty_enabled:
        try:
            from app.services.ocr_difficulty_analyzer import classify_image
            _difficulty = classify_image(input_path, fast=True)
            logger.debug(
                "DifficultyDecision %s: tier=%s conf=%.2f layout=%s pipeline=%s (%.1f ms)",
                input_path.name,
                _difficulty.tier.value,
                _difficulty.confidence,
                _difficulty.estimated_layout,
                _difficulty.recommended_pipeline,
                _difficulty.routing_time_ms,
            )
        except Exception as _exc:  # noqa: BLE001
            logger.debug("classify_image failed for %s: %s", input_path.name, _exc)

    # Adaptive noise thresholds follow document difficulty tier.
    if settings.ocr_noise_classifier_enabled:
        from app.services.ocr_noise_classifier import clear_noise_tier, set_noise_tier
        set_noise_tier(_difficulty.tier.value if _difficulty is not None else "medium")

    def _call() -> tuple[str, float | None, dict]:
        # Use difficulty-derived profile when available; fall back to IMAGE_PROFILE.
        profile = _difficulty.profile if _difficulty is not None else IMAGE_PROFILE
        routing = _difficulty.to_routing_decision() if _difficulty is not None else None
        with sem, use_profile(profile):
            return ocr_image(
                input_path,
                lang=effective_primary_ocr_lang(),
                allow_vision_escalation=allow_vision_escalation,
                gpu_id=primary_gpu,
                routing_decision=routing,
            )

    try:
        from app.services.ocr_quality_pipeline import use_quality_session

        _job_priority = _ocr_job_priority(
            _difficulty.tier.value if _difficulty is not None else None,
        )

        def _execute() -> tuple[str, float | None, dict]:
            if settings.ocr_oom_retry:
                return with_oom_retry(
                    _call,
                    gpu_id=primary_gpu,
                    max_retries=settings.ocr_oom_retry_max,
                    label=f"single_image[{(source_name or input_path.name)[:48]}]",
                )
            return _call()

        with use_quality_session() as _qsession:
            if settings.gpu_scheduler_enabled:
                try:
                    from app.services.dynamic_gpu_scheduler import TaskPriority
                    from app.services.gpu_scheduler import get_scheduler

                    sched = get_scheduler()
                    pri = (
                        TaskPriority(_job_priority)
                        if _job_priority is not None
                        else TaskPriority.NORMAL
                    )
                    text, confidence, meta = sched.submit_primary(
                        _execute, priority=pri,
                    ).result()
                    meta = {**meta, "scheduled": True, "priority": int(pri)}
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "GpuScheduler single-image submit failed, direct call: %s", exc,
                    )
                    text, confidence, meta = _execute()
            else:
                text, confidence, meta = _execute()

            if int(meta.get("crop_vlm_calls") or 0) > 0:
                _qsession.record_escalation(
                    from_engine=str(meta.get("engine", "fast_ocr")),
                    to_engine="minicpm",
                    reason=str(meta.get("escalation_reason") or "crop_vlm"),
                    lines_escalated=int(meta.get("crop_vlm_calls") or 0),
                )
                if "quality_session" in meta:
                    meta["quality_session"] = _qsession.to_dict()
    finally:
        if settings.ocr_noise_classifier_enabled:
            from app.services.ocr_noise_classifier import clear_noise_tier
            clear_noise_tier()

    elapsed = round(time.time() - t0, 3)
    engine = meta.get("engine", "fast_ocr")

    # ── benchmark metadata ─────────────────────────────────────────────────────
    diff_meta: dict = {}
    if _difficulty is not None:
        _difficulty.escalation_used = meta.get("crop_vlm_calls", 0) > 0
        diff_meta = {
            "difficulty_tier": _difficulty.tier.value,
            "routing_time_ms": round(_difficulty.routing_time_ms, 2),
            "selected_pipeline": _difficulty.recommended_pipeline,
            "estimated_layout": _difficulty.estimated_layout,
            "escalation_used": _difficulty.escalation_used,
        }

    quality_meta = {
        k: meta[k]
        for k in (
            "quality_score", "quality_tier", "quality_needs_recovery",
            "quality_actions", "quality_session", "quality",
        )
        if k in meta
    }

    warnings: list[str] = []
    if not text.strip():
        warnings.append("No text extracted from document")
    elif meta.get("quality_needs_recovery"):
        warnings.append(
            f"Low OCR quality (score={meta.get('quality_score')}, "
            f"tier={meta.get('quality_tier')})"
        )

    if update_job_progress and job_id is not None:
        job_store.update(
            job_id,
            progress=JobProgress(current=1, total=1, unit="pages", percent=100.0),
        )

    text = apply_polish_postprocess(text.strip())
    return ProcessingResult(
        source=source_name or input_path.name,
        source_kind=SourceKind.IMAGE,
        processed_at=datetime.now(UTC),
        page_count=1,
        language_detected=_languages_from_config(text),
        text=text,
        lines=parse_lines_from_meta(meta, default_page=1),
        confidence=confidence,
        engine=engine,
        warnings=warnings,
        metadata={
            "pipeline": "image_direct",
            "ocr_mode": "direct",
            "ocr_gpu_ids": [primary_gpu],
            "ocr_parallel": False,
            "ocr_batch_sec": elapsed,
            **diff_meta,
            **quality_meta,
            "pages": [{
                "page": 1,
                "documents_detected": 1,
                "ocr_mode": "direct",
                "engines": [engine],
                "elapsed_sec": elapsed,
            }],
            "documents": [{
                "page": 1,
                "document": 1,
                "text_chars": len(text),
                **meta,
            }],
            "documents_total": 1,
        },
    )


def _pdf_crop_line_level() -> bool:
    """Bulk PDF crops use RapidOCR unless explicitly enabled."""
    return settings.ocr_line_level_on_pdf_crops


def _ocr_job_priority(
    tier: str | None = None,
    *,
    retry: bool = False,
) -> int | None:
    """Map difficulty tier → GPU scheduler ``TaskPriority`` (stored on ``OcrJob``)."""
    if not settings.gpu_scheduler_enabled:
        return None
    from app.services.dynamic_gpu_scheduler import TaskPriority, priority_from_difficulty

    if retry:
        return int(TaskPriority.LOW)
    if tier is None:
        return int(TaskPriority.NORMAL)
    return int(priority_from_difficulty(tier))


def _classify_document_difficulty(input_path: Path, kind: SourceKind) -> str | None:
    """Return difficulty tier string (easy/medium/hard) for scheduler priority."""
    if not settings.ocr_difficulty_enabled:
        return None
    try:
        if kind == SourceKind.PDF:
            from app.services.ocr_difficulty_analyzer import classify_pdf_page

            dd = classify_pdf_page(input_path, page_index=0, fast=True)
        else:
            from app.services.ocr_difficulty_analyzer import classify_image

            dd = classify_image(input_path, fast=True)
        logger.debug(
            "Bulk difficulty %s: tier=%s (%.1f ms)",
            input_path.name,
            dd.tier.value,
            dd.routing_time_ms,
        )
        return dd.tier.value
    except Exception as exc:  # noqa: BLE001
        logger.debug("Document difficulty classification failed for %s: %s", input_path.name, exc)
        return None


def _retry_empty_regions(
    page_plans: list[_PagePlan],
    ocr_map: dict,
    work_dir: Path,
    *,
    vision_ok: bool | None,
    parallel: bool,
    page_parallel: bool,
    gpu_ids: list[int],
) -> None:
    """Re-OCR empty crops with contrast enhancement (faint column prints)."""
    extra_jobs: list[OcrJob] = []

    for plan in page_plans:
        for rp in plan.regions:
            key = f"p{plan.page_idx}:d{rp.region.index}"
            if key in plan.resplit_keys:
                continue
            res = ocr_map.get(key)
            if res is not None and res.text.strip():
                continue
            with Image.open(rp.path) as crop:
                crop.load()
                enhanced = prepare_crop_for_ocr(crop, force_strong=True)
            retry_path = work_dir / f"retry_{key.replace(':', '_')}.png"
            enhanced.save(retry_path, format="PNG")
            extra_jobs.append(OcrJob(
                key=f"{key}:retry",
                path=retry_path,
                lang=effective_primary_ocr_lang(),
                allow_vision_escalation=vision_ok,
                line_level=_pdf_crop_line_level(),
                priority=_ocr_job_priority(retry=True),
            ))

    if not extra_jobs:
        return
    extra_map = run_ocr_jobs(
        extra_jobs, parallel=parallel, page_parallel=page_parallel, gpu_ids=gpu_ids,
    )
    for plan in page_plans:
        for rp in plan.regions:
            key = f"p{plan.page_idx}:d{rp.region.index}"
            if ocr_map.get(key) and ocr_map[key].text.strip():
                continue
            retry = extra_map.get(f"{key}:retry")
            if retry and retry.text.strip():
                ocr_map[key] = retry
                logger.info("Page %d doc %d: recovered text after faint-crop retry",
                            plan.page_idx, rp.region.index)


def _retry_wrong_rotation(
    page_plans: list[_PagePlan],
    ocr_map: dict,
    work_dir: Path,
    *,
    vision_ok: bool | None,
    parallel: bool,
    page_parallel: bool,
    gpu_ids: list[int],
) -> None:
    """Re-OCR crops where 90°/270° rotation produced scrambled text."""
    extra_jobs: list[OcrJob] = []

    for plan in page_plans:
        with Image.open(plan.image_path) as page_pil:
            page_pil.load()
            for rp in plan.regions:
                key = f"p{plan.page_idx}:d{rp.region.index}"
                res = ocr_map.get(key)
                if res is None or rp.region.rotation not in (90, 270):
                    continue
                if not looks_garbled_ocr(res.text):
                    continue
                crop = page_pil.crop(rp.region.bbox)
                crop = prepare_crop_for_ocr(crop, force_strong=False)
                alt_path = work_dir / f"norot_{key.replace(':', '_')}.png"
                crop.save(alt_path, format="PNG")
                extra_jobs.append(OcrJob(
                    key=f"{key}:norot",
                    path=alt_path,
                    lang=effective_primary_ocr_lang(),
                    allow_vision_escalation=vision_ok,
                    line_level=_pdf_crop_line_level(),
                    priority=_ocr_job_priority(retry=True),
                ))

    if not extra_jobs:
        return
    extra_map = run_ocr_jobs(
        extra_jobs, parallel=parallel, page_parallel=page_parallel, gpu_ids=gpu_ids,
    )
    for plan in page_plans:
        for rp in plan.regions:
            key = f"p{plan.page_idx}:d{rp.region.index}"
            res = ocr_map.get(key)
            alt = extra_map.get(f"{key}:norot")
            if res is None or alt is None or not alt.text.strip():
                continue
            if len(alt.text.strip()) > len(res.text.strip()) * 1.1 or (
                looks_garbled_ocr(res.text) and not looks_garbled_ocr(alt.text)
            ):
                ocr_map[key] = alt
                logger.info(
                    "Page %d doc %d: replaced %d° OCR with upright (%d → %d chars)",
                    plan.page_idx, rp.region.index, rp.region.rotation,
                    len(res.text), len(alt.text),
                )


def _retry_inverted_180(
    page_plans: list[_PagePlan],
    ocr_map: dict,
    work_dir: Path,
    *,
    vision_ok: bool | None,
    parallel: bool,
    page_parallel: bool,
    gpu_ids: list[int],
) -> None:
    """Re-OCR crops where text is upside-down (detected 0° but actually 180°).

    Orientation detection based on ink-band scores cannot distinguish 0° from
    180° for receipts (symmetric text density).  After OCR we can detect the
    inverted reading order from the text itself and re-run OCR with the crop
    flipped 180°, which recovers both the correct line order AND the correct
    character orientation for barcodes / hash strings read upside-down.
    """
    extra_jobs: list[OcrJob] = []

    for plan in page_plans:
        with Image.open(plan.image_path) as page_pil:
            page_pil.load()
            for rp in plan.regions:
                key = f"p{plan.page_idx}:d{rp.region.index}"
                res = ocr_map.get(key)
                if res is None:
                    continue
                # Only act on crops where the text still looks upside-down:
                # – rotation==0: orientation detection missed the 180° case → flip now.
                # – rotation==180: detect_orientation applied 180° rotation but it was
                #   incorrect (e.g. ink-score falsely preferred 180°) → undo the flip
                #   by using the raw page crop without any rotation.
                if rp.region.rotation not in (0, 180):
                    continue
                if not looks_inverted_reading_order(res.text):
                    continue
                # Skip regions that contain MULTIPLE different documents (e.g. an
                # upright EMPIK receipt next to an upside-down AMIC invoice).
                # Flipping the whole crop would just swap which document is garbled;
                # let _handle_garbled_regions deal with those instead.
                if looks_like_merged_receipts(res.text) or looks_garbled_ocr(res.text):
                    continue
                crop = page_pil.crop(rp.region.bbox)
                if rp.region.rotation == 0:
                    # Orientation detection missed the 180° case — flip now.
                    crop = crop.rotate(180, expand=True)
                else:
                    # rotation==180 was incorrectly applied during segmentation
                    # (ink-band score false positive) — use the original unrotated crop.
                    pass  # crop already holds the raw page slice without any rotation
                desc = "inv180"
                crop = prepare_crop_for_ocr(crop, force_strong=False)
                alt_path = work_dir / f"inv180_{key.replace(':', '_')}.png"
                crop.save(alt_path, format="PNG")
                extra_jobs.append(OcrJob(
                    key=f"{key}:{desc}",
                    path=alt_path,
                    lang=effective_primary_ocr_lang(),
                    allow_vision_escalation=vision_ok,
                    line_level=_pdf_crop_line_level(),
                    priority=_ocr_job_priority(retry=True),
                ))
                logger.debug(
                    "Page %d doc %d: scheduling rotation-fix retry (rot=%d, inverted reading order)",
                    plan.page_idx, rp.region.index, rp.region.rotation,
                )

    if not extra_jobs:
        return
    extra_map = run_ocr_jobs(
        extra_jobs, parallel=parallel, page_parallel=page_parallel, gpu_ids=gpu_ids,
    )
    for plan in page_plans:
        for rp in plan.regions:
            key = f"p{plan.page_idx}:d{rp.region.index}"
            res = ocr_map.get(key)
            alt_key = f"{key}:inv180"
            alt = extra_map.get(alt_key)
            if res is None or alt is None or not alt.text.strip():
                continue
            # Accept the corrected version when:
            # (a) the new result is no longer flagged as inverted, AND
            # (b) it retains at least 80% of the original character count.
            alt_ok = not looks_inverted_reading_order(alt.text)
            has_enough = len(alt.text.strip()) >= len(res.text.strip()) * 0.80
            still_garbled = looks_garbled_ocr(alt.text)
            was_garbled = looks_garbled_ocr(res.text)
            if alt_ok and has_enough and (not still_garbled or was_garbled):
                ocr_map[key] = alt
                desc = "undone 180°" if rp.region.rotation == 180 else "180°-flip"
                logger.info(
                    "Page %d doc %d: replaced OCR via %s (%d → %d chars)",
                    plan.page_idx, rp.region.index, desc,
                    len(res.text), len(alt.text),
                )


def _retry_ksef_columns(
    page_plans: list[_PagePlan],
    ocr_map: dict,
    work_dir: Path,
    *,
    vision_ok: bool | None,
    parallel: bool,
    page_parallel: bool,
    gpu_ids: list[int],
) -> None:
    """Re-OCR KSeF pages as left/right columns when merged Sprzedawca|Nabywca."""
    for plan in page_plans:
        full_key = f"p{plan.page_idx}:full"
        full_res = ocr_map.get(full_key)
        need_retry = False
        if full_res and looks_like_ksef_merged_text(full_res.text):
            need_retry = True
        elif len(plan.regions) == 1:
            res = ocr_map.get(f"p{plan.page_idx}:d{plan.regions[0].region.index}")
            if res and looks_like_ksef_merged_text(res.text):
                need_retry = True
        else:
            parts: list[str] = []
            for rp in plan.regions:
                res = ocr_map.get(f"p{plan.page_idx}:d{rp.region.index}")
                if res and res.text.strip():
                    parts.append(res.text)
            if parts and looks_like_ksef_merged_text("\n".join(parts)):
                need_retry = True
        if not need_retry:
            continue

        with Image.open(plan.image_path) as page_pil:
            page_pil.load()
            ksef_regions = extract_ksef_column_regions(page_pil, ocr_page_image=page_pil)
        if not ksef_regions:
            continue

        extra_jobs: list[OcrJob] = []
        new_rps: list[_RegionPlan] = []
        for kr in ksef_regions:
            path = work_dir / f"ksef_p{plan.page_idx:04d}_c{kr.index:02d}.png"
            kr.image.save(path, format="PNG")
            key = f"p{plan.page_idx}:d{kr.index}"
            extra_jobs.append(OcrJob(
                key=key,
                path=path,
                lang=effective_primary_ocr_lang(),
                allow_vision_escalation=vision_ok,
                line_level=_pdf_crop_line_level(),
                priority=_ocr_job_priority(retry=True),
            ))
            new_rps.append(_RegionPlan(region=kr, path=path))

        if not extra_jobs:
            continue
        extra_map = run_ocr_jobs(
            extra_jobs, parallel=parallel, page_parallel=page_parallel, gpu_ids=gpu_ids,
        )
        plan.regions = new_rps
        plan.resplit_keys.clear()
        for kr in ksef_regions:
            key = f"p{plan.page_idx}:d{kr.index}"
            res = extra_map.get(key)
            if res and res.text.strip():
                ocr_map[key] = res
        for k in list(ocr_map):
            if k.startswith(f"p{plan.page_idx}:") and k != full_key:
                if k not in {f"p{plan.page_idx}:d{kr.index}" for kr in ksef_regions}:
                    ocr_map.pop(k, None)
        # Keep full-page result under a backup key so _ksef_smart_join can
        # extract the header (which appears before the interleaved columns).
        if full_key in ocr_map:
            ocr_map[f"{full_key}_backup"] = ocr_map[full_key]
        ocr_map.pop(full_key, None)
        logger.info(
            "Page %d: KSeF column re-OCR (%d columns)",
            plan.page_idx, len(ksef_regions),
        )


def _handle_garbled_regions(
    page_plans: list[_PagePlan],
    ocr_map: dict,
    work_dir: Path,
    *,
    vision_ok: bool | None,
    parallel: bool,
    page_parallel: bool,
    gpu_ids: list[int],
) -> None:
    """Re-segment and re-OCR regions that look like merged side-by-side receipts."""
    extra_jobs: list[OcrJob] = []

    for plan in page_plans:
        with Image.open(plan.image_path) as page_pil:
            page_pil.load()
        # Use the original unenhanced page for gap detection in resplit_bbox.
        # Scan enhancement fills white gaps between stacked documents, which
        # breaks text-mask gap detection and causes resplit_bbox to return only 1 box.
        if plan.orig_image_path is not None and plan.orig_image_path.exists():
            with Image.open(plan.orig_image_path) as orig_pil:
                orig_pil.load()
            resplit_img = orig_pil
        else:
            resplit_img = page_pil

        for rp in plan.regions:
            key = f"p{plan.page_idx}:d{rp.region.index}"
            res = ocr_map.get(key)
            merged_check = looks_like_merged_receipts(res.text) if res is not None else False
            if res is None or not merged_check:
                continue
            # Already rotated upright — garbled text is usually merged columns,
            # not fixable by splitting; splitting creates more fragments.
            sub_boxes = resplit_bbox(resplit_img, rp.region.bbox)
            if len(sub_boxes) < 2:
                continue
            sub_keys: list[str] = []
            for sub_idx, sub_bb in enumerate(sub_boxes, start=1):
                crop = page_pil.crop(sub_bb)
                rotation, _ = detect_orientation(
                    crop, min_confidence=settings.pdf_osd_min_confidence,
                )
                # Resplit sub-regions come from horizontal (top/bottom) splits so
                # they should only need 0° or 180° correction.  A spurious 90°/270°
                # result from the ink-score heuristic (common for near-landscape
                # crops like an upside-down invoice) would destroy the layout.
                # When 90°/270° is returned, explicitly probe 0° vs 180° using
                # the reading-order heuristic (FAKTURA/PARAGON in top half).
                if rotation in (90, 270):
                    rotation = _prefer_rotation_by_reading_order(crop)
                    if rotation not in (0, 180):
                        rotation = 0
                if rotation:
                    crop = crop.rotate(rotation, expand=True)
                crop = prepare_crop_for_ocr(crop, force_strong=True)
                sub_path = work_dir / (
                    f"resplit_p{plan.page_idx:04d}_d{rp.region.index:02d}_s{sub_idx:02d}.png"
                )
                crop.save(sub_path, format="PNG")
                sub_key = f"{key}:s{sub_idx}"
                sub_keys.append(sub_key)
                extra_jobs.append(OcrJob(
                    key=sub_key,
                    path=sub_path,
                    lang=effective_primary_ocr_lang(),
                    allow_vision_escalation=vision_ok,
                    line_level=_pdf_crop_line_level(),
                    priority=_ocr_job_priority(retry=True),
                ))
            if sub_keys:
                plan.resplit_keys[rp.region.index] = sub_keys
                plan.resplit_bboxes[rp.region.index] = list(sub_boxes)
                logger.info(
                    "Page %d doc %d: resplit into %d sub-regions after garbled OCR",
                    plan.page_idx, rp.region.index, len(sub_keys),
                )

    if not extra_jobs:
        return
    extra_map = run_ocr_jobs(
        extra_jobs, parallel=parallel, page_parallel=page_parallel, gpu_ids=gpu_ids,
    )
    ocr_map.update(extra_map)


def _assemble_page_results(
    plan: _PagePlan,
    ocr_map: dict,
    *,
    use_fullpage_fallback: bool,
    confidences: list[float],
    warnings: list[str],
    engines_used: set[str],
) -> tuple[list[str], list[dict], dict]:
    """Turn OCR map entries into document texts + metadata for one page."""
    engines_on_page: set[str] = set()
    page_doc_texts: list[str] = []
    page_doc_metas: list[dict] = []

    full_res = (
        ocr_map.get(f"p{plan.page_idx}:full")
        or ocr_map.get(f"p{plan.page_idx}:full_backup")
    )
    full_page_text = full_res.text if full_res else ""
    full_page_meta = full_res.meta if full_res else {}
    if full_res and full_res.confidence is not None:
        confidences.append(full_res.confidence)

    for rp in plan.regions:
        sub_keys = plan.resplit_keys.get(rp.region.index)
        sub_bboxes = plan.resplit_bboxes.get(rp.region.index) or []
        merge_parts = (
            sub_keys
            and settings.pdf_merge_resplit_parts
        )
        keys_to_use = sub_keys if sub_keys else [f"p{plan.page_idx}:d{rp.region.index}"]

        part_texts: list[str] = []
        part_metas: list[dict] = []
        for part_i, ocr_key in enumerate(keys_to_use, start=1):
            res = ocr_map.get(ocr_key)
            if res is None:
                warnings.append(
                    f"Missing OCR result page {plan.page_idx} "
                    f"doc {rp.region.index} part {part_i}"
                )
                continue
            engines_used.add(res.meta.get("engine", "tesseract"))
            engines_on_page.add(res.meta.get("engine", "tesseract"))
            if res.confidence is not None:
                confidences.append(res.confidence)
            if not res.text.strip():
                warnings.append(
                    f"Empty OCR on page {plan.page_idx}, document {rp.region.index}"
                )
            # For sub-parts, use the actual sub-region bbox so document merging
            # heuristics don't re-merge EMPIK and AMIC (which share the parent bbox).
            part_bbox = (
                list(sub_bboxes[part_i - 1])
                if sub_bboxes and part_i - 1 < len(sub_bboxes)
                else list(rp.region.bbox)
            )
            part_metas.append({
                "page": plan.page_idx,
                "document": rp.region.index,
                "bbox": part_bbox,
                "rotation": rp.region.rotation,
                "osd_confidence": round(rp.region.osd_confidence, 2),
                "layout_hint": rp.region.layout_hint or None,
                "text_chars": len(res.text),
                **res.meta,
            })
            # For sub-parts produced by resplit (each is an individual document),
            # always check inverted reading order with rotation=0 because
            # _handle_garbled_regions already applied orientation correction
            # to the crop before OCR. For non-split regions that look like merged
            # receipts, skip reversal (would corrupt the multi-doc block).
            if sub_keys:
                part_texts.append(
                    fix_inverted_reading_order(res.text, 0),
                )
            elif looks_like_merged_receipts(res.text):
                part_texts.append(res.text)
            else:
                part_texts.append(
                    fix_inverted_reading_order(res.text, rp.region.rotation),
                )

        if not part_texts:
            continue

        if merge_parts and len(part_texts) > 1:
            combined = "\n".join(t.strip() for t in part_texts if t.strip())
            meta_entry = {
                **part_metas[-1],
                "text_chars": len(combined),
                "resplit_merged": True,
                "resplit_parts": len(part_texts),
            }
            page_doc_metas.append(meta_entry)
            page_doc_texts.append(combined)
        else:
            for i, text in enumerate(part_texts):
                entry = part_metas[i]
                if sub_keys and len(part_texts) > 1:
                    entry = {
                        **entry,
                        "resplit_part": i + 1,
                        "resplit_total": len(part_texts),
                        # Prevent downstream merge functions from re-joining
                        # what resplit_bbox explicitly separated (e.g. EMPIK + AMIC).
                        "_no_auto_merge": True,
                    }
                page_doc_metas.append(entry)
                page_doc_texts.append(text)

    if (
        len(page_doc_texts) == 2
        and len(plan.regions) == 2
        and all(rp.region.layout_hint == "ksef_column" for rp in plan.regions)
    ):
        # Validate the ``ksef_column`` segmentation hint against the
        # full-page OCR text.  The detector can return ``ksef_column``
        # for any page that visually splits into two big blocks (letters,
        # postal notices, scans with a side stamp).  In that case the
        # "left + --- + right" join produces wrong reading order; we
        # must fall back to the natural full-page Y/X order instead.
        #
        # KSeF check: use both full-page text AND combined column text so
        # we catch cases where the full-page OCR missed the "KSeF" footer
        # but the column OCR captured it correctly.
        _col_combined_text = "\n".join(t.strip() for t in page_doc_texts)
        _ksef_text_for_check = full_page_text if full_page_text.strip() else _col_combined_text
        if _looks_like_real_ksef_invoice(_ksef_text_for_check) or _looks_like_real_ksef_invoice(_col_combined_text):
            # For genuine KSeF invoices prefer full-page OCR over column join.
            # Column-join cuts full-width headers ("Faktura" → "Fa"), reverses
            # table headers, and gives wrong top→bottom reading order.
            # Full-page OCR reads row-by-row and preserves natural document flow.
            _fp_chars = len(full_page_text.strip())
            _col_chars = sum(len(t.strip()) for t in page_doc_texts)
            _fp_garbled = looks_like_ksef_merged_text(full_page_text) if _fp_chars else True
            if not _fp_garbled and _fp_chars >= max(100, _col_chars * 0.65):
                # Clean full-page OCR available — use it for natural reading order.
                page_doc_texts = [full_page_text]
                page_doc_metas = [{
                    **page_doc_metas[0],
                    "layout": "ksef_full_page",
                    "ksef_columns": 2,
                    "text_chars": _fp_chars,
                }]
                logger.info(
                    "Page %d: KSeF invoice → full-page OCR (%d chars, col-join %d chars)",
                    plan.page_idx, _fp_chars, _col_chars,
                )
            else:
                # Full-page interleaved/garbled — use smart join:
                # take clean header from full-page OCR + seller/buyer bodies
                # from column crops to avoid cutting full-width headings.
                combined = _ksef_smart_join(
                    page_doc_texts[0], page_doc_texts[1], full_page_text,
                )
                page_doc_texts = [combined]
                page_doc_metas = [{
                    **page_doc_metas[0],
                    "layout": "ksef_smart_join",
                    "ksef_columns": 2,
                    "text_chars": len(combined),
                }]
                logger.info(
                    "Page %d: KSeF invoice → smart join "
                    "(full-page garbled=%s, fp=%d, col=%d chars)",
                    plan.page_idx, _fp_garbled, _fp_chars, _col_chars,
                )
        elif _looks_like_two_column_invoice_pair(page_doc_texts[0], page_doc_texts[1]):
            combined = _ksef_smart_join(
                page_doc_texts[0], page_doc_texts[1], full_page_text,
            )
            page_doc_texts = [combined]
            page_doc_metas = [{
                **page_doc_metas[0],
                "layout": "two_column_invoice",
                "ksef_columns": 2,
                "text_chars": len(combined),
            }]
            warnings.append(
                f"Page {plan.page_idx}: merged two invoice-like columns into one document"
            )
        else:
            # Columns are NOT a KSeF invoice.  Check whether they form ONE
            # continuous document (e.g. a government form, letter, or report
            # physically printed in two columns) or genuinely separate side-by-side
            # documents (receipts, individual invoices, etc.).
            _has_receipt = any("PARAGON FISKALNY" in t.upper() for t in page_doc_texts)
            _col_kinds = [classify_document(t) for t in page_doc_texts]
            _both_nondoc = all(
                k in (DocumentKind.OTHER, DocumentKind.LIST) for k in _col_kinds
            )
            _fp_chars_col = len(full_page_text.strip())
            _col_chars_total = sum(len(t.strip()) for t in page_doc_texts)
            _fp_ok = (
                _fp_chars_col >= max(100, _col_chars_total * 0.4)
                and not looks_garbled_ocr(full_page_text)
            )

            if not _has_receipt and _both_nondoc:
                # Single-document two-column page (government form, letter, etc.).
                # Use full-page OCR when available (reads across the full width so
                # split words like "DOKUM|ENT" are reassembled) or directly merge
                # the two column OCR texts.
                if full_page_text.strip() and _fp_ok:
                    page_doc_texts = [full_page_text]
                    page_doc_metas = [{
                        **page_doc_metas[0],
                        "layout": "singledoc_full_page",
                        "ksef_columns": 2,
                        "text_chars": _fp_chars_col,
                    }]
                    logger.info(
                        "Page %d: two-column single-document → full-page OCR (%d chars, "
                        "col total %d chars)",
                        plan.page_idx, _fp_chars_col, _col_chars_total,
                    )
                else:
                    # No clean full-page text — concatenate columns in reading order
                    # (left column first, then right column).
                    combined_col = "\n".join(
                        t.strip() for t in page_doc_texts if t.strip()
                    )
                    page_doc_texts = [combined_col]
                    page_doc_metas = [{
                        **page_doc_metas[0],
                        "layout": "merged_columns",
                        "ksef_columns": 2,
                        "text_chars": len(combined_col),
                    }]
                    logger.info(
                        "Page %d: two-column single-document → columns merged directly (%d chars)",
                        plan.page_idx, len(combined_col),
                    )
                warnings.append(
                    f"Page {plan.page_idx}: two-column non-receipt single document merged"
                )
            else:
                # Columns contain receipts or separate invoice documents — keep each
                # column as its own document so they aren't merged into one blob.
                logger.info(
                    "Page %d: ksef_column hint rejected (not KSeF invoice); "
                    "treating %d columns as separate documents",
                    plan.page_idx, len(page_doc_texts),
                )
                warnings.append(
                    f"Page {plan.page_idx}: ksef_column split retained as separate documents"
                )
                # Try to split each column text further: a column may contain
                # multiple receipts stacked vertically (e.g. 2 paragony per column).
                split_texts: list[str] = []
                split_metas: list[dict] = []
                for col_text, col_meta in zip(page_doc_texts, page_doc_metas):
                    sub_parts = split_multi_document_block(col_text)
                    if len(sub_parts) > 1:
                        logger.debug(
                            "Page %d: column split into %d sub-documents by header markers",
                            plan.page_idx, len(sub_parts),
                        )
                        for pi, pt in enumerate(sub_parts, 1):
                            m = dict(col_meta)
                            m["split_part"] = pi
                            m["split_of"] = len(sub_parts)
                            m["text_chars"] = len(pt)
                            # Prevent the merge functions from re-joining what we
                            # just explicitly split by header markers.
                            m["_no_auto_merge"] = True
                            split_texts.append(pt)
                            split_metas.append(m)
                    else:
                        m = dict(col_meta)
                        # Each column is a separate document; mark accordingly.
                        m["_no_auto_merge"] = True
                        split_texts.append(col_text)
                        split_metas.append(m)
                page_doc_texts = split_texts
                page_doc_metas = split_metas

    seg_chars = sum(len(t.strip()) for t in page_doc_texts)
    full_chars = len(full_page_text.strip())
    use_segmented = True
    n_regions = len(plan.regions)
    garbled_segmented = any(looks_garbled_ocr(t) for t in page_doc_texts)
    full_merged = looks_like_merged_receipts(full_page_text) if full_chars else False
    full_ksef_merged = looks_like_ksef_merged_text(full_page_text) if full_chars else False

    if use_fullpage_fallback and full_chars > 0:
        ratio = seg_chars / full_chars if full_chars else 0.0
        allow_fullpage = len(page_doc_texts) == 0
        if not allow_fullpage:
            if ratio < settings.pdf_fullpage_fallback_ratio:
                if n_regions <= settings.pdf_fullpage_fallback_max_regions:
                    allow_fullpage = True
                elif (
                    ratio < settings.pdf_fullpage_fallback_aggressive_ratio
                    or seg_chars < settings.pdf_fullpage_fallback_min_segmented_chars
                ):
                    # Tiny fragment — prefer full page even if side-by-side mix.
                    if not full_merged or ratio < 0.3 or seg_chars < 350:
                        allow_fullpage = True
            if garbled_segmented and not full_merged and full_chars >= seg_chars * 0.9:
                allow_fullpage = True
            if full_ksef_merged and any(
                rp.region.layout_hint == "ksef_column" for rp in plan.regions
            ):
                allow_fullpage = False
            # Letter / single-doc page: prefer full-page reading order
            # when the page is clearly a continuous letter / notice and
            # segmentation didn't dramatically increase content.
            if (
                not allow_fullpage
                and _looks_like_letter_or_singledoc(full_page_text)
                and not full_merged
                and not _looks_like_real_ksef_invoice(full_page_text)
                and ratio >= 0.6
                and full_chars >= seg_chars * 0.85
            ):
                allow_fullpage = True
        if allow_fullpage:
            use_segmented = False
            logger.info(
                "Page %d: full-page fallback (segmented %d chars, full %d chars, "
                "ratio %.2f, regions=%d, garbled=%s)",
                plan.page_idx, seg_chars, full_chars, ratio, n_regions, garbled_segmented,
            )
            warnings.append(
                f"Page {plan.page_idx}: used full-page OCR "
                f"(segmented {seg_chars} vs full {full_chars} chars)"
            )
        elif ratio < settings.pdf_fullpage_fallback_ratio and n_regions > 1:
            warnings.append(
                f"Page {plan.page_idx}: kept segmented OCR despite low char ratio "
                f"({seg_chars} vs full {full_chars}) to avoid mixing side-by-side receipts"
            )

    doc_texts: list[str] = []
    doc_metas: list[dict] = []
    if use_segmented:
        doc_metas = page_doc_metas
        doc_texts = page_doc_texts
    else:
        doc_texts = [full_page_text]
        doc_metas = [{
            "page": plan.page_idx,
            "document": 1,
            "mode": "full_page_fallback",
            "text_chars": full_chars,
            "segmented_chars": seg_chars,
            "regions_attempted": len(plan.regions),
            **full_page_meta,
        }]
        if full_page_meta:
            engines_used.add(full_page_meta.get("engine", "fast_ocr"))
            engines_on_page.add(full_page_meta.get("engine", "fast_ocr"))

    if use_segmented and len(doc_texts) > 1:
        doc_texts, doc_metas = _merge_incomplete_fragments(doc_texts, doc_metas)
        doc_texts, doc_metas = _merge_receipt_continuations(doc_texts, doc_metas)
    if (
        use_segmented
        and len(doc_texts) > 1
        and any(rp.region.layout_hint == "ksef_column" for rp in plan.regions)
    ):
        joined_upper = "\n".join(doc_texts).upper()
        if "PARAGON FISKALNY" not in joined_upper and "PARIAGON FISKALNY" not in joined_upper:
            combined = "\n\n---\n\n".join(t.strip() for t in doc_texts if t.strip())
            meta = dict(doc_metas[0]) if doc_metas else {"page": plan.page_idx}
            meta["layout"] = "column_page_merged"
            meta["text_chars"] = len(combined)
            doc_texts = [combined]
            doc_metas = [meta]
    if use_segmented and len(doc_texts) > 1:
        # Invoice-heavy pages should stay as one page-level document.
        # Do NOT apply this to receipts (paragony), where multiple docs/page
        # are expected and must remain separated.
        combined_upper = "\n".join(doc_texts).upper()
        invoice_like_page = (
            "PARAGON FISKALNY" not in combined_upper
            and
            (
                "FAKTURA" in combined_upper
                or ("SPRZEDAWCA" in combined_upper and "NABYWCA" in combined_upper)
                or "DO ZAPLATY" in combined_upper
                or "DO ZAPŁATY" in combined_upper
            )
        )
        if invoice_like_page:
            combined = "\n\n---\n\n".join(t.strip() for t in doc_texts if t.strip())
            meta = dict(doc_metas[0]) if doc_metas else {"page": plan.page_idx}
            meta["layout"] = "invoice_two_column_merged"
            meta["text_chars"] = len(combined)
            doc_texts = [combined]
            doc_metas = [meta]

    page_meta = {
        "page": plan.page_idx,
        "documents_detected": len(plan.regions) if use_segmented else 1,
        "ocr_mode": "segmented" if use_segmented else "full_page_fallback",
        "segmented_chars": seg_chars,
        "full_page_chars": full_chars if use_fullpage_fallback else None,
        "output_documents": len(doc_texts),
        "engines": sorted(engines_on_page) or ["tesseract"],
    }
    return doc_texts, doc_metas, page_meta


def _member_result_dir(result_dir: Path, member: ZipMember) -> Path:
    return result_dir.joinpath(*Path(member.rel_path).parts)


def _member_work_dir(work_dir: Path, member: ZipMember) -> Path:
    return work_dir / "members" / member.rel_path.replace("/", "__")


def _process_bulk_document(
    input_path: Path,
    work_dir: Path,
    *,
    source_name: str | None = None,
    job_id: UUID | None = None,
    pinned_gpu: int | None = None,
) -> ProcessingResult:
    """PDF (or single-page image treated as bulk) → ProcessingResult.

    ``pinned_gpu`` (used by the ZIP pipeline) forces every OCR job for this
    document onto a single GPU and disables internal multi-GPU parallelism.
    The ZIP queue still runs multiple members in parallel — one per GPU —
    so total throughput is preserved while VRAM contention drops.
    """
    kind = _source_kind(input_path)
    work_dir.mkdir(parents=True, exist_ok=True)

    _doc_difficulty_tier = _classify_document_difficulty(input_path, kind)
    _doc_priority = _ocr_job_priority(_doc_difficulty_tier)

    page_images, dpi_routing_meta = _collect_page_images(input_path, work_dir)
    total_pages = len(page_images)
    file_bytes = input_path.stat().st_size
    confidences: list[float] = []
    warnings: list[str] = []
    page_metadata: list[dict] = []
    document_texts: list[str] = []
    document_metadata: list[dict] = []
    engines_used: set[str] = set()

    segment_enabled = settings.pdf_segment_documents and kind == SourceKind.PDF
    pdf_escalate = settings.pdf_smart_escalate
    use_fullpage_fallback = kind == SourceKind.PDF
    vision_ok = pdf_escalate if kind == SourceKind.PDF else None

    page_plans: list[_PagePlan] = []
    enh_pages_dir = work_dir / "pages_enhanced"
    enh_pages_dir.mkdir(parents=True, exist_ok=True)

    orig_pages_dir = work_dir / "pages_orig"
    orig_pages_dir.mkdir(parents=True, exist_ok=True)

    for page_idx, image_path in enumerate(page_images, start=1):
        with Image.open(image_path) as page_pil:
            page_pil.load()
            page_for_ocr = prepare_page_for_ocr(page_pil)
        enh_path = enh_pages_dir / f"page_{page_idx:04d}.png"
        page_for_ocr.save(enh_path, format="PNG")
        orig_path = orig_pages_dir / f"page_{page_idx:04d}.png"
        page_pil.save(orig_path, format="PNG")

        regions = extract_documents(
            page_pil,
            ocr_page_image=page_for_ocr,
            enable_segmentation=segment_enabled,
            enable_auto_rotate=settings.pdf_auto_rotate,
            min_area_pct=settings.pdf_min_region_area_pct,
            min_side_pct=settings.pdf_min_region_side_pct,
            max_regions=settings.pdf_max_regions_per_page,
            osd_min_confidence=settings.pdf_osd_min_confidence,
        )
        # Some multi-receipt scanner pages are under-segmented with strict
        # defaults (0-1 region), which later forces full-page fallback and
        # merges several receipts into one block. Retry once with relaxed
        # thresholds and keep it only if it increases the region count.
        if segment_enabled and len(regions) <= 1:
            relaxed_regions = extract_documents(
                page_pil,
                ocr_page_image=page_for_ocr,
                enable_segmentation=True,
                enable_auto_rotate=settings.pdf_auto_rotate,
                min_area_pct=max(0.0015, settings.pdf_min_region_area_pct * 0.35),
                min_side_pct=max(0.015, settings.pdf_min_region_side_pct * 0.5),
                max_regions=max(settings.pdf_max_regions_per_page * 2, 24),
                osd_min_confidence=settings.pdf_osd_min_confidence,
            )
            if len(relaxed_regions) > len(regions):
                logger.info(
                    "Page %d: segmentation relaxed %d -> %d regions",
                    page_idx,
                    len(regions),
                    len(relaxed_regions),
                )
                regions = relaxed_regions
        has_ksef_columns = any(r.layout_hint == "ksef_column" for r in regions)
        plan = _PagePlan(
            page_idx=page_idx,
            image_path=enh_path,
            orig_image_path=orig_path,
            # Always OCR the full page for ksef layouts so we can prefer it
            # over the column join when the full-page reading order is clean.
            need_full_page=use_fullpage_fallback or has_ksef_columns,
        )
        for region in regions:
            doc_path = _persist_region_image(
                region.image, work_dir, page_idx, region.index,
            )
            plan.regions.append(_RegionPlan(region=region, path=doc_path))
        page_plans.append(plan)

    pdf_profile_name = (
        PDF_LINE_PROFILE.name if _pdf_crop_line_level() else PDF_CROP_PROFILE.name
    )
    ocr_jobs: list[OcrJob] = []
    for plan in page_plans:
        if plan.need_full_page:
            ocr_jobs.append(OcrJob(
                key=f"p{plan.page_idx}:full",
                path=plan.image_path,
                lang=effective_primary_ocr_lang(),
                allow_vision_escalation=vision_ok,
                line_level=_pdf_crop_line_level(),
                profile_name=pdf_profile_name,
                priority=_doc_priority,
            ))
        for rp in plan.regions:
            ocr_jobs.append(OcrJob(
                key=f"p{plan.page_idx}:d{rp.region.index}",
                path=rp.path,
                lang=effective_primary_ocr_lang(),
                allow_vision_escalation=vision_ok,
                line_level=_pdf_crop_line_level(),
                profile_name=pdf_profile_name,
                priority=_doc_priority,
            ))

    if pinned_gpu is not None:
        # ZIP context: keep this whole document on one GPU. The ZIP queue
        # spreads members across GPUs so the pool is still saturated.
        use_page_parallel = False
        use_parallel = False
        bulk_gpus = [pinned_gpu]
        multi_gpu = False
    else:
        use_page_parallel = should_use_page_parallel(total_pages)
        use_parallel = should_use_parallel_bulk(
            kind,
            page_count=total_pages,
            ocr_job_count=len(ocr_jobs),
            file_bytes=file_bytes,
        )
        multi_gpu = use_page_parallel or use_parallel
        bulk_gpus = get_bulk_ocr_gpu_ids() if multi_gpu else [get_primary_ocr_gpu_id()]
    # Warm up any secondary GPU we are about to use (pinned or multi-GPU)
    # so the first OCR call on that card doesn't pay the engine-build cost
    # under a held semaphore.
    if multi_gpu or (pinned_gpu is not None and pinned_gpu != get_primary_ocr_gpu_id()):
        _ensure_bulk_gpu_warmup()

    job_t0 = time.time()
    ocr_map = run_ocr_jobs(
        ocr_jobs,
        parallel=use_parallel,
        page_parallel=use_page_parallel,
        gpu_ids=bulk_gpus,
    )
    ocr_wall = time.time() - job_t0
    logger.info(
        "OCR batch [%s]: %d jobs in %.2fs page_parallel=%s region_parallel=%s GPUs %s",
        source_name or input_path.name,
        len(ocr_jobs), ocr_wall, use_page_parallel, use_parallel, bulk_gpus,
    )

    if kind == SourceKind.PDF:
        _retry_empty_regions(
            page_plans, ocr_map, work_dir,
            vision_ok=vision_ok,
            parallel=use_parallel,
            page_parallel=use_page_parallel,
            gpu_ids=bulk_gpus,
        )
        _retry_wrong_rotation(
            page_plans, ocr_map, work_dir,
            vision_ok=vision_ok,
            parallel=use_parallel,
            page_parallel=use_page_parallel,
            gpu_ids=bulk_gpus,
        )
        _retry_inverted_180(
            page_plans, ocr_map, work_dir,
            vision_ok=vision_ok,
            parallel=use_parallel,
            page_parallel=use_page_parallel,
            gpu_ids=bulk_gpus,
        )
        _retry_ksef_columns(
            page_plans, ocr_map, work_dir,
            vision_ok=vision_ok,
            parallel=use_parallel,
            page_parallel=use_page_parallel,
            gpu_ids=bulk_gpus,
        )
        if settings.pdf_resplit_garbled:
            _handle_garbled_regions(
                page_plans, ocr_map, work_dir,
                vision_ok=vision_ok,
                parallel=use_parallel,
                page_parallel=use_page_parallel,
                gpu_ids=bulk_gpus,
            )

    for plan in page_plans:
        page_t0 = time.time()
        doc_texts, doc_metas, page_meta = _assemble_page_results(
            plan, ocr_map,
            use_fullpage_fallback=use_fullpage_fallback,
            confidences=confidences,
            warnings=warnings,
            engines_used=engines_used,
        )
        doc_texts, doc_metas = _normalize_page_document_blocks(
            doc_texts, doc_metas, page_idx=plan.page_idx,
        )
        page_meta["output_documents"] = len(doc_texts)
        document_metadata.extend(doc_metas)
        document_texts.extend(doc_texts)
        page_meta["elapsed_sec"] = round(time.time() - page_t0, 2)
        page_metadata.append(page_meta)

        if job_id is not None:
            job_store.update(
                job_id,
                progress=JobProgress(
                    current=plan.page_idx,
                    total=total_pages,
                    unit="pages",
                    percent=round(100.0 * plan.page_idx / total_pages, 1),
                ),
            )

        merged_text = _merge_documents(document_texts, document_metadata)
    if not merged_text.strip():
        warnings.append("No text extracted from document")

    mean_confidence = (
        sum(confidences) / len(confidences) if confidences else None
    )
    engine_label = "+".join(sorted(engines_used)) if engines_used else "tesseract"

    structured_lines = aggregate_lines_from_metas(
        document_metadata, page_key="page",
    )

    if dpi_routing_meta:
        from app.services.pdf_dpi_routing import log_dpi_routing_metrics

        log_dpi_routing_metrics(label=source_name or input_path.name)

    return ProcessingResult(
        source=source_name or input_path.name,
        source_kind=kind,
        processed_at=datetime.now(UTC),
        page_count=total_pages,
        language_detected=_languages_from_config(merged_text),
        text=merged_text,
        lines=structured_lines,
        confidence=round(mean_confidence, 4) if mean_confidence is not None else None,
        engine=engine_label,
        warnings=warnings,
        metadata={
            "pipeline": "bulk",
            "pages": page_metadata,
            "documents": document_metadata,
            "documents_total": len(document_metadata),
            "ocr_gpu_ids": bulk_gpus,
            "ocr_page_parallel": use_page_parallel,
            "ocr_parallel": use_parallel,
            "ocr_batch_sec": round(ocr_wall, 2),
            "ocr_pinned_gpu": pinned_gpu,
            **({"dpi_routing": dpi_routing_meta} if dpi_routing_meta else {}),
        },
    )


def _process_zip_member(
    job_id: UUID,
    member: ZipMember,
    work_dir: Path,
    result_dir: Path,
    *,
    allow_vision_escalation: bool | None = None,
    pinned_gpu: int | None = None,
) -> tuple[Path, Path]:
    member_work = _member_work_dir(work_dir, member)
    with use_profile(ZIP_MEMBER_PROFILE):
        if member.kind == SourceKind.IMAGE:
            gpu_id = (
                pinned_gpu
                if pinned_gpu is not None
                else allocate_gpu(get_bulk_ocr_gpu_ids())
            )
            result = _process_single_image(
                job_id,
                member.path,
                source_name=member.rel_path,
                update_job_progress=False,
                allow_vision_escalation=allow_vision_escalation,
                gpu_id=gpu_id,
            )
        elif should_use_zip_simple_receipt(member.path, member_work):
            result = _process_zip_simple_receipt_pdf(
                member.path,
                member_work,
                source_name=member.rel_path,
                allow_vision_escalation=allow_vision_escalation,
                pinned_gpu=pinned_gpu,
            )
        else:
            result = _process_bulk_document(
                member.path,
                member_work,
                source_name=member.rel_path,
                pinned_gpu=pinned_gpu,
            )
    out_dir = _member_result_dir(result_dir, member)
    return write_results(result, out_dir)


@dataclass
class _MemberOutcome:
    """Result of one ZIP member — success carries paths, failure carries error."""

    ok: bool
    paths: tuple[Path, Path] | None = None
    error: str | None = None
    error_type: str | None = None


def _persist_zip_errors(
    result_dir: Path,
    failures: list[dict],
    total_files: int,
    succeeded: int,
) -> Path:
    """Write ``errors.json`` next to per-member result folders for ZIP jobs."""
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_files": total_files,
        "succeeded": succeeded,
        "failed": len(failures),
        "failures": failures,
    }
    path = result_dir / "errors.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def find_job_zip_upload(job_id: UUID) -> Path:
    """Locate the original ZIP upload for a job (``uploads/{job_id}_*.zip``)."""
    matches = sorted(settings.uploads_dir.glob(f"{job_id}_*"))
    zips = [p for p in matches if p.suffix.lower() == ".zip" and p.is_file()]
    if len(zips) == 1:
        return zips[0]
    if len(zips) > 1:
        names = ", ".join(p.name for p in zips)
        raise ValueError(f"Multiple ZIP uploads for job {job_id}: {names}")
    raise FileNotFoundError(
        f"No ZIP upload for job {job_id} in {settings.uploads_dir}",
    )


def _load_job_result_files(job_id: UUID) -> list[str]:
    job = job_store.get(job_id)
    if job is not None and job.result_files:
        return list(job.result_files)
    result_dir = get_result_dir(job_id)
    from app.services.zip_checkpoint import load_checkpoint

    ckpt = load_checkpoint(result_dir)
    if ckpt is not None:
        store = ZipCheckpointStore(
            job_id,
            result_dir,
            total_files=int(ckpt.get("total_files") or 0),
            data=ckpt,
        )
        files = store.seed_result_files()
        ckpt_path = checkpoint_path(result_dir)
        rel_ckpt = str(ckpt_path.relative_to(settings.project_root))
        if rel_ckpt not in files:
            files.append(rel_ckpt)
        if files:
            return files
    meta_path = result_dir / "job.json"
    if meta_path.is_file():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        files = data.get("result_files")
        if isinstance(files, list):
            return [str(f) for f in files]
    return []


def _zip_job_error_message(*, total: int, failed: int) -> str | None:
    if failed == 0:
        return None
    if failed == total:
        return f"All {total} ZIP members failed (see errors.json)"
    return (
        f"{failed}/{total} ZIP members failed "
        f"(partial result, see errors.json)"
    )


def resume_zip_checkpoint(
    job_id: UUID,
    *,
    allow_vision_escalation: bool | None = None,
) -> dict:
    """
    Resume a ZIP job from ``checkpoint.json`` (skip completed members).

    Requires the original archive in ``storage/uploads/{job_id}_*.zip``.
    """
    result_dir = get_result_dir(job_id)
    if not checkpoint_path(result_dir).is_file():
        raise FileNotFoundError(
            f"No checkpoint.json for job {job_id} ({checkpoint_path(result_dir)})",
        )

    ckpt = ZipCheckpointStore.load_or_create(
        job_id, result_dir, total_files=0,
    )
    if ckpt.status == "finished" and not ckpt.failed_members:
        return {
            "job_id": str(job_id),
            "resumed": 0,
            "skipped": ckpt.processed_count,
            "message": "checkpoint already finished",
        }

    zip_path = find_job_zip_upload(job_id)
    work_dir = settings.temp_dir / f"{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    job_store.update(job_id, status=JobStatus.PROCESSING)

    try:
        _run_zip_job(
            job_id,
            zip_path,
            work_dir,
            result_dir,
            source_name=zip_path.name,
            allow_vision_escalation=allow_vision_escalation,
            resume_from_checkpoint=True,
        )
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    refreshed = job_store.get(job_id)
    pending_left = 0
    if refreshed and refreshed.progress:
        pending_left = max(
            0,
            refreshed.progress.total - refreshed.progress.current,
        )
    return {
        "job_id": str(job_id),
        "resumed": refreshed.progress.current if refreshed and refreshed.progress else 0,
        "status": refreshed.status.value if refreshed else "unknown",
        "checkpoint": str(checkpoint_path(result_dir)),
        "pending_estimate": pending_left,
    }


def retry_zip_failed(
    job_id: UUID,
    *,
    allow_vision_escalation: bool | None = None,
) -> dict:
    """
    Re-process only ZIP members listed in ``errors.json`` for *job_id*.

    Requires the original archive in ``storage/uploads/{job_id}_*.zip``.
    Successful retries write member results and update ``errors.json`` / job state.
    """
    result_dir = get_result_dir(job_id)
    errors_path = result_dir / "errors.json"
    if not errors_path.is_file():
        raise FileNotFoundError(
            f"No errors.json for job {job_id} ({errors_path})",
        )

    payload = json.loads(errors_path.read_text(encoding="utf-8"))
    failures: list[dict] = list(payload.get("failures") or [])
    if not failures:
        return {
            "job_id": str(job_id),
            "retried": 0,
            "succeeded": 0,
            "still_failed": 0,
            "message": "errors.json has no failures to retry",
        }

    total_files = int(payload.get("total_files") or 0)
    prior_succeeded = int(payload.get("succeeded") or 0)
    failed_members = {f["member"] for f in failures}

    zip_path = find_job_zip_upload(job_id)
    work_dir = settings.temp_dir / f"{job_id}_retry"
    work_dir.mkdir(parents=True, exist_ok=True)
    extract_root = work_dir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
        members = plan_zip_members_from_archive(zip_path, extract_root)
        to_retry = [m for m in members if m.rel_path in failed_members]
        missing = failed_members - {m.rel_path for m in to_retry}
        if missing:
            logger.warning(
                "Retry job %s: members not in archive: %s",
                job_id, sorted(missing),
            )

        bulk_gpus = (
            get_bulk_ocr_gpu_ids() if settings.zip_pin_member_to_gpu else []
        )
        order_by_id = {id(m): i for i, m in enumerate(members)}

        def _pinned_for(member: ZipMember) -> int | None:
            if not bulk_gpus:
                return None
            idx = order_by_id.get(id(member), 0)
            return bulk_gpus[idx % len(bulk_gpus)]

        result_files = _load_job_result_files(job_id)
        still_failed: list[dict] = []
        recovered = 0

        for member in to_retry:
            pinned = _pinned_for(member)
            try:
                ensure_zip_member_extracted(zip_path, extract_root, member)
                try:
                    paths = _process_zip_member(
                        job_id,
                        member,
                        work_dir,
                        result_dir,
                        allow_vision_escalation=allow_vision_escalation,
                        pinned_gpu=pinned,
                    )
                finally:
                    release_zip_member_temp(extract_root, work_dir, member)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Retry failed for %s (job %s): %s",
                    member.rel_path, job_id, exc,
                )
                prior = next(
                    (f for f in failures if f["member"] == member.rel_path),
                    None,
                )
                retry_count = 1
                if prior is not None:
                    retry_count = int(prior.get("retry_count", 0)) + 1
                still_failed.append({
                    "member": member.rel_path,
                    "kind": member.kind.value,
                    "size_bytes": member.size_bytes,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "retry_count": retry_count,
                })
                ckpt_store = ZipCheckpointStore.load_or_create(
                    job_id, result_dir, total_files=total_files,
                )
                ckpt_store.record_failed(
                    member,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    retry_count=retry_count,
                )
                continue

            json_path, txt_path = paths
            result_files.extend([
                str(json_path.relative_to(settings.project_root)),
                str(txt_path.relative_to(settings.project_root)),
            ])
            recovered += 1
            ckpt_store = ZipCheckpointStore.load_or_create(
                job_id, result_dir, total_files=total_files,
            )
            ckpt_store.record_completed(
                member,
                json_path=json_path,
                txt_path=txt_path,
                retry_count=int(
                    next(
                        (f.get("retry_count", 0) for f in failures
                         if f["member"] == member.rel_path),
                        0,
                    ),
                ),
            )

        for f in failures:
            if f["member"] in missing:
                still_failed.append(f)

        new_succeeded = prior_succeeded + recovered
        if still_failed:
            err_path = _persist_zip_errors(
                result_dir, still_failed, total_files, new_succeeded,
            )
            if str(err_path.relative_to(settings.project_root)) not in result_files:
                result_files.append(
                    str(err_path.relative_to(settings.project_root)),
                )
            final_status = (
                JobStatus.FAILED if new_succeeded == 0 else JobStatus.DONE
            )
            error_msg = _zip_job_error_message(
                total=total_files, failed=len(still_failed),
            )
        else:
            errors_path.unlink(missing_ok=True)
            result_files = [
                f for f in result_files
                if not f.endswith("errors.json")
            ]
            final_status = JobStatus.DONE
            error_msg = None

        ckpt_finalize = ZipCheckpointStore.load_or_create(
            job_id, result_dir, total_files=total_files,
        )
        if not still_failed:
            ckpt_finalize.mark_finished()

        job_store.update(
            job_id,
            status=final_status,
            error=error_msg,
            result_files=result_files,
            progress=JobProgress(
                current=new_succeeded,
                total=total_files or new_succeeded + len(still_failed),
                unit="files",
                percent=round(
                    100.0 * new_succeeded
                    / max(total_files, new_succeeded + len(still_failed), 1),
                    1,
                ),
            ),
        )

        return {
            "job_id": str(job_id),
            "retried": len(to_retry),
            "succeeded": recovered,
            "still_failed": len(still_failed),
            "total_files": total_files,
            "members_ok": new_succeeded,
            "errors_json": still_failed != [],
            "status": final_status.value,
        }
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def _run_zip_job(
    job_id: UUID,
    input_path: Path,
    work_dir: Path,
    result_dir: Path,
    *,
    source_name: str | None,
    allow_vision_escalation: bool | None = None,
    resume_from_checkpoint: bool = False,
    members_to_process: list[ZipMember] | None = None,
    archive_total: int | None = None,
) -> None:
    extract_root = work_dir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    if members_to_process is not None:
        members = members_to_process
        total_files = archive_total if archive_total is not None else len(members)
    else:
        members = plan_zip_members_from_archive(input_path, extract_root)
        total_files = len(members)
    logger.info(
        "ZIP job %s: streaming plan — %d members (no bulk extract)",
        job_id,
        total_files,
    )
    checkpoint = ZipCheckpointStore.load_or_create(
        job_id,
        result_dir,
        total_files=total_files,
        source_archive=source_name or input_path.name,
    )
    result_files: list[str] = checkpoint.seed_result_files()
    failures: list[dict] = checkpoint.failure_dicts()
    skipped = 0
    if members_to_process is None and (
        resume_from_checkpoint or checkpoint.has_progress()
    ):
        pending = checkpoint.pending_members(members)
        skipped = len(members) - len(pending)
        if skipped:
            logger.info(
                "ZIP job %s: resuming from checkpoint — skip %d/%d done",
                job_id, skipped, total_files,
            )
        members = pending
    elif members_to_process is not None:
        members = [
            m for m in members
            if not member_has_results(result_dir, m.rel_path)
        ]
        skipped = len(members_to_process) - len(members)
        if skipped:
            logger.info(
                "ZIP job %s: missing-only run — skip %d already complete",
                job_id, skipped,
            )
    completed = checkpoint.processed_count
    zip_t0 = time.time()

    bulk_gpus = (
        get_bulk_ocr_gpu_ids() if settings.zip_pin_member_to_gpu else []
    )
    # id(member) → planned order index → pinned GPU.
    order_by_id: dict[int, int] = {id(m): i for i, m in enumerate(members)}

    def _pinned_for(member: ZipMember) -> int | None:
        if not bulk_gpus:
            return None
        idx = order_by_id.get(id(member), 0)
        return bulk_gpus[idx % len(bulk_gpus)]

    def _safe_processor(member: ZipMember) -> _MemberOutcome:
        pinned = _pinned_for(member)
        try:
            ensure_zip_member_extracted(input_path, extract_root, member)
            try:
                paths = _process_zip_member(
                    job_id, member, work_dir, result_dir,
                    allow_vision_escalation=allow_vision_escalation,
                    pinned_gpu=pinned,
                )
            finally:
                release_zip_member_temp(extract_root, work_dir, member)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "ZIP member %s failed (gpu=%s): %s",
                member.rel_path, pinned, exc,
            )
            return _MemberOutcome(
                ok=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )
        return _MemberOutcome(ok=True, paths=paths)

    def _on_member_done(member: ZipMember, outcome: _MemberOutcome) -> None:
        nonlocal completed
        completed += 1
        rel = member.rel_path
        failures[:] = [f for f in failures if f.get("member") != rel]
        if outcome.ok and outcome.paths is not None:
            json_path, txt_path = outcome.paths
            result_files.extend([
                str(json_path.relative_to(settings.project_root)),
                str(txt_path.relative_to(settings.project_root)),
            ])
            checkpoint.record_completed(
                member,
                json_path=json_path,
                txt_path=txt_path,
            )
        else:
            failure = {
                "member": rel,
                "kind": member.kind.value,
                "size_bytes": member.size_bytes,
                "error": outcome.error or "unknown error",
                "error_type": outcome.error_type or "Exception",
                "retry_count": 0,
            }
            failures.append(failure)
            checkpoint.record_failed(
                member,
                error=failure["error"],
                error_type=failure["error_type"],
            )
        job_store.update(
            job_id,
            progress=JobProgress(
                current=completed,
                total=total_files,
                unit="files",
                percent=round(100.0 * completed / total_files, 1),
            ),
        )

    worker_restarts: int | None = None
    if not members:
        succeeded = len(checkpoint.completed_members)
        failures = checkpoint.failure_dicts()
        if failures:
            errors_path = _persist_zip_errors(
                result_dir, failures, total_files, succeeded,
            )
            result_files.append(
                str(errors_path.relative_to(settings.project_root)),
            )
            final_status = (
                JobStatus.FAILED if succeeded == 0 else JobStatus.DONE
            )
            error_msg = _zip_job_error_message(
                total=total_files, failed=len(failures),
            )
        else:
            final_status = JobStatus.DONE
            error_msg = None
            checkpoint.mark_finished()
        ckpt_rel = str(checkpoint.path.relative_to(settings.project_root))
        if ckpt_rel not in result_files:
            result_files.append(ckpt_rel)
        job_store.update(
            job_id,
            status=final_status,
            progress=JobProgress(
                current=completed,
                total=total_files,
                unit="files",
                percent=round(100.0 * completed / max(total_files, 1), 1),
            ),
            error=error_msg,
            result_files=result_files,
        )
        logger.info(
            "ZIP job %s: nothing pending (%d skipped from checkpoint)",
            job_id, skipped,
        )
        return

    from app.services.zip_worker import ZipWorkResult

    def _on_pool_result(member: ZipMember, result: ZipWorkResult) -> None:
        if result.ok and result.json_path and result.txt_path:
            outcome = _MemberOutcome(
                ok=True,
                paths=(Path(result.json_path), Path(result.txt_path)),
            )
        else:
            outcome = _MemberOutcome(
                ok=False,
                error=result.error,
                error_type=result.error_type,
            )
        _on_member_done(member, outcome)

    # ── adaptive queue (3 concurrent tier-pools) vs. original FIFO queue ──
    if settings.zip_adaptive_queue_enabled and len(members) >= 2:
        adaptive_queue = AdaptiveZipQueue(members)
        logger.info(
            "ZIP job %s: adaptive queue — %s",
            job_id, adaptive_queue.tier_summary(),
        )
        pool_metrics = adaptive_queue.run_in_process_pool(
            job_id=job_id,
            archive_path=input_path,
            extract_root=extract_root,
            work_dir=work_dir,
            result_dir=result_dir,
            pinned_for=_pinned_for,
            allow_vision_escalation=allow_vision_escalation,
            gpu_ids=bulk_gpus,
            on_complete=_on_pool_result,
        )
        worker_restarts = pool_metrics.worker_restart_count
        # Include tier report in result_files so it is accessible via API
        tier_report_path = result_dir / "zip_tier_report.json"
        if tier_report_path.exists():
            try:
                result_files.append(
                    tier_report_path.relative_to(settings.project_root).as_posix()
                )
            except ValueError:
                pass
    else:
        # Original single-pool path (used when adaptive disabled or 1 member)
        queue = ZipProcessingQueue(members)
        if queue.max_workers <= 1:
            _warmup_zip_ocr_pool()
            queue.run(_safe_processor, on_complete=_on_member_done)
        else:
            pool_metrics = queue.run_in_process_pool(
                job_id=job_id,
                archive_path=input_path,
                extract_root=extract_root,
                work_dir=work_dir,
                result_dir=result_dir,
                pinned_for=_pinned_for,
                allow_vision_escalation=allow_vision_escalation,
                gpu_ids=bulk_gpus,
                on_complete=_on_pool_result,
            )
            worker_restarts = pool_metrics.worker_restart_count

    recycle_path = (result_dir / "worker_recycle.json").relative_to(settings.project_root)
    if recycle_path.as_posix() not in result_files:
        result_files.append(recycle_path.as_posix())

    failures = checkpoint.failure_dicts()
    succeeded = len(checkpoint.completed_members)
    if failures:
        errors_path = _persist_zip_errors(
            result_dir, failures, total_files, succeeded,
        )
        result_files.append(
            str(errors_path.relative_to(settings.project_root)),
        )

    if not failures:
        final_status = JobStatus.DONE
        error_msg: str | None = None
        checkpoint.mark_finished()
    elif succeeded == 0:
        final_status = JobStatus.FAILED
        error_msg = _zip_job_error_message(total=total_files, failed=len(failures))
    else:
        final_status = JobStatus.DONE
        error_msg = _zip_job_error_message(total=total_files, failed=len(failures))

    ckpt_rel = str(checkpoint.path.relative_to(settings.project_root))
    if ckpt_rel not in result_files:
        result_files.append(ckpt_rel)

    from app.services.ocr_cache import get_ocr_cache_metrics, log_ocr_cache_metrics
    from app.services.ocr_lang_routing import (
        get_ocr_lang_routing_metrics,
        log_ocr_lang_routing_metrics,
    )

    log_ocr_cache_metrics(label=f"ZIP job {job_id}")
    log_ocr_lang_routing_metrics(label=f"ZIP job {job_id}")
    cache_metrics_path = result_dir / "ocr_cache_metrics.json"
    try:
        cache_metrics_path.write_text(
            json.dumps(get_ocr_cache_metrics().to_dict(), indent=2),
            encoding="utf-8",
        )
        rel = str(cache_metrics_path.relative_to(settings.project_root))
        if rel not in result_files:
            result_files.append(rel)
        lang_metrics_path = result_dir / "ocr_lang_routing_metrics.json"
        lang_metrics_path.write_text(
            json.dumps(get_ocr_lang_routing_metrics().to_dict(), indent=2),
            encoding="utf-8",
        )
        rel_lang = str(lang_metrics_path.relative_to(settings.project_root))
        if rel_lang not in result_files:
            result_files.append(rel_lang)
    except OSError as exc:
        logger.debug("Could not write OCR cache metrics: %s", exc)

    job_store.update(
        job_id,
        status=final_status,
        progress=JobProgress(
            current=completed,
            total=total_files,
            unit="files",
            percent=round(100.0 * completed / max(total_files, 1), 1),
        ),
        error=error_msg,
        result_files=result_files,
    )
    log_extra = (
        f", worker_restarts={worker_restarts}" if worker_restarts is not None else ""
    )
    logger.info(
        "ZIP job %s: %d/%d ok, %d failed in %.2fs (workers=%d, pin_gpu=%s%s, "
        "checkpoint_skip=%d)",
        job_id, succeeded, total_files, len(failures),
        time.time() - zip_t0, queue.max_workers, bool(bulk_gpus), log_extra,
        skipped,
    )


def run_job(
    job_id: UUID,
    input_path: Path,
    *,
    source_name: str | None = None,
    allow_vision_escalation: bool | None = None,
    document_language: str | None = None,
) -> None:
    from app.services.document_language import document_language_context

    with document_language_context(document_language):
        _run_job_impl(
            job_id,
            input_path,
            source_name=source_name,
            allow_vision_escalation=allow_vision_escalation,
        )


def _run_job_impl(
    job_id: UUID,
    input_path: Path,
    *,
    source_name: str | None = None,
    allow_vision_escalation: bool | None = None,
) -> None:
    work_dir = settings.temp_dir / str(job_id)
    result_dir = allocate_result_dir(job_id, source_name)

    try:
        job_store.update(job_id, status=JobStatus.PROCESSING)
        work_dir.mkdir(parents=True, exist_ok=True)

        kind = _source_kind(input_path)

        # ── Fast path: single photo / paragon ───────────────────────────────
        if kind == SourceKind.IMAGE:
            result = _process_single_image(
                job_id,
                input_path,
                source_name=source_name,
                allow_vision_escalation=allow_vision_escalation,
            )
            write_results(result, result_dir)
            job_store.update(
                job_id,
                status=JobStatus.DONE,
                progress=JobProgress(current=1, total=1, unit="pages", percent=100.0),
                error=None,
                result_files=[
                    str((result_dir / "result.json").relative_to(settings.project_root)),
                    str((result_dir / "result.txt").relative_to(settings.project_root)),
                ],
            )
            return

        if kind == SourceKind.ZIP:
            _run_zip_job(
                job_id,
                input_path,
                work_dir,
                result_dir,
                source_name=source_name,
                allow_vision_escalation=allow_vision_escalation,
            )
            return

        # ── Bulk path: PDF ────────────────────────────────────────────────────
        result = _process_bulk_document(
            input_path, work_dir, source_name=source_name, job_id=job_id,
        )
        total_pages = result.page_count

        json_path, txt_path = write_results(result, result_dir)
        job_store.update(
            job_id,
            status=JobStatus.DONE,
            progress=JobProgress(
                current=total_pages,
                total=total_pages,
                unit="pages",
                percent=100.0,
            ),
            error=None,
            result_files=[
                str(json_path.relative_to(settings.project_root)),
                str(txt_path.relative_to(settings.project_root)),
            ],
        )
    except IngestError as exc:
        job_store.update(job_id, status=JobStatus.FAILED, error=str(exc))
    except Exception as exc:
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error=str(exc),
        )
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
