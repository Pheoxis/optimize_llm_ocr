"""Classify OCR document blocks (faktura / paragon / list) and merge or split them."""

from __future__ import annotations

import re
from enum import Enum


class DocumentKind(str, Enum):
    FAKTURA = "faktura"
    PARAGON = "paragon"
    LIST = "list"
    OTHER = "other"


_FV_NUMBER_RE = re.compile(
    r"\bFV\s*[/\\-]?\s*[\dA-Za-z./\\-]{4,}\b",
    re.IGNORECASE,
)
_INVOICE_NUM_RE = re.compile(
    r"(?:faktura\s+(?:vat\s+)?numer|numer\s+faktury)\s*[:\s]*([\w./\\-]+)",
    re.IGNORECASE,
)
_NIP_RE = re.compile(r"\b(?:NIP|NiP)\s*[:\s.]+\s*(\d[\d\s-]{8,12}\d)\b", re.IGNORECASE)
_POTWIERDZENIE_RE = re.compile(r"potwierdzenie\s+(?:sprzeda[żz]y|wydania)", re.IGNORECASE)


def _norm_nip(raw: str) -> str:
    return re.sub(r"\D", "", raw)


def extract_invoice_numbers(text: str) -> list[str]:
    """Distinct invoice / FV identifiers found in OCR text."""
    found: list[str] = []
    seen: set[str] = set()
    for m in _FV_NUMBER_RE.finditer(text):
        val = re.sub(r"\s+", "", m.group(0)).upper()
        if val not in seen:
            seen.add(val)
            found.append(val)
    for m in _INVOICE_NUM_RE.finditer(text):
        val = m.group(1).strip().upper()
        if val and val not in seen:
            seen.add(val)
            found.append(val)
    return found


def extract_seller_nips(text: str) -> list[str]:
    nips: list[str] = []
    seen: set[str] = set()
    for m in _NIP_RE.finditer(text):
        n = _norm_nip(m.group(1))
        if len(n) >= 10 and n not in seen:
            seen.add(n)
            nips.append(n)
    return nips


def classify_document(text: str) -> DocumentKind:
    """Coarse document type from OCR keywords (exact transcription preserved elsewhere)."""
    t = text.strip()
    if not t:
        return DocumentKind.OTHER
    u = t.upper()

    if "PARAGON FISKALNY" in u or "PARIAGON FISKALNY" in u or (
        "PARAGON" in u and any(m in u for m in ("SUMA PLN", "NR TRANSAKCJI", "KASJER"))
    ):
        return DocumentKind.PARAGON

    if any(
        m in u
        for m in (
            "FAKTURA VAT",
            "FAKTURA NUMER",
            "FAKTURA\n",
            " FAKTURA",
            "POTWIERDZENIE SPRZEDA",
            "POTWIERDZENIE WYDANIA",
        )
    ):
        return DocumentKind.FAKTURA
    # Catch garbled invoices where FAKTURA is at the very start (e.g. upside-down scans).
    if u.startswith("FAKTURA") and (
        "NIP" in u or "SPRZEDAWCA" in u or "NABYWCA" in u or "NUMER" in u
        or "WART" in u or "SUMA" in u
    ):
        return DocumentKind.FAKTURA
    if _POTWIERDZENIE_RE.search(t):
        return DocumentKind.FAKTURA
    if _FV_NUMBER_RE.search(t) and ("NIP" in u or "SPRZEDAWCA" in u or "NABYWCA" in u):
        return DocumentKind.FAKTURA

    if any(
        m in u
        for m in (
            "UBEZPIECZEN",
            "SZANOWNI PANSTWO",
            "SZANOWNI PAŃSTWO",
            "POLISA",
            "COMPENSA",
            "VIENNA INSURANCE",
        )
    ) and "FAKTURA" not in u:
        return DocumentKind.LIST

    return DocumentKind.OTHER


def document_start_markers(text: str) -> list[int]:
    """Line indices where a new logical document likely begins."""
    lines = text.splitlines()
    starts: list[int] = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        u = s.upper()
        if any(
            u.startswith(prefix)
            for prefix in (
                "FAKTURA VAT",
                "FAKTURA NUMER",
                "FAKTURA ",
                "PARAGON FISKALNY",
                "PARIAGON FISKALNY",  # OCR variant of PARAGON FISKALNY
                "PARAGON ",
            )
        ):
            starts.append(i)
            continue
        if _POTWIERDZENIE_RE.search(s):
            starts.append(i)
            continue
        if _FV_NUMBER_RE.match(s) and i > 0:
            starts.append(i)
    return starts


def split_multi_document_block(text: str) -> list[str]:
    """Split one OCR block that accidentally contains 2+ separate documents."""
    t = text.strip()
    if not t:
        return []

    markers = document_start_markers(t)
    inv_nums = extract_invoice_numbers(t)
    # Need at least two distinct invoice ids OR two strong start markers.
    if len(markers) < 2 and len(inv_nums) < 2:
        return [t]

    lines = t.splitlines()
    if len(markers) >= 2:
        # Receipt boundary keywords that signal the previous document has ended.
        _RECEIPT_END_MARKERS = frozenset({
            "ZAPLACONO", "ZAPŁACONO", "SUMA PLN", "NR TRANSAKCJI", "NR. TRANSAKCJI",
            "NR TRANSAKC", "EBT ", "ROZLICZENIE", "ZAKUP:", "HASH:", "HASH ",
        })
        # For each marker boundary, find the real split point: look back from the
        # marker for the previous receipt's last line, then include everything
        # after it as the pre-header of the new document.
        actual_starts: list[int] = [0]  # First doc always starts from line 0
        for idx in range(1, len(markers)):
            start = markers[idx]
            prev_marker_start = markers[idx - 1]
            actual = start  # default: split right at the marker
            for lookback in range(1, min(20, start - prev_marker_start)):
                candidate = start - lookback
                candidate_line = lines[candidate].strip().upper() if candidate >= 0 else ""
                if any(em in candidate_line for em in _RECEIPT_END_MARKERS):
                    # Previous receipt ends at candidate; new doc pre-header starts after it.
                    actual = candidate + 1
                    break
            actual_starts.append(actual)

        # Build parts with non-overlapping ranges.
        parts: list[str] = []
        for idx, a_start in enumerate(actual_starts):
            end = actual_starts[idx + 1] if idx + 1 < len(actual_starts) else len(lines)
            chunk = "\n".join(lines[a_start:end]).strip()
            if chunk:
                parts.append(chunk)
        if len(parts) >= 2:
            return parts

    if len(inv_nums) >= 2:
        # Split on second occurrence of a different FV pattern line.
        second_pat = inv_nums[1]
        idx = t.upper().find(second_pat)
        if idx > 80:
            left = t[:idx].strip()
            right = t[idx:].strip()
            if left and right:
                return [left, right]

    # OCR can glue several receipts/faktury into one long line/block where
    # line-start markers are lost.  Split by repeated strong headers inside
    # the text body.
    upper = t.upper()
    hard_headers = (
        "PARAGON FISKALNY",
        "PARIAGON FISKALNY",
        "FAKTURA VAT",
        " FAKTURA ",
        "\nFAKTURA",
        "DOKUMENT NIEFISKALNY",
    )
    cut_points: list[int] = []
    for hh in hard_headers:
        start = 0
        while True:
            pos = upper.find(hh, start)
            if pos < 0:
                break
            # Skip the very first header at block start.
            if pos > 120:
                cut_points.append(pos)
            start = pos + max(1, len(hh) // 2)
    if cut_points:
        cuts = sorted(set(cut_points))
        # Keep only meaningful boundaries to avoid tiny garbage splits.
        filtered: list[int] = []
        prev = 0
        for c in cuts:
            if c - prev >= 120:
                filtered.append(c)
                prev = c
        if filtered:
            parts: list[str] = []
            boundaries = [0] + filtered + [len(t)]
            for i in range(len(boundaries) - 1):
                chunk = t[boundaries[i]:boundaries[i + 1]].strip()
                if len(chunk) >= 40:
                    parts.append(chunk)
            if len(parts) >= 2:
                return parts

    return [t]


def is_invoice_continuation(prev: str, nxt: str) -> bool:
    """Next block is part of the same invoice (e.g. Nabywca column / page footer)."""
    pu, nu = prev.upper(), nxt.upper()
    prev_nums = extract_invoice_numbers(prev)
    next_nums = extract_invoice_numbers(nxt)

    # New invoice number → not a continuation.
    if next_nums and prev_nums and next_nums[0] != prev_nums[0]:
        if classify_document(nxt) == DocumentKind.FAKTURA:
            return False

    # Strong new document header on next block.
    if any(
        nu.startswith(h)
        for h in (
            "FAKTURA VAT",
            "FAKTURA NUMER",
            "PARAGON FISKALNY",
        )
    ):
        if not prev_nums or (next_nums and next_nums[0] != prev_nums[0]):
            return False

    if _POTWIERDZENIE_RE.search(nxt) and classify_document(prev) == DocumentKind.FAKTURA:
        return False

    # Tail: buyer section, totals, second column of same invoice.
    if "NABYWCA" in nu and classify_document(prev) == DocumentKind.FAKTURA:
        if not any(nu.startswith(h) for h in ("FAKTURA VAT", "FAKTURA NUMER")):
            prev_nips = extract_seller_nips(prev)
            next_nips = extract_seller_nips(nxt)
            if not next_nips or not prev_nips or next_nips[-1] == prev_nips[-1]:
                return True

    if classify_document(prev) == DocumentKind.FAKTURA:
        if not classify_document(nxt) == DocumentKind.FAKTURA:
            if "NABYWCA" in nu or "RAZEM:" in nu or "STRONA" in nu:
                return True
        if prev_nums and next_nums and prev_nums[0] == next_nums[0]:
            return True

    return False


def is_receipt_continuation(prev: str, nxt: str) -> bool:
    """Existing receipt-specific continuation heuristics."""
    ua, ub = prev.upper(), nxt.upper()

    # A block that starts with a new document header is never a continuation.
    _new_doc_headers = (
        "FAKTURA VAT",
        "FAKTURA NUMER",
        "FAKTURA\n",
        "PARAGON FISKALNY",
    )
    nxt_stripped = nxt.strip().upper()
    if any(nxt_stripped.startswith(h) for h in _new_doc_headers):
        return False
    # If next block starts with FAKTURA and prev is a receipt, not a continuation.
    if nxt_stripped.startswith("FAKTURA") and classify_document(prev) == DocumentKind.PARAGON:
        return False

    if "CHINKALI" in ua and (
        "CHINKALI" in ub or "NORD" in ub or "PELME" in ub or "ROZLICZENIE" in ub
    ):
        return True
    if "AMIC" in ua and "AMIC" in ub:
        return True
    if "CARREFOUR" in ua and "CARREFOUR" in ub:
        return True
    if "EMPIK" in ua and ("EMPIK" in ub or "EMPIK.COM" in ub):
        return True
    if "PARAGON" in ua and ("SUMA" in ub or "ZAPLACONO" in ub or "ZAPŁACONO" in ub):
        if "PARAGON FISKALNY" not in ub:
            return True
    return False


def _looks_invoice_fragment(text: str) -> bool:
    u = text.upper()
    if "PARAGON FISKALNY" in u:
        return False
    markers = (
        "FAKTURA",
        "SPRZEDAWCA",
        "NABYWCA",
        "DO ZAPLATY",
        "DO ZAPŁATY",
        "WG STAWKI",
        "WARTOŚĆ",
        "WARTOSC",
        "NIP",
    )
    return sum(1 for m in markers if m in u) >= 2


def should_merge_blocks(prev: str, nxt: str) -> bool:
    # Strong receipts/invoices rules.
    if is_invoice_continuation(prev, nxt):
        return True
    if is_receipt_continuation(prev, nxt):
        return True

    pk = classify_document(prev)
    nk = classify_document(nxt)
    if (
        (pk == DocumentKind.FAKTURA and nk in (DocumentKind.OTHER, DocumentKind.LIST))
        or (nk == DocumentKind.FAKTURA and pk in (DocumentKind.OTHER, DocumentKind.LIST))
    ) and _looks_invoice_fragment(prev) and _looks_invoice_fragment(nxt):
        return True

    # For non-invoice documents (letters/announcements), OCR segmentation
    # often produces “left column” + “right column” fragments.
    # Users want a continuous flow; merge LIST/OTHER adjacent fragments.
    if pk in (DocumentKind.LIST, DocumentKind.OTHER) and nk in (
        DocumentKind.LIST,
        DocumentKind.OTHER,
    ):
        return True

    return False


def merge_adjacent_blocks(texts: list[str]) -> list[str]:
    """Join consecutive OCR blocks that belong to one physical document."""
    if not texts:
        return []
    out: list[str] = [texts[0].strip()]
    for nxt in texts[1:]:
        t = nxt.strip()
        if not t:
            continue
        if should_merge_blocks(out[-1], t):
            # Prevent OCR “left/right columns” boundary from creating
            # visible gaps by trimming trailing whitespace from the previous
            # fragment before concatenation.
            out[-1] = out[-1].rstrip() + "\n" + t
        else:
            out.append(t)
    return out


def consolidate_page_blocks(
    texts: list[str],
    metas: list[dict],
    *,
    merge_same_page: bool = True,
) -> list[str]:
    """
    Turn per-region OCR into one block per page when regions are fragments
    of a single centered invoice/letter (typical SKMBT scan: 1 doc / page).
    """
    if not texts:
        return []

    if not merge_same_page or not metas:
        return merge_adjacent_blocks(texts)

    # Each entry: (original_index, text, no_auto_merge_flag)
    by_page: dict[int, list[tuple[int, str, bool]]] = {}
    for i, (txt, meta) in enumerate(zip(texts, metas)):
        page = int(meta.get("page", 0) or 0)
        no_merge = bool(meta.get("_no_auto_merge"))
        by_page.setdefault(page, []).append((i, txt.strip(), no_merge))

    page_blocks: list[str] = []
    for page in sorted(by_page):
        sorted_items = sorted(by_page[page], key=lambda x: x[0])
        items_with_flags = [(t, nm) for _, t, nm in sorted_items if t]
        if not items_with_flags:
            continue
        if len(items_with_flags) == 1:
            page_blocks.append(items_with_flags[0][0])
            continue

        # Split into "merge groups" at _no_auto_merge boundaries.
        # Items marked _no_auto_merge are emitted as individual blocks.
        # Adjacent items without the flag can be merged together.
        groups: list[list[str]] = []
        current_group: list[str] = []
        for txt, no_merge in items_with_flags:
            if no_merge:
                # Flush current group first, then emit this item alone.
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([txt])
            else:
                current_group.append(txt)
        if current_group:
            groups.append(current_group)

        for group in groups:
            if len(group) == 1:
                page_blocks.append(group[0])
            else:
                merged = merge_adjacent_blocks(group)
                if len(merged) > 1:
                    page_blocks.extend(merged)
                else:
                    page_blocks.append(merged[0])

    return page_blocks


def kind_label(kind: DocumentKind, index: int, total: int) -> str | None:
    if kind == DocumentKind.FAKTURA:
        return f"Faktura {index}/{total}" if total > 1 else "Faktura"
    if kind == DocumentKind.PARAGON:
        return f"Paragon {index}/{total}" if total > 1 else "Paragon"
    if kind == DocumentKind.LIST:
        return f"List {index}/{total}" if total > 1 else "List"
    return None


def clean_visual_rule_lines(block: str) -> str:
    """Remove OCR lines that are only printed horizontal rules (---, ___)."""
    lines = block.splitlines()
    kept: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s and set(s) <= {"-", "_"} and len(s) >= 3:
            continue
        kept.append(ln)
    out: list[str] = []
    blank = 0
    for ln in kept:
        if not ln.strip():
            blank += 1
            if blank <= 2:
                out.append("")
            continue
        blank = 0
        out.append(ln)
    return "\n".join(out).strip()


def format_document_stream(texts: list[str], *, separator: str = "\n\n") -> str:
    """Classify, split, and label documents for downstream LLM consumption."""
    expanded: list[str] = []
    for raw in texts:
        t = raw.strip()
        if not t or len(t) < 40:
            continue
        for part in split_multi_document_block(t):
            p = clean_visual_rule_lines(part)
            if p and len(p) >= 40:
                expanded.append(p)

    if not expanded:
        return ""

    kinds = [classify_document(t) for t in expanded]
    invoice_total = sum(1 for k in kinds if k == DocumentKind.FAKTURA)
    paragon_total = sum(1 for k in kinds if k == DocumentKind.PARAGON)
    list_total = sum(1 for k in kinds if k == DocumentKind.LIST)
    f_i = p_i = l_i = 0

    parts: list[str] = []
    prev_kind: DocumentKind | None = None
    for block, kind in zip(expanded, kinds):
        if parts:
            # For LIST/OTHER adjacent fragments we want a continuous flow
            # (no “left vs right column” split).
            if (
                prev_kind in (DocumentKind.LIST, DocumentKind.OTHER)
                and kind in (DocumentKind.LIST, DocumentKind.OTHER)
            ):
                parts.append("\n")
            else:
                parts.append(separator)
        if kind == DocumentKind.FAKTURA:
            f_i += 1
            label = kind_label(kind, f_i, invoice_total)
        elif kind == DocumentKind.PARAGON:
            p_i += 1
            label = kind_label(kind, p_i, paragon_total)
        elif kind == DocumentKind.LIST:
            l_i += 1
            label = kind_label(kind, l_i, list_total)
        else:
            label = None
        if label:
            parts.append(label + "\n")
        # When merging LIST/OTHER fragments, normalize whitespace at the
        # boundary so we don't create a visible "gap" between columns.
        if (
            prev_kind in (DocumentKind.LIST, DocumentKind.OTHER)
            and kind in (DocumentKind.LIST, DocumentKind.OTHER)
        ):
            if parts:
                parts[-1] = parts[-1].rstrip()
            block = block.lstrip()
        parts.append(block)
        prev_kind = kind
    out = "".join(parts)
    # OCR often injects multiple blank lines around layout boundaries.
    # Keep at most one empty line to preserve a “continuous” flow.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out
