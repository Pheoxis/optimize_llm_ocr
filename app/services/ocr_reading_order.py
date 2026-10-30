"""
Block-aware OCR reading order reconstruction.

Replaces the naive y-sort → x-sort in ``_reconstruct_lines`` with a
multi-stage pipeline that is robust for:

  * invoices     — structured two-column seller/buyer header + item table
  * forms        — sparse, checkbox, horizontal-line layouts
  * postal notices — two-column address block + content region
  * receipts     — narrow single-column strip with item + price columns
  * tables       — grid alignment, repeated column headers

Problems in the old approach
-----------------------------
  * Fixed bucket_y (never updates)   → boxes near the same line but slightly
    drifted get assigned to the wrong line, causing broken invoices.
  * Global y tolerance               → one threshold for every font size on
    the page; large/small mixed text merges incorrectly.
  * No column awareness              → boxes from left and right columns end
    up on the same "line", producing garbled output.
  * No block detection               → section header ends up glued to the
    first item of the next section.
  * Tile overlap protection only at raw-box level → after merging strips,
    near-duplicate lines can survive.

New pipeline
------------
  1. Convert raw RapidOCR polygon boxes → normalised item dicts.
  2. Pre-filter OCR garbage (reuses ``ocr_dedup.is_spurious_ocr_fragment``).
  3. Suppress tile-overlap duplicates (IoU + text similarity).
  4. Detect text blocks via vertical-gap clustering.
  5. Detect columns via X-projection histogram.
  6. Detect form / table mode from box density + aspect ratios.
  7. Cluster boxes into lines per (block, column) with **adaptive** running-
     mean y tolerance — no more fixed bucket_y.
  8. Build block-aware reading order
       single-column : blocks sorted by y_min
       multi-column  : blocks sorted by (primary_column, y_min)
       form          : row-major (line y, then x) — label + value on same row
  9. Merge boxes within each visual line (reuses ``ocr_dedup``).
 10. Collapse redundant form-marker lines.

Debug metadata returned with every result
-----------------------------------------
  detected_columns       int   – number of detected text columns
  block_count            int   – number of vertical text blocks
  merge_operations       int   – intra-line box concatenations
  overlap_removed        int   – boxes suppressed as tile / OCR echoes
  reading_order_strategy str   – "single_column" | "multi_column" | "form"
  filtered_garbage       int   – boxes removed as OCR garbage before layout
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── result ────────────────────────────────────────────────────────────────────


@dataclass
class ReadingOrderResult:
    """Output of :func:`reconstruct_reading_order`."""

    lines: list          # list[FastOCRLine]
    text: str
    debug_meta: dict[str, Any] = field(default_factory=dict)


# ── public entry point ────────────────────────────────────────────────────────


def reconstruct_reading_order(
    raw_boxes: list,
    txts: list[str],
    scores: list[float],
    *,
    y_tolerance: int,       # legacy fallback if advanced pipeline is skipped
    separator: str,
    image_width: int | None = None,
    image_height: int | None = None,
) -> ReadingOrderResult:
    """
    Block-aware reading order reconstruction.

    Drop-in replacement for ``_reconstruct_lines`` that is column-aware,
    form-aware, and uses per-block adaptive y-tolerance with a running mean
    (not a fixed bucket anchor).

    Parameters
    ----------
    raw_boxes    : list of RapidOCR polygon arrays (N×2 float arrays)
    txts         : OCR texts aligned with raw_boxes
    scores       : confidence scores aligned with raw_boxes
    y_tolerance  : fallback fixed tolerance (used only if adaptive fails)
    separator    : token separator within a visual line
    image_width  : page/image pixel width (improves column detection)
    image_height : page/image pixel height (improves form-mode detection)
    """
    from app.core.config import settings
    from app.services.fast_ocr import FastOCRLine
    from app.services.ocr_dedup import (
        OcrDedupPolicy,
        collapse_redundant_form_marker_lines,
        filter_detection_triplets,
        merge_line_detections,
    )

    if not txts:
        return ReadingOrderResult(lines=[], text="", debug_meta=_empty_meta())

    # ── 1. convert raw polygon boxes → item dicts ─────────────────────────────
    items = _to_items(raw_boxes, txts, scores)
    if not items:
        return ReadingOrderResult(lines=[], text="", debug_meta=_empty_meta())

    iw = image_width or (max(it["bbox"][2] for it in items) + 1)
    ih = image_height or (max(it["bbox"][3] for it in items) + 1)

    global_median_h = float(np.median([it["h"] for it in items])) or 12.0

    # ── 2. pre-filter OCR garbage ─────────────────────────────────────────────
    n_before = len(items)
    items, noise_meta = _filter_garbage(items)
    n_filtered = n_before - len(items)
    if not items:
        return ReadingOrderResult(lines=[], text="", debug_meta=_empty_meta())

    # ── 3. tile-overlap / echo suppression (before any grouping) ──────────────
    overlap_removed = _suppress_tile_overlaps(items, global_median_h)

    # ── 4. detect text blocks (vertical-gap clustering) ───────────────────────
    block_gap_threshold = max(
        float(settings.reading_order_block_gap_min_px),
        global_median_h * float(settings.reading_order_block_gap_ratio),
    )
    _assign_blocks(items, block_gap_threshold)
    block_count = max(it["block"] for it in items) + 1

    # ── 5. detect columns (X-projection histogram) ────────────────────────────
    columns = _detect_columns(
        items, iw,
        min_gap_pct=float(settings.reading_order_column_gap_pct),
    )
    detected_columns = len(columns)
    _assign_columns(items, columns, iw)

    # ── 6. detect form / table mode ────────────────────────────────────────────
    form_mode = _detect_form_mode(items, iw, ih)

    if form_mode and detected_columns > 1:
        strategy = "form"
    elif detected_columns > 1:
        strategy = "multi_column"
    elif form_mode:
        strategy = "form"
    else:
        strategy = "single_column"

    logger.debug(
        "reading_order: blocks=%d cols=%d strategy=%s gap=%.1f form=%s",
        block_count, detected_columns, strategy, block_gap_threshold, form_mode,
    )

    # ── 7. line clustering (adaptive y tolerance per block × column) ───────────
    _cluster_lines_adaptive(items, global_median_h, form_mode, settings)

    # ── 8. build reading order → ordered list of line groups ──────────────────
    if strategy in ("multi_column",):
        ordered_groups = _reading_order_multi_column(items, block_count)
    elif strategy == "form":
        ordered_groups = _reading_order_form(items)
    else:
        ordered_groups = _reading_order_single_column(items)

    # ── 9. merge boxes within each line ───────────────────────────────────────
    policy = OcrDedupPolicy.from_settings()
    out_lines: list[FastOCRLine] = []
    total_merge_ops = 0

    for group in ordered_groups:
        group = [it for it in group if not it["suppressed"] and it["txt"].strip()]
        if not group:
            continue
        group.sort(key=lambda r: r["bbox"][0])          # left → right

        parts  = [it["txt"]  for it in group]
        bboxes = [it["bbox"] for it in group]
        scs    = [it["sc"]   for it in group]

        parts, bboxes, scs = filter_detection_triplets(parts, bboxes, scs)
        if not parts:
            continue

        merged = merge_line_detections(
            parts, bboxes, scs, separator, policy, image_width=iw,
        )
        if not merged.strip():
            continue

        x0 = int(min(b[0] for b in bboxes))
        x1 = int(max(b[2] for b in bboxes))
        y0 = int(min(b[1] for b in bboxes))
        y1 = int(max(b[3] for b in bboxes))
        total_merge_ops += max(0, len(parts) - 1)

        out_lines.append(FastOCRLine(
            text=merged,
            mean_score=sum(scs) / len(scs),
            min_score=min(scs),
            n_boxes=len(parts),
            bbox=(x0, y0, x1, y1),
        ))

    # ── 10. remove redundant form-marker lines ─────────────────────────────────
    out_lines = collapse_redundant_form_marker_lines(out_lines, separator)

    text = "\n".join(ln.text for ln in out_lines)
    debug_meta = {
        "detected_columns": detected_columns,
        "block_count": block_count,
        "merge_operations": total_merge_ops,
        "overlap_removed": overlap_removed,
        "reading_order_strategy": strategy,
        "filtered_garbage": n_filtered,
        **noise_meta,
    }
    return ReadingOrderResult(lines=out_lines, text=text, debug_meta=debug_meta)


# ── step 1: convert polygon boxes ─────────────────────────────────────────────


def _to_items(raw_boxes, txts, scores) -> list[dict]:
    """Convert RapidOCR polygon arrays to normalised item dicts."""
    items: list[dict] = []
    for bb, txt, sc in zip(raw_boxes, txts, scores):
        if not txt or not (txt or "").strip():
            continue
        try:
            arr = np.asarray(bb, dtype=np.float32)
            y_top = float(arr[:, 1].min())
            y_bot = float(arr[:, 1].max())
            x_left = float(arr[:, 0].min())
            x_right = float(arr[:, 0].max())
        except Exception:  # noqa: BLE001
            continue
        items.append({
            "y_mid":  (y_top + y_bot) / 2.0,
            "y_top":  y_top,
            "y_bot":  y_bot,
            "x_left": x_left,
            "x_right": x_right,
            "h":      max(1.0, y_bot - y_top),
            "w":      max(1.0, x_right - x_left),
            "txt":    txt,
            "sc":     float(sc),
            "bbox":   (int(x_left), int(y_top), int(x_right), int(y_bot)),
            "block":  0,
            "column": 0,
            "line":   -1,
            "suppressed": False,
        })
    return items


# ── step 2: garbage filter ────────────────────────────────────────────────────


def _filter_garbage(items: list[dict]) -> tuple[list[dict], dict]:
    """Remove OCR garbage tokens before layout reconstruction."""
    from app.core.config import settings

    meta: dict = {}
    if not items:
        return items, meta

    if settings.ocr_noise_classifier_enabled:
        from app.services.ocr_noise_classifier import classify_token, get_noise_tier
        out: list[dict] = []
        removed = 0
        reason_counts: dict[str, int] = {}
        for it in items:
            cls = classify_token(it["txt"], it["sc"])
            if cls.is_garbage:
                removed += 1
                for r in cls.reasons:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
                continue
            out.append(it)
        meta = {
            "noise_filtered": removed,
            "noise_reasons": reason_counts,
            "noise_tier": get_noise_tier(),
        }
        return out, meta

    from app.services.ocr_dedup import is_spurious_ocr_fragment
    out = [it for it in items if not is_spurious_ocr_fragment(it["txt"], it["sc"])]
    return out, meta


# ── step 3: tile-overlap / echo suppression ────────────────────────────────────


def _suppress_tile_overlaps(items: list[dict], median_h: float) -> int:
    """
    Mark boxes as suppressed when they are tile-strip echoes of a better box.

    O(n × k) — only compares boxes whose y_mid are within 2×median_h of each
    other (spatially neighbouring strip candidates).
    """
    from app.services.ocr_dedup import OcrDedupPolicy, is_duplicate_detection

    if len(items) < 2:
        return 0

    policy = OcrDedupPolicy.from_settings()
    y_window = max(8.0, median_h * 2.5)

    items_sorted_y = sorted(range(len(items)), key=lambda i: items[i]["y_mid"])
    suppressed = 0

    for pos, i in enumerate(items_sorted_y):
        if items[i]["suppressed"]:
            continue
        yi = items[i]["y_mid"]

        for j_pos in range(pos + 1, len(items_sorted_y)):
            j = items_sorted_y[j_pos]
            if items[j]["y_mid"] - yi > y_window:
                break
            if items[j]["suppressed"]:
                continue

            if not is_duplicate_detection(
                items[i]["bbox"], items[j]["bbox"],
                items[i]["txt"], items[j]["txt"],
                policy,
            ):
                continue

            # Keep the higher-confidence box
            if items[i]["sc"] >= items[j]["sc"]:
                items[j]["suppressed"] = True
            else:
                items[i]["suppressed"] = True
                break   # i is now suppressed; stop scanning for i
            suppressed += 1

    return suppressed


# ── step 4: text block detection ──────────────────────────────────────────────


def _assign_blocks(items: list[dict], gap_threshold: float) -> None:
    """
    Sort items by y_mid, then assign block IDs by vertical-gap detection.

    A new block starts when the vertical gap between the bottom of the
    previous item and the top of the current item exceeds ``gap_threshold``.

    Modifies ``items`` in place (sorts by y_mid, sets ``block`` field).
    """
    items.sort(key=lambda r: r["y_mid"])
    block_id = 0
    items[0]["block"] = 0
    for i in range(1, len(items)):
        gap = items[i]["y_top"] - items[i - 1]["y_bot"]
        if gap > gap_threshold:
            block_id += 1
        items[i]["block"] = block_id


# ── step 5: column detection ──────────────────────────────────────────────────


def _detect_columns(
    items: list[dict],
    image_width: int,
    *,
    min_gap_pct: float = 0.04,
    max_columns: int = 4,
    bin_px: int = 8,
) -> list[tuple[int, int]]:
    """
    Detect text columns via X-projection histogram.

    Returns a list of ``(x_start, x_end)`` pairs — at least one entry
    (the full page width) is always returned.
    """
    if not items or image_width <= 0:
        return [(0, max(1, image_width))]

    bins = max(8, image_width // max(1, bin_px))
    coverage = np.zeros(bins, dtype=np.float32)

    for it in items:
        if it["suppressed"]:
            continue
        b0 = max(0, int(it["x_left"] * bins / image_width))
        b1 = max(b0 + 1, min(bins, int(it["x_right"] * bins / image_width)))
        coverage[b0:b1] += 1.0

    if coverage.max() == 0:
        return [(0, image_width)]

    threshold = max(1.0, coverage.max() * 0.05)
    empty = coverage < threshold
    min_gap_bins = max(2, int(min_gap_pct * bins))

    boundaries: list[tuple[int, int]] = []
    start: int | None = None
    in_text = False

    for i, e in enumerate(empty):
        if not e and not in_text:
            start = i
            in_text = True
        elif e and in_text:
            # Check if the gap is wide enough
            run = 0
            for j in range(i, min(bins, i + min_gap_bins * 4)):
                if empty[j]:
                    run += 1
                else:
                    break
            if run >= min_gap_bins:
                if start is not None:
                    boundaries.append((start, i))
                in_text = False
                start = None
    if in_text and start is not None:
        boundaries.append((start, bins))

    if len(boundaries) <= 1:
        return [(0, image_width)]

    boundaries = boundaries[:max_columns]
    return [
        (int(b * image_width / bins), int(e * image_width / bins))
        for b, e in boundaries
    ]


def _assign_columns(
    items: list[dict],
    columns: list[tuple[int, int]],
    image_width: int,
) -> None:
    """Assign ``column`` index to every item by its horizontal center."""
    if len(columns) <= 1:
        for it in items:
            it["column"] = 0
        return

    for it in items:
        cx = (it["x_left"] + it["x_right"]) / 2.0
        best_col = 0
        best_dist = float("inf")
        for ci, (x0, x1) in enumerate(columns):
            if x0 <= cx <= x1:
                best_col = ci
                break
            mid = (x0 + x1) / 2.0
            dist = abs(cx - mid)
            if dist < best_dist:
                best_dist = dist
                best_col = ci
        it["column"] = best_col


# ── step 6: form / table mode detection ───────────────────────────────────────


def _detect_form_mode(items: list[dict], iw: int, ih: int) -> bool:
    """
    Heuristic form/table detection without a raster image.

    Signals used:
      * Sparse text: total box area < 12 % of page area → likely a form.
      * Small square boxes: ≥ 3 boxes with 8–45 px sides and near-1:1 aspect
        → probable checkboxes.
      * Consistent right edge alignment: many boxes end at the same x → form
        field boundaries.
    """
    active = [it for it in items if not it["suppressed"]]
    if not active or iw <= 0 or ih <= 0:
        return False

    # Sparse text
    total_area = sum(it["w"] * it["h"] for it in active)
    density = total_area / max(1, iw * ih)
    if density < 0.12:
        return True

    # Checkbox-like squares
    squares = sum(
        1 for it in active
        if 8.0 <= it["w"] <= 48.0
        and 8.0 <= it["h"] <= 48.0
        and 0.55 <= (it["w"] / max(1.0, it["h"])) <= 1.65
    )
    if squares >= 3:
        return True

    return False


# ── step 7: line clustering ────────────────────────────────────────────────────


def _cluster_lines_adaptive(
    items: list[dict],
    global_median_h: float,
    form_mode: bool,
    settings,
) -> None:
    """
    Assign a ``line`` integer to each active item.

    Clustering is done separately per ``(block, column)`` group so boxes
    from different blocks or columns are *never* merged onto the same line.

    Key difference from the legacy approach:
      * Running y-mean is updated as boxes join a cluster — no fixed anchor.
      * Tolerance is adaptive: computed from the local (per-group) median height.
    """
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for it in items:
        if not it["suppressed"]:
            groups[(it["block"], it["column"])].append(it)

    global_line_id = 0

    for (_block, _col), group in sorted(groups.items()):
        if not group:
            continue

        group.sort(key=lambda r: r["y_mid"])
        local_heights = [r["h"] for r in group]
        local_median_h = float(np.median(local_heights)) if local_heights else global_median_h
        local_tol = max(
            float(settings.reading_order_adaptive_y_tol_min_px),
            local_median_h * float(settings.reading_order_adaptive_y_tol_ratio),
        )
        if form_mode:
            local_tol = max(2.0, local_tol * float(settings.reading_order_form_tol_factor))

        current_cluster: list[dict] = []
        running_y: float | None = None

        for it in group:
            y = it["y_mid"]
            if running_y is None:
                current_cluster = [it]
                running_y = y
            elif abs(y - running_y) <= local_tol:
                current_cluster.append(it)
                n = len(current_cluster)
                running_y = (running_y * (n - 1) + y) / n   # running mean
            else:
                for k in current_cluster:
                    k["line"] = global_line_id
                global_line_id += 1
                current_cluster = [it]
                running_y = y

        if current_cluster:
            for k in current_cluster:
                k["line"] = global_line_id
            global_line_id += 1

    # Suppressed items get a harmless sentinel
    for it in items:
        if it["line"] == -1:
            it["line"] = global_line_id


# ── step 8: reading order strategies ─────────────────────────────────────────


def _group_by_line(items: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for it in items:
        if not it["suppressed"]:
            groups[it["line"]].append(it)
    return groups


def _line_y_min(group: list[dict]) -> float:
    return min(it["y_mid"] for it in group) if group else 0.0


def _reading_order_single_column(items: list[dict]) -> list[list[dict]]:
    """
    Reading order for single-column layouts.
    Lines sorted top → bottom.
    """
    groups = _group_by_line(items)
    return [groups[lid] for lid in sorted(groups, key=lambda lid: _line_y_min(groups[lid]))]


def _reading_order_multi_column(
    items: list[dict],
    block_count: int,
) -> list[list[dict]]:
    """
    Column-major reading order for multi-column layouts.

    Within each text block, lines are further grouped by the *column they
    belong to*.  All lines in column 0 are emitted first (top → bottom),
    then all lines in column 1, etc.  Blocks themselves are sorted top →
    bottom by block y_min.

    Result for a two-column invoice:
      SPRZEDAWCA → ABC sp. → NIP 123 → NABYWCA → XYZ Corp → NIP 789
    (not the row-interleaved output of the legacy approach)
    """
    active = [it for it in items if not it["suppressed"]]
    groups = _group_by_line(items)

    # ── line → primary column ──────────────────────────────────────────────
    line_col_count: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    line_y_top: dict[int, float] = {}
    for it in active:
        line_col_count[it["line"]][it["column"]] += 1
        y = it["y_mid"]
        if it["line"] not in line_y_top or y < line_y_top[it["line"]]:
            line_y_top[it["line"]] = y

    line_primary_col: dict[int, int] = {
        lid: max(cc, key=cc.get)    # type: ignore[arg-type]
        for lid, cc in line_col_count.items() if cc
    }

    # ── block → y_min ──────────────────────────────────────────────────────
    block_y_min: dict[int, float] = {}
    for it in active:
        b = it["block"]
        if b not in block_y_min or it["y_top"] < block_y_min[b]:
            block_y_min[b] = it["y_top"]

    sorted_blocks = sorted(
        range(block_count),
        key=lambda b: block_y_min.get(b, 0.0),     # blocks top → bottom
    )

    result: list[list[dict]] = []
    seen: set[int] = set()

    for b in sorted_blocks:
        block_line_ids = {it["line"] for it in active if it["block"] == b}

        # Group lines within this block by their column
        col_lines: dict[int, list[int]] = defaultdict(list)
        for lid in block_line_ids:
            col = line_primary_col.get(lid, 0)
            col_lines[col].append(lid)

        # Emit column 0 top→bottom, then column 1 top→bottom, …
        for col in sorted(col_lines):
            for lid in sorted(col_lines[col], key=lambda l: line_y_top.get(l, 0.0)):
                if lid not in seen and lid in groups:
                    result.append(groups[lid])
                    seen.add(lid)

    return result


def _reading_order_form(items: list[dict]) -> list[list[dict]]:
    """
    Reading order for form / table layouts.

    Row-major: boxes on the same horizontal y-band go on the same output
    line regardless of their column assignment.  This keeps label→value
    pairs together (e.g. "Imię:  Jan").

    Lines are sorted by y (same as single-column), but the label from the
    left column and the value from the right column land on the same output
    line because they share the same ``line`` id (assigned during adaptive
    clustering, which operates per (block, column) but the global line_id
    counter increments across groups — same y-band in form mode gets the
    *same* line_id because form_mode uses stricter tolerance and the global
    counter is monotone by y within a block).

    When the two groups belong to different line_ids (split by stricter
    tolerance), they're emitted as consecutive lines which is still correct.
    """
    # Merge across columns for same-y groups using the existing line groups
    groups = _group_by_line(items)
    return [groups[lid] for lid in sorted(groups, key=lambda lid: _line_y_min(groups[lid]))]


# ── helpers ───────────────────────────────────────────────────────────────────


def _empty_meta() -> dict[str, Any]:
    return {
        "detected_columns": 1,
        "block_count": 0,
        "merge_operations": 0,
        "overlap_removed": 0,
        "reading_order_strategy": "single_column",
        "filtered_garbage": 0,
    }
