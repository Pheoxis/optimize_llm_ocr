"""
Document segmentation for scanner output.

A single scan page (especially from SKMBT / batch receipt scanners) can
contain multiple separate documents (receipts, invoices) placed side by
side, sometimes rotated 180° or 90°.  Treating the entire page as one OCR
target produces garbled text where lines from neighbouring receipts get
interleaved by Y position.

This module splits a rendered page into individual ``documents`` and
detects the rotation of each one independently.

Pipeline:
  1. Threshold the page (dark pixels = text/ink).
  2. Anisotropic morphological closing — small horizontally (do NOT merge
     neighbouring receipts), larger vertically (merge text lines within a
     single receipt into one blob).
  3. Connected components → candidate bounding boxes.
  4. Iterative gap-based splitting: if a candidate box has an internal
     vertical / horizontal "white corridor" wider than a threshold, split
     it.  Handles cases where step 2 was slightly too aggressive.
  5. For each region, run Tesseract OSD to detect rotation (0/90/180/270).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


# ── public types ─────────────────────────────────────────────────────────────

@dataclass
class DocumentRegion:
    """One detected document on a page (e.g. a single receipt)."""

    index: int                      # 1-based, in reading order on the page
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) on the original page
    image: Image.Image              # rotated, ready for OCR
    rotation: int                   # 0 / 90 / 180 / 270 applied
    osd_confidence: float           # orientation_conf from Tesseract OSD
    layout_hint: str = ""           # e.g. "ksef_column" for Sprzedawca|Nabywca split


# ── segmentation ─────────────────────────────────────────────────────────────

def _build_text_mask(gray: np.ndarray, *, threshold: int = 200) -> np.ndarray:
    """Binary mask: 255 where pixel is darker than `threshold` (likely text)."""
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask


def _build_layout_mask(gray: np.ndarray, *, threshold: int = 200) -> np.ndarray:
    """Wider mask for column/gutter detection (includes faint grey print)."""
    _, dark = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    soft_thr = min(245, threshold + 38)
    _, soft = cv2.threshold(gray, soft_thr, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_or(dark, soft)


def _morph_close_anisotropic(
    mask: np.ndarray, *, w: int, h: int,
    h_div: int = 70, v_div: int = 70,
    h_min: int = 15, v_min: int = 25,
    h_max: int = 32,
) -> np.ndarray:
    """Close mask with a kernel narrow horizontally, taller vertically.

    Horizontal kernel must be SMALLER than the gap between neighbouring
    receipts (typically ~60-100 px on 1654×2338 scans), so we don't merge
    them.  Vertical kernel must be LARGER than the gap between text lines
    inside a single receipt (~10-25 px), so we merge lines into one blob.
    """
    kx = min(h_max, max(h_min, w // h_div))
    ky = max(v_min, h // v_div)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _find_gaps(density: np.ndarray, *, min_gap: int, edge_margin: int) -> list[tuple[int, int]]:
    """Return list of (start, end) indices of empty stretches in `density`."""
    gaps: list[tuple[int, int]] = []
    n = len(density)
    in_gap = False
    start = 0
    for i, d in enumerate(density):
        if d == 0 and not in_gap:
            start = i
            in_gap = True
        elif d > 0 and in_gap:
            if (i - start) >= min_gap and start > edge_margin and i < n - edge_margin:
                gaps.append((start, i))
            in_gap = False
    return gaps


def _split_if_gaps(
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    *,
    min_gap_ratio: float = 0.06,
    min_side: int = 100,
    depth: int = 0,
    max_depth: int = 4,
    prefer_vertical: bool = False,
) -> list[tuple[int, int, int, int]]:
    """Split a bbox recursively along empty corridors inside the mask."""
    if depth >= max_depth:
        return [bbox]

    x0, y0, x1, y1 = bbox
    sub = mask[y0:y1, x0:x1]
    if sub.size == 0:
        return [bbox]
    h, w = sub.shape

    col_density = (sub > 0).sum(axis=0)
    row_density = (sub > 0).sum(axis=1)

    min_col_gap = max(12, int(w * min_gap_ratio))
    min_row_gap = max(18, int(h * min_gap_ratio))
    splits_x = _find_gaps(col_density, min_gap=min_col_gap, edge_margin=8)
    splits_y = _find_gaps(row_density, min_gap=min_row_gap, edge_margin=8)

    if splits_x and (prefer_vertical or not splits_y):
        boundaries = [0] + [(a + b) // 2 for a, b in splits_x] + [w]
        out: list[tuple[int, int, int, int]] = []
        for i in range(len(boundaries) - 1):
            sub_x0 = x0 + boundaries[i]
            sub_x1 = x0 + boundaries[i + 1]
            if sub_x1 - sub_x0 < min_side:
                continue
            out.extend(_split_if_gaps(
                (sub_x0, y0, sub_x1, y1), mask,
                min_gap_ratio=min_gap_ratio, min_side=min_side,
                depth=depth + 1, max_depth=max_depth,
                prefer_vertical=prefer_vertical,
            ))
        return out if out else [bbox]

    if splits_y and not prefer_vertical:
        boundaries = [0] + [(a + b) // 2 for a, b in splits_y] + [h]
        out = []
        for i in range(len(boundaries) - 1):
            sub_y0 = y0 + boundaries[i]
            sub_y1 = y0 + boundaries[i + 1]
            if sub_y1 - sub_y0 < min_side:
                continue
            out.extend(_split_if_gaps(
                (x0, sub_y0, x1, sub_y1), mask,
                min_gap_ratio=min_gap_ratio, min_side=min_side,
                depth=depth + 1, max_depth=max_depth,
            ))
        return out

    return [bbox]


def _trim_to_content(
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    *, pad: int,
) -> tuple[int, int, int, int]:
    """Trim a bbox tight around the mask content, then re-add a small pad."""
    x0, y0, x1, y1 = bbox
    sub = mask[y0:y1, x0:x1]
    if sub.size == 0:
        return bbox
    H, W = mask.shape

    cols = (sub > 0).any(axis=0)
    rows = (sub > 0).any(axis=1)
    if not cols.any() or not rows.any():
        return bbox
    x_idx = np.where(cols)[0]
    y_idx = np.where(rows)[0]
    nx0 = x0 + int(x_idx.min())
    ny0 = y0 + int(y_idx.min())
    nx1 = x0 + int(x_idx.max()) + 1
    ny1 = y0 + int(y_idx.max()) + 1
    return (
        max(0, nx0 - pad),
        max(0, ny0 - pad),
        min(W, nx1 + pad),
        min(H, ny1 + pad),
    )


def _ink_page_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight bbox around all ink on the page."""
    h, w = mask.shape
    rows = (mask > 0).any(axis=1)
    cols = (mask > 0).any(axis=0)
    if not rows.any() or not cols.any():
        return (0, 0, w, h)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)


def _detect_column_slices(
    mask: np.ndarray,
    page_w: int,
    page_h: int,
    *,
    min_col_width_ratio: float = 0.10,
    min_gap_ratio: float = 0.016,
) -> list[tuple[int, int, int, int]]:
    """Split page into vertical columns using white gutters between receipts."""
    col_density = (mask > 0).sum(axis=0).astype(np.float32)
    if page_w > 80:
        k = max(5, page_w // 80) | 1
        col_density = cv2.blur(col_density.reshape(1, -1), (1, k)).flatten()

    peak = float(col_density.max()) if col_density.size else 0.0
    empty_thresh = max(1.0, peak * 0.02)
    col_empty = (col_density <= empty_thresh).astype(np.int32)
    min_gap = max(10, int(page_w * min_gap_ratio))
    edge = max(8, int(page_w * 0.015))
    gaps = _find_gaps(col_empty, min_gap=min_gap, edge_margin=edge)
    if not gaps:
        return [(0, 0, page_w, page_h)]

    # Merge column boundaries that are very close (avoid duplicate side-by-side cols).
    merged_gaps: list[tuple[int, int]] = []
    for g in gaps:
        if merged_gaps and g[0] - merged_gaps[-1][1] < min_gap:
            merged_gaps[-1] = (merged_gaps[-1][0], g[1])
        else:
            merged_gaps.append(g)
    gaps = merged_gaps

    boundaries = [0] + [(a + b) // 2 for a, b in gaps] + [page_w]
    cols: list[tuple[int, int, int, int]] = []
    min_col_w = max(80, int(page_w * min_col_width_ratio))
    for i in range(len(boundaries) - 1):
        x0, x1 = boundaries[i], boundaries[i + 1]
        if x1 - x0 < min_col_w:
            continue
        if not (mask[:, x0:x1] > 0).any():
            continue
        cols.append((x0, 0, x1, page_h))

    if len(cols) >= 2:
        cols = sorted(cols, key=lambda c: c[0])
        merged_cols: list[tuple[int, int, int, int]] = [cols[0]]
        for col in cols[1:]:
            prev = merged_cols[-1]
            if col[0] - prev[2] < max(50, int(page_w * 0.03)):
                merged_cols[-1] = (prev[0], prev[1], col[2], prev[3])
            else:
                merged_cols.append(col)
        cols = merged_cols
    return cols if len(cols) >= 2 else [(0, 0, page_w, page_h)]


def _merge_vertical_fragments(
    boxes: list[tuple[int, int, int, int]],
    *,
    x_overlap_ratio: float = 0.50,
    max_y_gap: int = 140,
) -> list[tuple[int, int, int, int]]:
    """Merge boxes in the same column that were over-split horizontally."""
    if len(boxes) <= 1:
        return boxes
    sorted_boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    merged: list[tuple[int, int, int, int]] = []
    cur = sorted_boxes[0]
    for nxt in sorted_boxes[1:]:
        cw, nw = cur[2] - cur[0], nxt[2] - nxt[0]
        overlap = max(0, min(cur[2], nxt[2]) - max(cur[0], nxt[0]))
        min_w = max(1, min(cw, nw))
        y_gap = nxt[1] - cur[3]
        if overlap / min_w >= x_overlap_ratio and 0 <= y_gap <= max_y_gap:
            cur = (
                min(cur[0], nxt[0]),
                min(cur[1], nxt[1]),
                max(cur[2], nxt[2]),
                max(cur[3], nxt[3]),
            )
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return merged


def _filter_fragment_boxes(
    boxes: list[tuple[int, int, int, int]],
    page_w: int,
    page_h: int,
    *,
    min_area_pct: float = 0.012,
    min_width: int = 160,
    min_height: int = 130,
) -> list[tuple[int, int, int, int]]:
    """Drop tiny OCR crops (noise / partial letters between receipts)."""
    page_area = page_w * page_h
    kept: list[tuple[int, int, int, int]] = []
    for bb in boxes:
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        if bw < min_width or bh < min_height:
            continue
        if bw * bh < min_area_pct * page_area:
            continue
        kept.append(bb)
    return kept


def _components_in_roi(
    closed: np.ndarray,
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    page_w: int,
    *,
    min_area: float,
    min_side_w: int,
    min_side_h: int,
    split_gap_ratio: float,
    pad: int,
) -> list[tuple[int, int, int, int]]:
    """Connected components + gap splits inside one column/ROI."""
    x0, y0, x1, y1 = roi
    sub_closed = closed[y0:y1, x0:x1]
    sub_mask = mask[y0:y1, x0:x1]
    sh, sw = sub_closed.shape
    if sh == 0 or sw == 0:
        return []

    col_width_ratio = sw / page_w
    narrow_column = col_width_ratio < 0.48

    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        sub_closed, connectivity=8,
    )
    min_side = max(40, min(min_side_w, min_side_h) // 2)
    candidates: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area or cw < min_side or ch < min_side:
            continue
        local = (int(x), int(y), int(x + cw), int(y + ch))
        candidates.append(local)

    refined: list[tuple[int, int, int, int]] = []
    for bb in candidates:
        if narrow_column:
            # One column = keep each connected blob whole (avoid cutting one receipt
            # into head/body/tail). Stacked receipts in the same column stay as
            # separate CC components only.
            splits = [bb]
        else:
            splits = _split_if_gaps(
                bb, sub_mask,
                min_gap_ratio=split_gap_ratio,
                min_side=min_side,
            )
            wide_splits: list[tuple[int, int, int, int]] = []
            for part in splits:
                cw = part[2] - part[0]
                if cw > sw * 0.55:
                    wide_splits.extend(_split_if_gaps(
                        part, sub_mask,
                        min_gap_ratio=max(0.03, split_gap_ratio * 0.65),
                        min_side=min_side,
                        prefer_vertical=True,
                        max_depth=5,
                    ))
                else:
                    wide_splits.append(part)
            splits = wide_splits

        refined.extend(splits)

    refined = _merge_vertical_fragments(refined)

    out: list[tuple[int, int, int, int]] = []
    for bb in refined:
        gx0, gy0, gx1, gy1 = bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0
        trimmed = _trim_to_content((gx0, gy0, gx1, gy1), mask, pad=pad)
        tw, th = trimmed[2] - trimmed[0], trimmed[3] - trimmed[1]
        if tw >= min_side_w and th >= min_side_h:
            out.append(trimmed)
    return out


def _find_stack_valleys(
    row_ink: np.ndarray,
    *,
    valley_ratio: float = 0.08,
    min_valley_rows: int = 38,
    edge_margin_ratio: float = 0.07,
) -> list[tuple[int, int]]:
    """Low-ink horizontal bands between receipts stacked in one column."""
    if row_ink.size == 0:
        return []
    peak = float(row_ink.max())
    if peak < 15:
        return []
    thr = peak * valley_ratio
    bh = len(row_ink)
    edge = int(bh * edge_margin_ratio)
    valleys: list[tuple[int, int]] = []
    in_gap = False
    start = 0
    for i, v in enumerate(row_ink):
        if v < thr and not in_gap:
            start = i
            in_gap = True
        elif v >= thr and in_gap:
            if i - start >= min_valley_rows and start >= edge and i <= bh - edge:
                valleys.append((start, i))
            in_gap = False
    return valleys


def _split_box_vertical_stack(
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    page_h: int,
    page_w: int,
    *,
    pad: int = 12,
    depth: int = 0,
    max_depth: int = 4,
) -> list[tuple[int, int, int, int]]:
    """Recursively split a tall column into separate stacked receipts."""
    if depth >= max_depth:
        return [bbox]

    x0, y0, x1, y1 = bbox
    bh = y1 - y0
    bw = x1 - x0
    min_box_h = int(page_h * 0.36)
    if bh < min_box_h:
        return [bbox]

    sub = mask[y0:y1, x0:x1]
    row_ink = (sub > 0).sum(axis=1).astype(np.float32)
    valleys = _find_stack_valleys(row_ink)
    if not valleys:
        return [bbox]

    narrow = bw < page_w * 0.42
    min_part_h = int(page_h * (0.17 if narrow else 0.11))

    best_mid: int | None = None
    best_score = -1.0
    peak = float(row_ink.max())
    for v0, v1 in valleys:
        mid = (v0 + v1) // 2
        top_h, bot_h = mid, bh - mid
        if top_h < min_part_h or bot_h < min_part_h:
            continue
        gap_max = float(row_ink[v0:v1].max()) if v1 > v0 else peak
        depth_score = 1.0 - gap_max / peak
        score = depth_score * min(top_h, bot_h)
        if score > best_score:
            best_score = score
            best_mid = mid

    if best_mid is None:
        return [bbox]

    top_bb = _trim_to_content((x0, y0, x1, y0 + best_mid), mask, pad=pad)
    bot_bb = _trim_to_content((x0, y0 + best_mid, x1, y1), mask, pad=pad)
    parts: list[tuple[int, int, int, int]] = []
    for part in (top_bb, bot_bb):
        ph = part[3] - part[1]
        pw = part[2] - part[0]
        if ph >= min_part_h and pw >= 80:
            parts.append(part)

    if len(parts) < 2:
        return [bbox]

    out: list[tuple[int, int, int, int]] = []
    for part in parts:
        out.extend(
            _split_box_vertical_stack(
                part, mask, page_h, page_w, pad=pad, depth=depth + 1, max_depth=max_depth,
            )
        )
    return out


def _split_vertical_document_stacks(
    boxes: list[tuple[int, int, int, int]],
    mask: np.ndarray,
    page_h: int,
    page_w: int,
    *,
    pad: int = 12,
) -> list[tuple[int, int, int, int]]:
    """Split tall regions where multiple receipts share one column (e.g. invoice + paragon)."""
    out: list[tuple[int, int, int, int]] = []
    for box in boxes:
        parts = _split_box_vertical_stack(box, mask, page_h, page_w, pad=pad)
        if len(parts) > 1:
            logger.debug("Vertical stack: %s → %d documents", box, len(parts))
        out.extend(parts)
    return out


def _clip_boxes_below_y(
    boxes: list[tuple[int, int, int, int]],
    y_cut: int,
    *,
    margin: int = 12,
    min_height: int = 50,
) -> list[tuple[int, int, int, int]]:
    """Trim upright regions so they do not overlap a sideways band below."""
    cut = y_cut - margin
    out: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in boxes:
        if y1 <= cut:
            out.append((x0, y0, x1, y1))
        elif y0 >= y_cut:
            continue
        elif y0 < cut:
            out.append((x0, y0, x1, cut))
    return [b for b in out if b[3] - b[1] >= min_height]


def _extract_sideways_bottom_bands(
    boxes: list[tuple[int, int, int, int]],
    mask: np.ndarray,
    page_w: int,
    page_h: int,
    *,
    bottom_y_ratio: float = 0.58,
    max_band_height_ratio: float = 0.34,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Merge low/small fragments into one landscape band per sideways receipt."""
    y_thr = int(page_h * bottom_y_ratio)
    max_bh = int(page_h * max_band_height_ratio)
    bottom_frags: list[tuple[int, int, int, int]] = []
    upright: list[tuple[int, int, int, int]] = []

    for box in boxes:
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        if bw > page_w * 0.42 and bh < max_bh:
            bottom_frags.append(box)
            continue
        if box[1] >= y_thr and bh <= max_bh:
            bottom_frags.append(box)
        else:
            upright.append(box)

    sideways: list[tuple[int, int, int, int]] = []
    if bottom_frags:
        union = (
            min(b[0] for b in bottom_frags),
            min(b[1] for b in bottom_frags),
            max(b[2] for b in bottom_frags),
            max(b[3] for b in bottom_frags),
        )
        trimmed = _trim_to_content(union, mask, pad=12)
        tw, th = trimmed[2] - trimmed[0], trimmed[3] - trimmed[1]
        if tw > page_w * 0.14 and 40 <= th < max_bh:
            sideways.append(trimmed)
            upright = _clip_boxes_below_y(upright, trimmed[1])

    return sideways, upright


def _merge_stacked_regions(
    boxes: list[tuple[int, int, int, int]],
    *,
    x_overlap_ratio: float = 0.42,
    max_vertical_gap: int = 200,
) -> list[tuple[int, int, int, int]]:
    """Merge regions in the same column that belong to one tall receipt."""
    if len(boxes) <= 1:
        return boxes

    sorted_boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    merged: list[tuple[int, int, int, int]] = [sorted_boxes[0]]
    for box in sorted_boxes[1:]:
        prev = merged[-1]
        ix0 = max(prev[0], box[0])
        ix1 = min(prev[2], box[2])
        overlap_w = max(0, ix1 - ix0)
        min_w = max(1, min(prev[2] - prev[0], box[2] - box[0]))
        x_ratio = overlap_w / min_w
        # gap < 0 means vertical overlap (still same receipt strip)
        vertical_gap = box[1] - prev[3]
        if x_ratio >= x_overlap_ratio and vertical_gap < max_vertical_gap:
            merged[-1] = (
                min(prev[0], box[0]),
                min(prev[1], box[1]),
                max(prev[2], box[2]),
                max(prev[3], box[3]),
            )
        else:
            merged.append(box)
    return merged


def _sort_reading_order(
    boxes: list[tuple[int, int, int, int]], page_h: int, page_w: int,
) -> list[tuple[int, int, int, int]]:
    """Column-aware order: left columns first, top-to-bottom within each column."""
    if len(boxes) <= 1:
        return boxes
    col_gap = max(40, page_w // 12)
    sorted_x = sorted(boxes, key=lambda b: b[0])
    columns: list[list[tuple[int, int, int, int]]] = [[sorted_x[0]]]
    for box in sorted_x[1:]:
        if box[0] - columns[-1][-1][2] < col_gap:
            columns[-1].append(box)
        else:
            columns.append([box])
    band = max(1, page_h // 12)
    ordered: list[tuple[int, int, int, int]] = []
    for col in columns:
        col.sort(key=lambda r: (r[1] // band, r[0]))
        ordered.extend(col)
    return ordered


def _boxes_cover_ink(
    boxes: list[tuple[int, int, int, int]],
    ink_bbox: tuple[int, int, int, int],
) -> float:
    """Fraction of ink bbox area covered by union of region boxes."""
    ix0, iy0, ix1, iy1 = ink_bbox
    ink_area = max(1, (ix1 - ix0) * (iy1 - iy0))
    covered = np.zeros((iy1 - iy0, ix1 - ix0), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        rx0 = max(0, x0 - ix0)
        ry0 = max(0, y0 - iy0)
        rx1 = min(ix1 - ix0, x1 - ix0)
        ry1 = min(iy1 - iy0, y1 - iy0)
        if rx1 > rx0 and ry1 > ry0:
            covered[ry0:ry1, rx0:rx1] = 1
    return float((covered > 0).sum()) / ink_area


def try_ksef_two_column_boxes(
    layout_mask: np.ndarray,
    mask: np.ndarray,
    w: int,
    h: int,
    *,
    pad: int = 12,
) -> list[tuple[int, int, int, int]] | None:
    """KSeF PDF visualization: Sprzedawca | Nabywca — split at central gutter."""
    if h < 350 or w < 700:
        return None
    col_density = (layout_mask > 0).sum(axis=0).astype(np.float32)
    if col_density.size < 80:
        return None
    k = max(5, w // 100) | 1
    col_density = cv2.blur(col_density.reshape(1, -1), (1, k)).flatten()
    peak = float(col_density.max()) if col_density.size else 0.0
    if peak < 10:
        return None
    empty_thresh = max(1.0, peak * 0.03)

    mid_lo, mid_hi = int(w * 0.36), int(w * 0.64)
    segment = col_density[mid_lo:mid_hi]
    if segment.size < 30:
        return None
    gutter_rel = int(np.argmin(segment))
    gutter_x = mid_lo + gutter_rel

    # Widen gutter: low-ink run around the minimum.
    x0g = gutter_x
    while x0g > mid_lo and col_density[x0g - 1] <= empty_thresh:
        x0g -= 1
    x1g = gutter_x
    while x1g < mid_hi - 1 and col_density[x1g + 1] <= empty_thresh:
        x1g += 1
    gutter_w = x1g - x0g + 1
    if gutter_w > int(w * 0.22):
        return None
    # PDF KSeF often has a 1–3 px white line between columns — still split there.
    split_x = (x0g + x1g + 1) // 2

    left_box = (pad, pad, max(pad + 80, split_x - pad), h - pad)
    right_box = (min(w - pad - 80, split_x + pad), pad, w - pad, h - pad)
    if left_box[2] - left_box[0] < int(w * 0.22) or right_box[2] - right_box[0] < int(w * 0.22):
        return None

    left_ink = int((layout_mask[:, left_box[0]:left_box[2]] > 0).sum())
    right_ink = int((layout_mask[:, right_box[0]:right_box[2]] > 0).sum())
    min_ink = max(3000, int(w * h * 0.0025))
    if left_ink < min_ink or right_ink < min_ink:
        return None

    boxes = [
        _trim_to_content(left_box, mask, pad=pad),
        _trim_to_content(right_box, mask, pad=pad),
    ]
    if boxes[0][2] - boxes[0][0] < 120 or boxes[1][2] - boxes[1][0] < 120:
        return None
    logger.debug(
        "KSeF two-column layout at gutter x=%d..%d (%dx%d page)",
        x0g, x1g, w, h,
    )
    return boxes


def looks_like_ksef_merged_text(text: str) -> bool:
    """True when full-page OCR has BADLY interleaved a KSeF two-column layout.

    Only fires when there is evidence of true line-level column mixing —
    e.g. SPRZEDAWCA and NABYWCA content landing on the SAME output line.
    Merely having both roles present in the text is expected for any proper
    invoice and must NOT be treated as garbling.
    """
    if not text or len(text) < 100:
        return False
    upper = text.upper()
    ksef_markers = (
        "KSEF" in upper
        or "WIZUALIZACJA" in upper
        or "QR.KSEF" in upper.replace(" ", "")
        or "KSEF.MF.GOV" in upper.replace(" ", "")
    )
    if not ksef_markers:
        return False
    # Strong signal: SPRZEDAWCA and NABYWCA on the SAME output line
    # → OCR reading order mixed the two columns at the character/word level.
    same_line = any(
        "SPRZEDAWCA" in ln.upper() and "NABYWCA" in ln.upper()
        for ln in text.splitlines()
    )
    if same_line:
        return True
    # Multiple NIP labels on a single line → two parties' data merged.
    dup_nip_line = (
        sum(1 for ln in text.splitlines() if ln.upper().count("NIP") >= 2) >= 2
    )
    return dup_nip_line


def detect_document_bboxes(
    bgr: np.ndarray,
    *,
    min_area_pct: float = 0.01,
    min_side_pct: float = 0.04,
    max_regions: int = 20,
    pad: int = 15,
    split_gap_ratio: float = 0.06,
) -> tuple[list[tuple[int, int, int, int]], bool]:
    """Return (bounding boxes, whether layout is KSeF two-column split)."""
    orig_h, orig_w = bgr.shape[:2]

    # Normalize to ≈100 DPI equivalent (A4 ≈ 827 px wide) before detection.
    # At 150–300 DPI scanner-glass edges and fold shadows grow large enough to
    # be mistaken for real document regions.  Running detection on the
    # downscaled image avoids those DPI-dependent phantom boxes; bboxes are
    # then scaled back to the original resolution for cropping.
    # A4 at 100 DPI = 827 px wide; cap at 900 to stay safely below 150 DPI.
    _MAX_DETECT_W = 900
    if orig_w > _MAX_DETECT_W:
        _scale_down = _MAX_DETECT_W / orig_w
        bgr = cv2.resize(
            bgr,
            (_MAX_DETECT_W, int(orig_h * _scale_down)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        _scale_down = 1.0

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    mask = _build_text_mask(gray)
    layout_mask = _build_layout_mask(gray)

    def _scale_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if _scale_down == 1.0:
            return boxes
        inv = 1.0 / _scale_down
        return [(int(x0*inv), int(y0*inv), int(x1*inv), int(y1*inv)) for x0,y0,x1,y1 in boxes]

    ksef_cols = try_ksef_two_column_boxes(layout_mask, mask, w, h, pad=pad)
    if ksef_cols and len(ksef_cols) == 2:
        return _scale_boxes(ksef_cols), True
    closed = _morph_close_anisotropic(mask, w=w, h=h)
    ink_bbox = _ink_page_bbox(layout_mask)

    min_area = min_area_pct * w * h
    min_side_w = int(min_side_pct * w)
    min_side_h = int(min_side_pct * h)

    columns = _detect_column_slices(layout_mask, w, h)
    final: list[tuple[int, int, int, int]] = []
    for col in columns:
        final.extend(_components_in_roi(
            closed, mask, col,
            page_w=w,
            min_area=min_area,
            min_side_w=min_side_w,
            min_side_h=min_side_h,
            split_gap_ratio=split_gap_ratio,
            pad=pad,
        ))

    final = _dedup_boxes(final, iou_threshold=0.45)
    final = _drop_contained_boxes(final)
    final = _merge_stacked_regions(final)
    final = _filter_fragment_boxes(final, w, h)
    final = _split_vertical_document_stacks(final, layout_mask, h, w, pad=pad)

    # If we missed large parts of the page, retry with tighter column gutters.
    if _boxes_cover_ink(final, ink_bbox) < 0.72:
        narrow_cols = _detect_column_slices(
            layout_mask, w, h, min_col_width_ratio=0.08, min_gap_ratio=0.015,
        )
        retry: list[tuple[int, int, int, int]] = []
        for col in narrow_cols:
            retry.extend(_components_in_roi(
                closed, mask, col,
                page_w=w,
                min_area=min_area * 0.6,
                min_side_w=max(50, min_side_w // 2),
                min_side_h=max(50, min_side_h // 2),
                split_gap_ratio=max(0.03, split_gap_ratio * 0.7),
                pad=pad,
            ))
        retry = _dedup_boxes(retry, iou_threshold=0.45)
        retry = _drop_contained_boxes(retry)
        retry = _merge_stacked_regions(retry)
        retry = _filter_fragment_boxes(retry, w, h, min_area_pct=0.008)
        if _boxes_cover_ink(retry, ink_bbox) > _boxes_cover_ink(final, ink_bbox):
            final = retry
            final = _split_vertical_document_stacks(final, layout_mask, h, w, pad=pad)
            logger.debug("Segmentation retry improved ink coverage on page")

    # Single blob but ink spans most of the page width → split side-by-side receipts.
    ix0, iy0, ix1, iy1 = ink_bbox
    ink_w = ix1 - ix0
    if len(final) <= 2 and ink_w > w * 0.52:
        page_split = _split_if_gaps(
            ink_bbox, layout_mask,
            min_gap_ratio=max(0.018, split_gap_ratio * 0.45),
            min_side=min_side_w,
            prefer_vertical=True,
            max_depth=6,
        )
        if len(page_split) > len(final):
            page_boxes: list[tuple[int, int, int, int]] = []
            for bb in page_split:
                trimmed = _trim_to_content(bb, mask, pad=pad)
                tw, th = trimmed[2] - trimmed[0], trimmed[3] - trimmed[1]
                if tw >= min_side_w and th >= min_side_h:
                    page_boxes.append(trimmed)
            page_boxes = _dedup_boxes(page_boxes, iou_threshold=0.45)
            page_boxes = _drop_contained_boxes(page_boxes)
            page_boxes = _filter_fragment_boxes(page_boxes, w, h)
            if len(page_boxes) > len(final):
                final = _sort_reading_order(page_boxes, h, w)
                logger.debug(
                    "Page-wide vertical split: %d regions (was %d)",
                    len(final), len(page_split),
                )

    sideways, final = _extract_sideways_bottom_bands(final, layout_mask, w, h)
    if sideways:
        final = final + sideways
        logger.debug(
            "Sideways band(s): %d (upright regions=%d)", len(sideways), len(final) - len(sideways),
        )

    final = _sort_reading_order(final, h, w)

    if len(final) > max_regions:
        logger.warning(
            "Page segmentation produced %d regions (> max_regions=%d); "
            "keeping the largest", len(final), max_regions,
        )
        final = sorted(
            final, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True,
        )[:max_regions]
        final = _sort_reading_order(final, h, w)

    return _scale_boxes(final), False


def looks_garbled_ocr(text: str) -> bool:
    """Heuristic: OCR text is scrambled (wrong rotation / bad crop)."""
    if not text or len(text) < 80:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    upper = text.upper()
    # Duplicate table headers / columns interleaved (common after wrong 90°).
    dup_headers = sum(
        upper.count(k)
        for k in (
            "EGZEMPLARZ", "PODSUMOWANIE", "RAZEM:", "RAZEM :",
            "WARTOŚĆ", "WARTOSC", "STAWKA VAT", "KWOTA NETTO",
        )
    )
    if dup_headers >= 4:
        return True
    # Known garbage tokens from sideways / column-mixed OCR.
    garbage_hits = sum(
        1 for tok in (
            "OUNJQ", "(IZ) TEA", "MLEJSCE", "ZAKOÑCZEN", "ZAKONCZEN",
            "»L NAZWA", "UŽYTKOWE    UŽYTKOWE", "WIZUALIZACJA FAK",
        )
        if tok in upper
    )
    if garbage_hits >= 2:
        return True
    # Totals before invoice header (upside-down or 90° column mix).
    head = "\n".join(lines[:12]).upper()
    if ("WARTO" in head or "BRUTTO" in head) and "FAKTURA" in head:
        first_fakt = next(
            (i for i, ln in enumerate(lines[:15]) if "FAKTURA" in ln.upper()),
            99,
        )
        first_wart = next(
            (i for i, ln in enumerate(lines[:15]) if "WARTO" in ln.upper() or "BRUTTO" in ln.upper()),
            99,
        )
        if first_wart < first_fakt:
            return True
    short = sum(1 for ln in lines if len(ln) < 14)
    long_merged = sum(1 for ln in lines if len(ln) > 70)
    if short / len(lines) > 0.5 and long_merged >= 2:
        return True
    # Many lines with almost no letters (OCR noise).
    noise = sum(
        1 for ln in lines
        if sum(c.isalpha() for c in ln) < max(3, len(ln) // 5)
    )
    if noise / len(lines) > 0.35:
        return True
    # Two NIP numbers on the same line → seller & buyer columns merged
    nip_pat = re.compile(r"NIP[\s:.\-]*\d")
    for ln in lines:
        if len(re.findall(nip_pat, ln.upper())) >= 2:
            return True
    # Column header repeated twice on same line (e.g. "Nazwa Ilość Cena ... Nazwa Ilość Cena")
    for header_kw in ("WART.NETTO", "WART. NETTO", "WARTOŚĆ NETTO", "STAWKA VAT", "KWOTA VAT"):
        for ln in lines:
            if ln.upper().count(header_kw) >= 2:
                return True
    return False


def looks_like_merged_receipts(text: str) -> bool:
    """Heuristic: OCR text probably mixes multiple side-by-side receipts."""
    if not text or len(text) < 80:
        return False
    upper = text.upper()
    headers = (
        upper.count("PARAGON FISKALNY")
        + upper.count("FAKTURA")
        + upper.count("NIEFISKALNY")
    )
    if headers >= 3:
        return True
    if headers == 2 and "FAKTURA" in upper and "PARAGON FISKALNY" in upper:
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 12:
        return False
    short = sum(1 for ln in lines if len(ln) < 18)
    long_lines = sum(1 for ln in lines if len(ln) > 55)
    # Interleaved columns: many short fragments + a few very long merged lines
    if short / len(lines) > 0.45 and long_lines >= 2 and headers >= 1:
        return True
    return False


def _find_largest_horizontal_gap(
    mask: np.ndarray,
    *,
    min_gap_px: int = 6,
    edge_margin_ratio: float = 0.10,
) -> int | None:
    """Return y-center of the widest horizontal empty band in the mask, or None.

    Used as a last resort to split stacked regions where gap-ratio-based
    detection fails (e.g. the gap between EMPIK receipts and an upside-down
    AMIC invoice that is narrower than any standard min_gap_ratio threshold).
    """
    h = mask.shape[0]
    edge = max(5, int(h * edge_margin_ratio))
    row_density = (mask > 0).sum(axis=1)

    best_gap_size = 0
    best_gap_center = None
    in_gap = False
    gap_start = 0

    for i, d in enumerate(row_density):
        is_empty = int(d) == 0
        if is_empty and not in_gap:
            gap_start = i
            in_gap = True
        elif not is_empty and in_gap:
            gap_size = i - gap_start
            if gap_size >= min_gap_px and gap_start >= edge and i <= h - edge:
                if gap_size > best_gap_size:
                    best_gap_size = gap_size
                    best_gap_center = (gap_start + i) // 2
            in_gap = False

    return best_gap_center if best_gap_size >= min_gap_px else None


def resplit_bbox(
    page_image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    pad: int = 12,
) -> list[tuple[int, int, int, int]]:
    """Aggressively split one region that still contains multiple receipts."""
    bgr = cv2.cvtColor(np.array(page_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    layout_mask = _build_layout_mask(gray)
    mask = _build_text_mask(gray)
    x0, y0, x1, y1 = bbox
    sub_mask = layout_mask[y0:y1, x0:x1]
    h, w = sub_mask.shape[:2]
    if h < 50 or w < 50:
        return [bbox]

    min_side = max(50, min(w, h) // 8)
    local_boxes = _split_if_gaps(
        (0, 0, w, h), sub_mask,
        min_gap_ratio=0.03,
        min_side=min_side,
        prefer_vertical=True,
        max_depth=6,
    )
    if len(local_boxes) <= 1:
        local_boxes = _split_if_gaps(
            (0, 0, w, h), sub_mask,
            min_gap_ratio=0.025,
            min_side=min_side,
            max_depth=6,
        )
    # For tall single-column regions (h > w*1.5, e.g. a scan column with an
    # upright EMPIK section above an upside-down AMIC invoice), the layout mask
    # often bridges the narrow inter-document gap, but the text mask can still
    # find it.  Try a clean two-way horizontal split first before falling back
    # to complex multi-fragment gap-ratio searches on the layout mask.
    if len(local_boxes) <= 1 and h > w * 1.5:
        gray_sub_rc = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        text_sub_rc = _build_text_mask(gray_sub_rc)
        split_y_rc = _find_largest_horizontal_gap(text_sub_rc, min_gap_px=6)
        if split_y_rc is not None and split_y_rc > min_side and (h - split_y_rc) > min_side:
            local_boxes = [(0, 0, w, split_y_rc), (0, split_y_rc, w, h)]
        else:
            # Fall back to increasingly lenient gap-ratio searches on layout mask.
            for gap_ratio in (0.015, 0.010, 0.005):
                local_boxes = _split_if_gaps(
                    (0, 0, w, h), sub_mask,
                    min_gap_ratio=gap_ratio,
                    min_side=max(40, min_side // 2),
                    max_depth=4,
                )
                if len(local_boxes) >= 2:
                    break

    # Final fallback for non-tall regions: try text-mask horizontal split.
    if len(local_boxes) <= 1:
        text_sub = _build_text_mask(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
        split_y = _find_largest_horizontal_gap(text_sub, min_gap_px=6)
        if split_y is not None and split_y > min_side and (h - split_y) > min_side:
            local_boxes = [(0, 0, w, split_y), (0, split_y, w, h)]

    out: list[tuple[int, int, int, int]] = []
    for lx0, ly0, lx1, ly1 in local_boxes:
        if lx1 - lx0 < min_side or ly1 - ly0 < min_side:
            continue
        out.append(_trim_to_content(
            (x0 + lx0, y0 + ly0, x0 + lx1, y0 + ly1),
            mask, pad=pad,
        ))
    out = _dedup_boxes(out)
    return out if len(out) >= 2 else [bbox]


def _drop_contained_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Remove small boxes that lie mostly inside a larger one."""
    if len(boxes) <= 1:
        return boxes
    sorted_boxes = sorted(
        boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True,
    )
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted_boxes:
        bx0, by0, bx1, by1 = box
        ba = (bx1 - bx0) * (by1 - by0)
        contained = False
        for big in kept:
            ix0 = max(bx0, big[0])
            iy0 = max(by0, big[1])
            ix1 = min(bx1, big[2])
            iy1 = min(by1, big[3])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                if inter / max(1, ba) > 0.82:
                    contained = True
                    break
        if not contained:
            kept.append(box)
    return kept


def _dedup_boxes(
    boxes: list[tuple[int, int, int, int]], iou_threshold: float = 0.7
) -> list[tuple[int, int, int, int]]:
    """Greedily drop boxes that overlap >iou_threshold with a kept one."""
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        if all(_iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


# ── orientation detection ────────────────────────────────────────────────────

def _orientation_ink_score(gray: np.ndarray) -> float:
    """Score how 'horizontal' text bands are (high = upright Latin lines)."""
    if gray.size == 0:
        return 0.0
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_ink = (binary > 0).sum(axis=1).astype(np.float32)
    col_ink = (binary > 0).sum(axis=0).astype(np.float32)
    row_thr = max(1.0, float(row_ink.max()) * 0.12)
    col_thr = max(1.0, float(col_ink.max()) * 0.12)
    row_bands = int((row_ink > row_thr).sum())
    col_bands = int((col_ink > col_thr).sum())
    if row_bands < 3:
        return 0.0
    # Upright receipts: many horizontal text rows, not column-like layout.
    return float(row_bands) / max(1.0, col_bands) * min(row_bands, 100)


def _rotate_gray(gray: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return gray
    pil = Image.fromarray(gray).rotate(degrees, expand=True)
    return np.array(pil.convert("L"))


def _scores_0_and_180_tied(gray: np.ndarray) -> bool:
    s0 = _orientation_ink_score(_rotate_gray(gray, 0))
    s180 = _orientation_ink_score(_rotate_gray(gray, 180))
    return abs(s0 - s180) <= max(3.0, min(s0, s180) * 0.05)


def _prefer_rotation_by_reading_order(crop: Image.Image) -> int:
    """Break 0°/180° ties: FAKTURA / PARAGON should appear in the top half."""
    small = crop.copy()
    if max(small.size) > 520:
        small.thumbnail((520, 520))
    try:
        from app.services.fast_ocr import _get_engine, _run_rapid_on_image

        engine = _get_engine(0)
    except Exception as exc:
        logger.debug("Reading-order rotation probe skipped: %s", exc)
        return 0

    best_rot, best_score = 0, -1e9
    for rot in (0, 180):
        im = small.rotate(rot, expand=True) if rot else small
        boxes, txts, _ = _run_rapid_on_image(engine, im)
        if not txts:
            continue
        h = im.size[1]
        score = 0.0
        for i, line in enumerate(txts):
            lu = line.upper()
            y_mid = h / 2
            if boxes and i < len(boxes):
                y_mid = float(np.asarray(boxes[i]).mean(axis=0)[1])
            if any(k in lu for k in ("FAKTURA", "PARAGON FISKAL", "PARAGON", "SKLAD")):
                score += (h - y_mid) * 2
            elif "BR/" in lu or lu.startswith("NUMER") or lu.startswith("NUM8R"):
                score += (h - y_mid)
            if any(k in lu for k in ("ZAPLACONO", "ZAPŁACONO", "ROZLICZENIE PLATN")):
                score -= y_mid
        if score > best_score:
            best_score = score
            best_rot = rot
    return best_rot


def looks_inverted_reading_order(text: str) -> bool:
    """OCR lines read bottom-to-top (upside-down scan, rotation missed)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    if n < 4:
        return False

    # Use non-overlapping top/bottom zones so a short receipt (< 14 lines) does
    # not have the same line in both "head" and "tail".
    # Zone size: ~30% of lines, minimum 3, maximum 7.
    zone = max(3, min(7, n // 3))
    head = "\n".join(lines[:zone]).upper()
    tail = "\n".join(lines[-zone:]).upper()
    head5 = "\n".join(lines[:5]).upper()

    # Receipt/invoice opening keywords (appear at the very top of a correct scan).
    # Avoid bare "PARAGON" — it also appears in the footer phrase
    # "PARAGON STANOWI FAKTURĘ" and causes false positives.
    doc_headers = ("FAKTURA", "PARAGON FISKALNY", "PARAGON FISKAL", "SKLAD")
    # Receipt/invoice closing keywords (appear at the bottom of a correct scan).
    doc_footers = (
        "ZAPLACONO", "ZAPŁACONO", "ZAPLACNO", "ROZLICZENIE PLATN",
        "NR TRANSAKCJI", "DO ZAPLATY",
        # Receipt/invoice notice footers printed at the bottom of originals
        # but appearing at the top when a page is scanned upside-down.
        "NIEFISKALNY", "PARAGON STANOWI FAKTURE", "PARAGON STANOWI FAKT",
        "DOKUMENT NIEFISKALNY",
    )
    header_at_bottom = any(h in tail for h in doc_headers) and not any(
        h in head for h in doc_headers
    )
    # Extra guard: footer notice keywords in the TAIL means the receipt is in
    # correct orientation (footer at bottom), not inverted.
    if header_at_bottom and any(f in tail for f in ("NIEFISKALNY", "PARAGON STANOWI FAKT")):
        header_at_bottom = False
    footer_at_top = any(f in head for f in doc_footers) and not any(
        f in tail for f in doc_footers
    )
    # FAKTURA after BR/ line number (inverted invoice block)
    if "FAKTURA" in tail and "BR/" in "\n".join(lines[-12:-1]).upper():
        return True
    # Long hex/barcode string at very top + receipt header at bottom
    # (transaction hash printed at bottom of receipt, OCR reads it first when upside-down)
    first_ln = lines[0].upper() if lines else ""
    if (
        len(first_ln) >= 20
        and sum(1 for c in first_ln if c in "0123456789ABCDEF") >= len(first_ln) * 0.55
        and any(h in tail for h in doc_headers)
    ):
        return True
    # EBT / Nr dok. / Nr transakcji tokens in first 5 lines → receipt footer read first
    if any(
        tok in head5
        for tok in ("EBT ", "NR DOK.", "NR TRAN", "NR DOK ", "P/00")
    ) and any(h in tail for h in ("PARAGON", "FISKAL", "SKLAD")):
        return True
    return header_at_bottom or footer_at_top


def fix_inverted_reading_order(text: str, rotation: int = 0) -> str:
    """Reorder lines top-to-bottom when a receipt was scanned upside down."""
    if rotation == 180 or not text.strip():
        return text
    if not looks_inverted_reading_order(text):
        return text
    lines = text.split("\n")
    logger.debug("Inverted reading order detected — reversing %d lines", len(lines))
    return "\n".join(reversed(lines))


def _score_all_rotations(gray: np.ndarray) -> tuple[int, float]:
    """Pick best of 0/90/180/270 by ink-band layout score."""
    best_rot, best_score = 0, -1.0
    for rot in (0, 90, 180, 270):
        score = _orientation_ink_score(_rotate_gray(gray, rot))
        if score > best_score:
            best_score = score
            best_rot = rot
    return best_rot, best_score


def _orientation_landscape_crop(crop: Image.Image) -> tuple[int, float]:
    """Wide crop — may be sideways receipt OR upright invoice table strip."""
    w, h = crop.size
    gray = np.array(crop.convert("L"))
    # Page-wide summary band (e.g. FVS totals) — compare all four angles.
    if h >= min(400, int(w * 0.22)):
        best_rot, best_score = _score_all_rotations(gray)
        if _scores_0_and_180_tied(gray) and best_rot in (0, 180):
            probe_rot = _prefer_rotation_by_reading_order(crop)
            if probe_rot != best_rot:
                best_rot = probe_rot
                best_score = _orientation_ink_score(_rotate_gray(gray, best_rot))
        return best_rot, best_score
    # Narrow sideways strip — usually 90° or 270°.
    best_rot, best_score = 90, -1.0
    for rot in (90, 270):
        score = _orientation_ink_score(_rotate_gray(gray, rot))
        if score > best_score:
            best_score = score
            best_rot = rot
    return best_rot, best_score


def detect_orientation(
    crop: Image.Image, *, min_confidence: float = 1.5,
) -> tuple[int, float]:
    """Pick rotation (0/90/180/270) so OCR reads top-to-bottom, left-to-right.

    Uses Tesseract OSD when confident; otherwise scores ink band layout at
    each angle (fixes 90°/270° receipts lying sideways on batch scans).
    """
    w, h = crop.size
    if w > h * 1.55:
        rot, score = _orientation_landscape_crop(crop)
        logger.debug(
            "Landscape crop %dx%d → rotate %d° (layout score=%.1f)",
            w, h, rot, score,
        )
        return rot, score

    osd_rot, osd_conf = 0, 0.0
    try:
        osd = pytesseract.image_to_osd(
            crop, config="--psm 0", output_type=pytesseract.Output.DICT
        )
        osd_rot = int(osd.get("rotate", 0)) % 360
        osd_conf = float(osd.get("orientation_conf", 0.0))
    except (pytesseract.TesseractError, RuntimeError) as exc:
        logger.debug("OSD failed: %s", exc)

    if osd_conf >= min_confidence and osd_rot in (0, 90, 180, 270):
        return osd_rot, osd_conf

    gray = np.array(crop.convert("L"))
    best_rot = osd_rot if osd_conf >= min_confidence else 0
    best_score = _orientation_ink_score(_rotate_gray(gray, best_rot))

    for rot in (0, 90, 180, 270):
        score = _orientation_ink_score(_rotate_gray(gray, rot))
        if score > best_score * 1.08:
            best_score = score
            best_rot = rot

    # Tall crop (sideways document on landscape page): prefer 90/270.
    if h > w * 1.1 and best_rot in (0, 180):
        for rot in (90, 270):
            score = _orientation_ink_score(_rotate_gray(gray, rot))
            if score > best_score:
                best_score = score
                best_rot = rot

    # Portrait KSeF / invoice — do not rotate 90° unless clearly sideways strip.
    if h >= w * 0.72 and best_rot in (90, 270):
        s0 = _orientation_ink_score(_rotate_gray(gray, 0))
        if s0 >= best_score * 0.88:
            best_rot = 0
            best_score = s0

    if _scores_0_and_180_tied(gray) and best_rot in (0, 180):
        probe_rot = _prefer_rotation_by_reading_order(crop)
        if probe_rot != best_rot:
            logger.debug(
                "0°/180° tie — reading-order probe chose %d° (was %d°)",
                probe_rot, best_rot,
            )
            best_rot = probe_rot
            best_score = _orientation_ink_score(_rotate_gray(gray, best_rot))

    if best_rot != 0:
        logger.debug(
            "Orientation %d° (osd=%.1f, layout score=%.1f, size %dx%d)",
            best_rot, osd_conf, best_score, w, h,
        )
    # Keep osd_confidence separate from layout scores (layout can be >> 15).
    return best_rot, osd_conf


# ── public API ───────────────────────────────────────────────────────────────

def extract_ksef_column_regions(
    page_image: Image.Image,
    *,
    ocr_page_image: Image.Image | None = None,
) -> list[DocumentRegion] | None:
    """Return two upright column regions for KSeF visualization pages, or None."""
    bgr = cv2.cvtColor(np.array(page_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    ocr_bgr = (
        cv2.cvtColor(np.array(ocr_page_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        if ocr_page_image is not None
        else bgr
    )
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = _build_text_mask(gray)
    layout_mask = _build_layout_mask(gray)
    pad = 12
    boxes = try_ksef_two_column_boxes(layout_mask, mask, w, h, pad=pad)
    if not boxes or len(boxes) != 2:
        # Fallback: split near page centre (KSeF Sprzedawca | Nabywca).
        mid = w // 2
        boxes = [
            _trim_to_content((pad, pad, mid - pad, h - pad), mask, pad=pad),
            _trim_to_content((mid + pad, pad, w - pad, h - pad), mask, pad=pad),
        ]
        if boxes[0][2] - boxes[0][0] < 120 or boxes[1][2] - boxes[1][0] < 120:
            return None
    regions: list[DocumentRegion] = []
    for idx, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        crop_pil = Image.fromarray(
            cv2.cvtColor(ocr_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB),
        )
        regions.append(DocumentRegion(
            index=idx,
            bbox=(int(x0), int(y0), int(x1), int(y1)),
            image=crop_pil,
            rotation=0,
            osd_confidence=0.0,
            layout_hint="ksef_column",
        ))
    return regions


def extract_documents(
    page_image: Image.Image,
    *,
    ocr_page_image: Image.Image | None = None,
    enable_segmentation: bool = True,
    enable_auto_rotate: bool = True,
    min_area_pct: float = 0.01,
    min_side_pct: float = 0.04,
    max_regions: int = 20,
    osd_min_confidence: float = 1.5,
) -> list[DocumentRegion]:
    """Split a page into individual documents, each rotated upright.

    If `enable_segmentation` is False or no regions are detected, returns
    a single region covering the entire page.
    """
    bgr = cv2.cvtColor(np.array(page_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    ocr_bgr = (
        cv2.cvtColor(np.array(ocr_page_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        if ocr_page_image is not None
        else bgr
    )
    h, w = bgr.shape[:2]

    ksef_layout = False
    if enable_segmentation:
        bboxes, ksef_layout = detect_document_bboxes(
            bgr,
            min_area_pct=min_area_pct,
            min_side_pct=min_side_pct,
            max_regions=max_regions,
        )
    else:
        bboxes = []

    if not bboxes:
        # Fallback: when component-based segmentation fails, try splitting the
        # full page by large white gutters (common in scanner sheets with many
        # receipts). Only keep meaningful splits; otherwise use full page.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = _build_text_mask(gray)
        layout_mask = _build_layout_mask(gray)
        page_split = _split_if_gaps(
            (0, 0, w, h),
            layout_mask,
            min_gap_ratio=0.02,
            min_side=max(80, int(min(w, h) * 0.12)),
            prefer_vertical=True,
            max_depth=6,
        )
        retry_boxes: list[tuple[int, int, int, int]] = []
        for bb in page_split:
            trimmed = _trim_to_content(bb, mask, pad=12)
            tw, th = trimmed[2] - trimmed[0], trimmed[3] - trimmed[1]
            if tw < max(90, int(min_side_pct * w * 0.6)):
                continue
            if th < max(110, int(min_side_pct * h * 0.6)):
                continue
            retry_boxes.append(trimmed)
        retry_boxes = _dedup_boxes(retry_boxes, iou_threshold=0.45)
        retry_boxes = _drop_contained_boxes(retry_boxes)
        retry_boxes = _filter_fragment_boxes(retry_boxes, w, h, min_area_pct=max(0.002, min_area_pct * 0.35))
        if len(retry_boxes) >= 2:
            bboxes = _sort_reading_order(retry_boxes, h, w)
            logger.info("Fallback page split recovered %d regions", len(bboxes))
        else:
            bboxes = [(0, 0, w, h)]

    regions: list[DocumentRegion] = []
    for idx, (x0, y0, x1, y1) in enumerate(bboxes, start=1):
        raw_crop_pil = Image.fromarray(
            cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB),
        )
        crop_pil = Image.fromarray(
            cv2.cvtColor(ocr_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB),
        )

        rotation = 0
        osd_conf = 0.0
        layout_hint = "ksef_column" if ksef_layout else ""
        if enable_auto_rotate and not ksef_layout:
            # Orientation on raw scan — enhancement can break 0°/180° ink scores.
            rotation, osd_conf = detect_orientation(
                raw_crop_pil, min_confidence=osd_min_confidence,
            )
            if rotation:
                # Tesseract OSD `rotate` is how many degrees to rotate the
                # image counter-clockwise to make it upright; PIL.rotate(deg)
                # also rotates counter-clockwise for positive degrees.
                crop_pil = crop_pil.rotate(rotation, expand=True)

        regions.append(DocumentRegion(
            index=idx,
            bbox=(int(x0), int(y0), int(x1), int(y1)),
            image=crop_pil,
            rotation=rotation,
            osd_confidence=osd_conf,
            layout_hint=layout_hint,
        ))
    return regions
