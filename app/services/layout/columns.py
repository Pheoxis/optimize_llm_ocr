"""Column detection via vertical-projection histogram."""

from __future__ import annotations

import numpy as np

from app.services.layout.types import LayoutBox


def _column_boundaries(
    boxes: list[LayoutBox],
    *,
    image_width: int,
    min_gap_pct: float = 0.04,
    bin_px: int = 8,
) -> list[tuple[int, int]]:
    """Return ``[(x_start, x_end), ...]`` for detected columns.

    Heuristic: project text-box widths along X; find horizontal gaps wider
    than ``min_gap_pct`` of page width with no text → column boundary.
    """
    if not boxes or image_width <= 0:
        return [(0, max(1, image_width))]

    bins = max(8, image_width // max(1, bin_px))
    coverage = np.zeros(bins, dtype=np.float32)
    for b in boxes:
        x0 = max(0, int(b.bbox[0] * bins / image_width))
        x1 = max(x0 + 1, int(b.bbox[2] * bins / image_width))
        coverage[x0:x1] += 1

    if coverage.max() == 0:
        return [(0, image_width)]

    # Smooth + mask of "no text"
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
            run = 0
            j = i
            while j < bins and empty[j]:
                run += 1
                j += 1
            if run >= min_gap_bins:
                end = i
                if start is not None:
                    boundaries.append((start, end))
                in_text = False
                start = None
    if in_text and start is not None:
        boundaries.append((start, bins))

    if len(boundaries) <= 1:
        return [(0, image_width)]

    return [
        (int(b * image_width / bins), int(e * image_width / bins))
        for b, e in boundaries
    ]


def detect_columns(
    boxes: list[LayoutBox],
    *,
    image_width: int,
    min_columns: int = 1,
    max_columns: int = 4,
) -> list[tuple[int, int]]:
    """Detect text columns. Returns a list of ``(x_start, x_end)`` ranges.

    Always returns at least one column (the full page) so callers can
    treat single-column layouts the same way.
    """
    ranges = _column_boundaries(boxes, image_width=image_width)
    if len(ranges) > max_columns:
        ranges = ranges[:max_columns]
    if len(ranges) < min_columns:
        ranges = [(0, image_width)]
    return ranges


def assign_columns(boxes: list[LayoutBox], columns: list[tuple[int, int]]) -> None:
    """Assign each box's ``column`` index by bbox center X. In place."""
    if len(columns) <= 1:
        for b in boxes:
            b.column = 0
        return
    for b in boxes:
        cx = (b.bbox[0] + b.bbox[2]) / 2
        idx = 0
        for i, (x0, x1) in enumerate(columns):
            if x0 <= cx <= x1:
                idx = i
                break
            if cx < x0:
                idx = max(0, i - 1) if i > 0 else 0
                break
            idx = i
        b.column = idx
