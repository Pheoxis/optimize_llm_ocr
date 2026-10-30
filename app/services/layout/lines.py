"""Robust line clustering with skew tolerance."""

from __future__ import annotations

import numpy as np

from app.services.layout.types import LayoutBox


def _box_center_y(b: LayoutBox) -> float:
    return (b.bbox[1] + b.bbox[3]) / 2.0


def _box_height(b: LayoutBox) -> float:
    return max(1.0, b.bbox[3] - b.bbox[1])


def cluster_lines(
    boxes: list[LayoutBox],
    *,
    skew_tolerance_ratio: float = 0.45,
    per_column: bool = True,
) -> int:
    """Assign each box a ``line`` index. Returns line count.

    Greedy clustering that walks boxes top-down within a column and groups
    boxes whose vertical center stays within ``skew_tolerance_ratio × median_h``
    of the current line's running mean. Tolerates moderate skew because the
    running mean drifts with the line, unlike a global y-tolerance.

    With ``per_column=True`` lines are reset per column so multi-column forms
    do not merge across columns.
    """
    if not boxes:
        return 0

    columns: dict[int, list[int]] = {}
    for i, b in enumerate(boxes):
        if b.suppressed:
            continue
        col = b.column if per_column else 0
        columns.setdefault(col, []).append(i)

    line_idx = 0
    for col in sorted(columns):
        indices = columns[col]
        indices.sort(key=lambda i: (_box_center_y(boxes[i]), boxes[i].bbox[0]))
        heights = [_box_height(boxes[i]) for i in indices]
        median_h = float(np.median(heights)) if heights else 12.0
        tol = max(4.0, median_h * skew_tolerance_ratio)

        current_line: list[int] = []
        running_y: float | None = None

        for i in indices:
            y = _box_center_y(boxes[i])
            if running_y is None:
                current_line = [i]
                running_y = y
                continue
            if abs(y - running_y) <= tol:
                current_line.append(i)
                running_y = (running_y * (len(current_line) - 1) + y) / len(current_line)
            else:
                for k in current_line:
                    boxes[k].line = line_idx
                line_idx += 1
                current_line = [i]
                running_y = y
        if current_line:
            for k in current_line:
                boxes[k].line = line_idx
            line_idx += 1
    return line_idx
