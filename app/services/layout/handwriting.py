"""Heuristic handwriting flag based on OCR box statistics.

We do **not** run a separate handwriting model; instead we flag boxes that
look like handwriting (low confidence, irregular box size, low character
density) so reading-order and downstream consumers can treat them as
low-confidence regions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.layout.types import LayoutBox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HandwritingConfig:
    max_mean_conf: float = 0.78
    min_box_height_ratio: float = 0.6  # vs median line height
    max_char_density: float = 0.06     # chars per pixel² (very rough)
    min_text_length: int = 1


def _char_density(box: LayoutBox) -> float:
    w = max(1, box.bbox[2] - box.bbox[0])
    h = max(1, box.bbox[3] - box.bbox[1])
    return len((box.text or "")) / (w * h)


def flag_handwriting_lines(
    boxes: list[LayoutBox],
    *,
    config: HandwritingConfig | None = None,
) -> int:
    """Mark suspected handwriting boxes in place. Returns count flagged."""
    if not boxes:
        return 0
    cfg = config or HandwritingConfig()

    heights = [
        max(1, b.bbox[3] - b.bbox[1])
        for b in boxes
        if not b.suppressed and b.text
    ]
    if not heights:
        return 0
    import numpy as np

    median_h = float(np.median(heights))
    flagged = 0
    for b in boxes:
        if b.suppressed or not b.text or len(b.text) < cfg.min_text_length:
            continue
        if b.score >= cfg.max_mean_conf:
            continue
        h = max(1, b.bbox[3] - b.bbox[1])
        if h < median_h * cfg.min_box_height_ratio:
            continue
        if _char_density(b) > cfg.max_char_density:
            continue
        b.is_handwriting = True
        flagged += 1
    return flagged
