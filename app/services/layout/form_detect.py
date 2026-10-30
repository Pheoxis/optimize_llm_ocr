"""Lightweight form-feature detector (checkboxes, horizontal lines, sparse layout)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PIL import Image

from app.services.layout.types import LayoutBox, LayoutFeatures

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FormDetectorConfig:
    max_side: int = 1500           # downscale longest side for cv2 ops
    h_line_min_length_ratio: float = 0.15
    h_line_min_count: int = 3
    checkbox_min_px: int = 8
    checkbox_max_px: int = 42
    checkbox_min_count: int = 2
    sparse_text_area_ratio: float = 0.18


def _downscale_gray(image: Image.Image, max_side: int):
    import numpy as np

    rgb = image.convert("L")
    w, h = rgb.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return np.asarray(rgb), rgb.size


def _count_horizontal_lines(gray, *, min_length_ratio: float) -> int:
    import cv2
    import numpy as np

    if gray.size == 0:
        return 0
    inv = cv2.bitwise_not(gray)
    _, thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    min_len = max(20, int(thr.shape[1] * min_length_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.boundingRect(c)[2] >= min_len)


def _count_checkboxes(gray, *, min_px: int, max_px: int) -> int:
    import cv2

    inv = cv2.bitwise_not(gray)
    _, thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (min_px <= w <= max_px and min_px <= h <= max_px):
            continue
        ratio = w / max(1, h)
        if ratio < 0.7 or ratio > 1.4:
            continue
        area = w * h
        cnt_area = cv2.contourArea(c)
        if cnt_area <= 0:
            continue
        # Square-ish, not too filled, not too empty
        if 0.15 <= cnt_area / area <= 0.95:
            count += 1
    return count


def _text_area_ratio(boxes: list[LayoutBox], image_size: tuple[int, int]) -> float:
    w, h = image_size
    page_area = max(1, w * h)
    text_area = sum(
        max(0, b.bbox[2] - b.bbox[0]) * max(0, b.bbox[3] - b.bbox[1])
        for b in boxes
        if not b.suppressed and not b.is_barcode
    )
    return text_area / page_area


def detect_form_features(
    image: Image.Image,
    boxes: list[LayoutBox],
    *,
    config: FormDetectorConfig | None = None,
) -> LayoutFeatures:
    """Run lightweight CV detectors and return :class:`LayoutFeatures`.

    Fails gracefully if OpenCV is missing or detection throws — falls back to
    text-only features. Never raises.
    """
    cfg = config or FormDetectorConfig()
    features = LayoutFeatures()
    features.text_area_ratio = _text_area_ratio(boxes, image.size)
    features.sparse_text = features.text_area_ratio < cfg.sparse_text_area_ratio

    try:
        gray, _ = _downscale_gray(image, cfg.max_side)
        h_lines = _count_horizontal_lines(
            gray, min_length_ratio=cfg.h_line_min_length_ratio,
        )
        checkboxes = _count_checkboxes(
            gray, min_px=cfg.checkbox_min_px, max_px=cfg.checkbox_max_px,
        )
        features.horizontal_line_count = h_lines
        features.checkbox_count = checkboxes
        features.has_checkboxes = checkboxes >= cfg.checkbox_min_count
    except Exception as exc:  # noqa: BLE001
        logger.debug("form feature detection failed: %s", exc)

    features.form_mode = (
        features.has_checkboxes
        or features.horizontal_line_count >= cfg.h_line_min_count
        or features.sparse_text
    )
    return features
