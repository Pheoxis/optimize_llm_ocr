"""Barcode region detection and optional decoding.

Uses OpenCV's BarcodeDetector when available; falls back to a morphology
heuristic (dense vertical-stripe regions) when OpenCV's detector is missing.
``pyzbar`` is used opportunistically for decoding if installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BarcodeRegion:
    bbox: tuple[int, int, int, int]
    data: str | None = None
    decoder: str = "heuristic"


def _bounds_from_points(points) -> tuple[int, int, int, int]:
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _opencv_barcode_detect(image: Image.Image) -> list[BarcodeRegion]:
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return []

    detector_cls = getattr(cv2, "barcode_BarcodeDetector", None) or getattr(
        cv2, "BarcodeDetector", None,
    )
    if detector_cls is None:
        return []

    try:
        detector = detector_cls()
        gray = np.asarray(image.convert("L"))
        ok, _decoded, _types, points = detector.detectAndDecodeMulti(gray)
        if not ok or points is None:
            return []
    except Exception as exc:  # noqa: BLE001
        logger.debug("OpenCV barcode detect failed: %s", exc)
        return []

    out: list[BarcodeRegion] = []
    points = np.asarray(points)
    if points.ndim == 3:
        for i in range(points.shape[0]):
            bbox = _bounds_from_points(points[i])
            data = ""
            if isinstance(_decoded, (list, tuple)) and i < len(_decoded):
                data = _decoded[i] or ""
            out.append(BarcodeRegion(bbox=bbox, data=data or None, decoder="opencv"))
    return out


def _heuristic_barcode_detect(image: Image.Image) -> list[BarcodeRegion]:
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return []

    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return []

    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    grad = cv2.convertScaleAbs(grad_x - grad_y)
    blur = cv2.blur(grad, (9, 9))
    _, thr = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    out: list[BarcodeRegion] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 60 or ch < 12:
            continue
        aspect = cw / max(1, ch)
        if aspect < 2.5 or aspect > 25:
            continue
        if cw * ch < (w * h) * 0.004:
            continue
        out.append(
            BarcodeRegion(bbox=(int(x), int(y), int(x + cw), int(y + ch)), decoder="heuristic"),
        )
    return out


def _decode_with_pyzbar(image: Image.Image, regions: list[BarcodeRegion]) -> list[BarcodeRegion]:
    if not regions:
        return regions
    try:
        from pyzbar.pyzbar import decode  # type: ignore
    except Exception:  # noqa: BLE001
        return regions

    rgb = image.convert("RGB")
    out: list[BarcodeRegion] = []
    for r in regions:
        if r.data:
            out.append(r)
            continue
        x0, y0, x1, y1 = r.bbox
        crop = rgb.crop((max(0, x0 - 4), max(0, y0 - 4), x1 + 4, y1 + 4))
        try:
            results = decode(crop)
        except Exception:  # noqa: BLE001
            results = []
        decoded = None
        for res in results:
            try:
                decoded = res.data.decode("utf-8", errors="ignore")
                break
            except Exception:  # noqa: BLE001
                continue
        out.append(BarcodeRegion(bbox=r.bbox, data=decoded, decoder="pyzbar" if decoded else r.decoder))
    return out


def detect_barcode_regions(image: Image.Image) -> list[BarcodeRegion]:
    """Return barcode regions on the page. Never raises."""
    try:
        regions = _opencv_barcode_detect(image)
        if not regions:
            regions = _heuristic_barcode_detect(image)
        return _decode_with_pyzbar(image, regions)
    except Exception as exc:  # noqa: BLE001
        logger.debug("barcode detect failed: %s", exc)
        return []
