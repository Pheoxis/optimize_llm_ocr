"""
Image preprocessing before OCR.

Photo pipeline  → auto-crop document → deskew → adaptive threshold → upscale
Scan pipeline   → light normalisation only
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# ── pil ↔ cv2 ────────────────────────────────────────────────────────────────

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# ── detection ─────────────────────────────────────────────────────────────────

def is_photo(img: Image.Image) -> bool:
    """Phone photo has high luminance variance; clean scans are low."""
    arr = np.array(img.convert("L"), dtype=np.float32)
    return float(arr.std()) > 28.0


def is_receipt_like(img: Image.Image) -> bool:
    """Tall narrow image — typical phone photo of a receipt."""
    w, h = img.size
    return w > 0 and h / w >= 1.25 and h >= 1000


# ── document region extraction ────────────────────────────────────────────────

def find_document_bbox(gray: np.ndarray, min_coverage: float = 0.10) -> tuple[int, int, int, int] | None:
    """
    Find the bounding box of the document (white paper) in the photo.
    Returns (x1, y1, x2, y2) or None if detection fails.
    """
    h, w = gray.shape

    # Blur to suppress text noise, then threshold for bright (paper) regions
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    _, thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)

    # Morphological close: fill the receipt area including dark text on paper
    kernel = np.ones((40, 40), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour = document
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < min_coverage * w * h:
        return None  # nothing big enough found

    x, y, bw, bh = cv2.boundingRect(cnt)
    margin = 12
    return (
        max(0, x - margin),
        max(0, y - margin),
        min(w, x + bw + margin),
        min(h, y + bh + margin),
    )


def crop_document(gray: np.ndarray, color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop to detected document area. Returns (gray_crop, color_crop)."""
    bbox = find_document_bbox(gray)
    if bbox is None:
        return gray, color
    x1, y1, x2, y2 = bbox
    return gray[y1:y2, x1:x2], color[y1:y2, x1:x2]


# ── deskew ────────────────────────────────────────────────────────────────────

def deskew(gray: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """Correct small rotation (≤ max_angle degrees)."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 200:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    angle = -float(angle)
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REPLICATE)


# ── pipelines ─────────────────────────────────────────────────────────────────

def prepare_photo(img: Image.Image, scale: float = 1.5) -> Image.Image:
    """
    Phone photo → clean binary image:
    greyscale → crop document → deskew → denoise → adaptive threshold → upscale
    """
    arr = pil_to_cv(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    # Auto-crop to remove background (table, hands, etc.)
    gray, arr = crop_document(gray, arr)

    gray = deskew(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=5)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51,
        C=11,
    )

    if scale != 1.0:
        h, w = binary.shape
        binary = cv2.resize(binary, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_LANCZOS4)

    return cv_to_pil(binary)


def prepare_scan(img: Image.Image) -> Image.Image:
    """Clean scan → light normalisation only."""
    arr = pil_to_cv(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv_to_pil(gray)


def prepare_for_ocr(
    img: Image.Image,
    *,
    scale: float = 1.0,
    force_photo: bool | None = None,
) -> Image.Image:
    """Auto-detect photo vs scan and apply the appropriate pipeline."""
    use_photo = force_photo if force_photo is not None else is_photo(img)
    if use_photo:
        return prepare_photo(img, scale=scale)
    return prepare_scan(img)
