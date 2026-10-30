"""
Enhance batch scanner pages — faint grey print → darker, OCR-friendly contrast.

Used for PDF page renders and per-receipt crops before RapidOCR.
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

Strength = Literal["auto", "mild", "strong"]


def _pil_to_gray(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def _gray_to_pil(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))


def analyze_faintness(gray: np.ndarray) -> dict[str, float | bool]:
    """Detect pages/crops where text is light grey on white paper."""
    dark_thr = 200
    soft_thr = 238
    dark_px = int((gray < dark_thr).sum())
    soft_px = int(((gray >= dark_thr) & (gray < soft_thr)).sum())
    ink_px = dark_px + soft_px
    total = max(1, gray.size)
    mean = float(gray.mean())
    std = float(gray.std())

    soft_ratio = soft_px / max(1, ink_px)
    ink_coverage = ink_px / total
    # Faint: much ink only visible at soft threshold, or low contrast bright page
    faint = (
        (soft_ratio > 0.18 and ink_px > 500)
        or (std < 48 and mean > 168 and ink_coverage > 0.02)
        or (soft_px > dark_px * 0.35 and soft_px > 2000)
    )
    return {
        "faint": faint,
        "soft_ratio": round(soft_ratio, 4),
        "ink_coverage": round(ink_coverage, 4),
        "mean_luma": round(mean, 1),
        "std_luma": round(std, 1),
    }


def _darken_faint_ink(gray: np.ndarray, *, strong: bool) -> np.ndarray:
    """Darken grey ink while preserving text coverage (no white-wash)."""
    out = gray.astype(np.float32)
    soft = (gray >= 195) & (gray < 242)
    mid = (gray >= 160) & (gray < 195)
    if soft.any():
        factor = 0.42 if strong else 0.62
        out[soft] = out[soft] * factor
    if mid.any():
        out[mid] = out[mid] * (0.72 if strong else 0.85)

    dark_u8 = np.clip(out, 0, 255).astype(np.uint8)
    # Top-hat boosts grey strokes on white paper
    k = 31 if strong else 25
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    tophat = cv2.morphologyEx(dark_u8, cv2.MORPH_TOPHAT, kernel)
    boosted = cv2.add(dark_u8, tophat)

    clip = 3.0 if strong else 2.2
    eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(boosted)
    gamma = 0.62 if strong else 0.80
    lut = (np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255).astype(np.uint8)
    return cv2.LUT(eq, lut)


def _detect_skew_angle(gray: np.ndarray) -> float:
    """Estimate document skew angle (degrees) using text-line analysis.

    Returns the rotation needed to straighten the image. Positive angle means
    the document is rotated counter-clockwise and needs a clockwise correction.
    Returns 0.0 when no reliable skew is detected.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Dilate horizontally to merge individual characters into text-line blobs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    dilated = cv2.dilate(binary, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles: list[float] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            continue
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]  # always in [-90, 0)
        w_r, h_r = rect[1]
        if w_r < h_r:
            angle = 90.0 + angle
        if abs(angle) <= 15.0:
            angles.append(angle)

    if len(angles) < 3:
        return 0.0
    return float(np.median(angles))


def deskew_image(img: Image.Image) -> tuple[Image.Image, float]:
    """Detect and correct small-angle skew (< 15°) caused by scanner tilt.

    Returns (corrected_image, detected_angle_degrees).
    The image is returned unchanged when the detected skew is below 0.5° or
    above 15° (the latter likely indicates a deliberate 90° rotation that
    ``detect_orientation`` should handle separately).
    """
    gray = np.array(img.convert("L"))
    angle = _detect_skew_angle(gray)

    if abs(angle) < 0.5:
        return img, 0.0

    h, w = gray.shape[:2]
    center = (w / 2, h / 2)
    # angle < 0 means the document is tilted clockwise — rotate counter-clockwise to fix.
    correction = -angle
    M = cv2.getRotationMatrix2D(center, correction, 1.0)
    img_arr = np.array(img.convert("RGB"))
    corrected = cv2.warpAffine(
        img_arr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug("Deskewed image by %.2f° (correction=%.2f°)", angle, correction)
    return Image.fromarray(corrected), angle


def enhance_scan_image(
    image: Image.Image,
    *,
    strength: Strength = "auto",
) -> tuple[Image.Image, dict]:
    """
    Return an RGB image optimized for OCR.

    ``mild`` — light CLAHE on every crop (safe default for batch scans).
    ``strong`` — aggressive darken for detected faint grey print.
    ``auto`` — mild always; strong when :func:`analyze_faintness` says faint.
    """
    gray = _pil_to_gray(image)
    info = analyze_faintness(gray)
    info["strength_applied"] = "none"

    if strength == "mild":
        out = _darken_faint_ink(gray, strong=False)
        info["strength_applied"] = "mild"
        return _gray_to_pil(out), info

    if strength == "strong":
        out = _darken_faint_ink(gray, strong=True)
        info["strength_applied"] = "strong"
        return _gray_to_pil(out), info

    # auto
    out = _darken_faint_ink(gray, strong=bool(info["faint"]))
    info["strength_applied"] = "strong" if info["faint"] else "mild"
    if info["faint"]:
        logger.debug(
            "Faint scan enhanced (soft_ratio=%.2f mean=%.0f std=%.0f)",
            info["soft_ratio"], info["mean_luma"], info["std_luma"],
        )
    return _gray_to_pil(out), info


def prepare_page_for_ocr(page: Image.Image) -> Image.Image:
    """Enhance a full rendered PDF page before segmentation + OCR."""
    from app.core.config import settings

    if not settings.pdf_enhance_scans:
        return page
    strength: Strength = "auto" if settings.pdf_enhance_faint_auto else "mild"
    enhanced, _ = enhance_scan_image(page, strength=strength)
    return enhanced


def prepare_crop_for_ocr(crop: Image.Image, *, force_strong: bool = False) -> Image.Image:
    """Enhance a single receipt crop before saving / OCR."""
    from app.core.config import settings

    if not settings.pdf_enhance_scans:
        return crop
    # Correct small-angle scanner tilt before enhancement so OCR reads clean rows.
    if settings.pdf_deskew_crops:
        crop, _skew = deskew_image(crop)
    strength: Strength = "strong" if force_strong else "auto"
    enhanced, _ = enhance_scan_image(crop, strength=strength)
    return enhanced
