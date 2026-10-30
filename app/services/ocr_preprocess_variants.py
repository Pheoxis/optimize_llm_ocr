"""Multi-variant preprocessing for difficult OCR crops."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def generate_crop_variants(crop: Image.Image) -> list[tuple[str, Image.Image]]:
    """Return (name, image) variants for retry OCR."""
    rgb = crop.convert("RGB")
    gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    variants: list[tuple[str, Image.Image]] = [("original", rgb)]

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    variants.append(("clahe", Image.fromarray(cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB))))

    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8,
    )
    variants.append(("adaptive_thresh", Image.fromarray(cv2.cvtColor(adapt, cv2.COLOR_GRAY2RGB))))

    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    variants.append(("sharpen", Image.fromarray(cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB))))

    den = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
    variants.append(("denoise", Image.fromarray(cv2.cvtColor(den, cv2.COLOR_GRAY2RGB))))

    w, h = rgb.size
    if max(w, h) < 1200:
        up = rgb.resize((w * 2, h * 2), Image.LANCZOS)
        variants.append(("upscale2x", up))

    return variants
