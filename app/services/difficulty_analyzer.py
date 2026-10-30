"""
Lightweight OCR difficulty analyzer.

Classifies images/PDF pages before OCR pipeline selection using fast
CPU-only heuristics. No heavy ML models are used.

Targets:
  * <20 ms per image
  * <50 ms per PDF page
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── enums ─────────────────────────────────────────────────────────────────────


class DocumentType(str, Enum):
    RECEIPT = "receipt"
    INVOICE = "invoice"
    FORM = "form"
    LETTER = "letter"
    DENSE_SCAN = "dense_scan"
    PHOTO_DOCUMENT = "photo_document"
    UNKNOWN = "unknown"


class LayoutType(str, Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    FORM_GRID = "form_grid"
    RECEIPT_STRIP = "receipt_strip"
    SPARSE = "sparse"
    UNKNOWN = "unknown"


class OcrQuality(str, Enum):
    EXCELLENT = "excellent"   # ≥90
    GOOD = "good"             # 70–89
    FAIR = "fair"             # 50–69
    POOR = "poor"             # 30–49
    VERY_POOR = "very_poor"   # <30


# ── result dataclasses ────────────────────────────────────────────────────────


@dataclass
class ImageFeatures:
    """Raw feature measurements for a single raster image."""
    width: int = 0
    height: int = 0
    dpi_hint: int | None = None          # from EXIF / PIL info if present
    luma_mean: float = 0.0
    luma_std: float = 0.0
    contrast_rms: float = 0.0            # RMS contrast
    blur_laplacian_var: float = 0.0      # higher = sharper
    noise_sigma: float = 0.0
    skew_deg: float = 0.0
    text_density: float = 0.0            # fraction of dark pixels in text-band
    barcode_present: bool = False
    checkbox_count: int = 0
    hline_count: int = 0
    handwriting_ratio: float = 0.0       # 0=all print, 1=all handwritten
    is_photo: bool = False
    aspect_ratio: float = 1.0            # h/w


@dataclass
class PdfPageFeatures:
    """Raw feature measurements for a PDF page."""
    page_index: int = 0
    width_pt: float = 0.0
    height_pt: float = 0.0
    page_count: int = 1
    has_embedded_text: bool = False      # digital PDF with text layer
    image_coverage: float = 0.0          # fraction of page covered by images
    char_count: int = 0                  # embedded text chars (digital PDF)
    rotation_deg: float = 0.0
    multi_column_score: float = 0.0      # 0–1 estimate
    probe_luma_std: float | None = None  # from low-res render probe


@dataclass
class DifficultyResult:
    """Output of the difficulty analyzer for one image or PDF page."""
    difficulty_score: float                              # 0–100 (higher = harder)
    document_type: DocumentType = DocumentType.UNKNOWN
    layout_type: LayoutType = LayoutType.UNKNOWN
    expected_ocr_quality: OcrQuality = OcrQuality.GOOD
    image_features: ImageFeatures | None = None
    pdf_page_features: PdfPageFeatures | None = None
    reasons: list[str] = field(default_factory=list)    # human-readable contributors
    analysis_ms: float = 0.0                            # benchmark

    def to_dict(self) -> dict[str, Any]:
        return {
            "difficulty_score": round(self.difficulty_score, 1),
            "document_type": self.document_type.value,
            "layout_type": self.layout_type.value,
            "expected_ocr_quality": self.expected_ocr_quality.value,
            "analysis_ms": round(self.analysis_ms, 2),
            "reasons": self.reasons,
        }


# ── internal helpers ──────────────────────────────────────────────────────────


def _gray_array(img) -> np.ndarray:
    """Convert PIL Image or ndarray to uint8 grayscale ndarray."""
    from PIL import Image as _PILImage
    if isinstance(img, _PILImage.Image):
        return np.asarray(img.convert("L"), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.ndim == 3:
        # BGR or RGB → luma
        return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
    return arr.astype(np.uint8)


def _downscale(gray: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Fast downscale using numpy slice (no cv2 dependency for path-critical ops)."""
    h, w = gray.shape
    longest = max(h, w)
    if longest <= max_side:
        return gray
    scale = max_side / longest
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    # Use step-based sub-sampling — fast and sufficient for heuristics
    step_y = max(1, h // new_h)
    step_x = max(1, w // new_w)
    return gray[::step_y, ::step_x]


def _measure_blur(gray: np.ndarray) -> float:
    """Laplacian variance — proxy for focus quality. Higher = sharper."""
    try:
        import cv2
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:  # noqa: BLE001
        # Pure numpy fallback (slower but no cv2 required)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap = np.convolve(gray.flatten(), kernel.flatten(), mode="same")
        return float(np.var(lap))


def _measure_noise(gray: np.ndarray) -> float:
    """
    Estimate noise sigma via median absolute deviation of high-frequency
    detail (difference image from strong blur).
    """
    try:
        import cv2
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0)
        diff = gray.astype(np.float32) - blurred
        return float(np.median(np.abs(diff)))
    except Exception:  # noqa: BLE001
        return float(np.std(gray.astype(np.float32)) * 0.1)


def _measure_skew(gray: np.ndarray) -> float:
    """
    Estimate document skew angle in degrees using projection profile peak
    search. Fast: no Hough lines needed.
    Returns angle in [-10, 10] degrees. 0 = no skew detected.
    """
    try:
        import cv2
    except ImportError:
        return 0.0

    h, w = gray.shape
    if h < 30 or w < 30:
        return 0.0

    # Otsu threshold on downscaled image
    small = _downscale(gray, max_side=400)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cx, cy = small.shape[1] / 2, small.shape[0] / 2

    def _proj_score(angle: float) -> float:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(
            binary, M, (small.shape[1], small.shape[0]),
            flags=cv2.INTER_NEAREST, borderValue=0,
        )
        return float(np.var(rot.sum(axis=1).astype(np.float64)))

    # Coarse pass: ±10° in 2° steps (11 probes)
    best_angle = 0.0
    best_score = -1.0
    for a in range(-10, 11, 2):
        s = _proj_score(float(a))
        if s > best_score:
            best_score = s
            best_angle = float(a)

    # Fine pass: ±2° around coarse best in 0.5° steps (9 probes)
    for a_tenth in range(int((best_angle - 2) * 2), int((best_angle + 2) * 2) + 1):
        angle = a_tenth * 0.5
        s = _proj_score(angle)
        if s > best_score:
            best_score = s
            best_angle = angle

    return best_angle


def _measure_text_density(gray: np.ndarray) -> float:
    """Fraction of pixels that are 'dark' (likely ink) after Otsu threshold."""
    try:
        import cv2
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return float(binary.sum()) / (255.0 * binary.size + 1e-9)
    except Exception:  # noqa: BLE001
        threshold = float(np.mean(gray)) * 0.8
        return float(np.mean(gray < threshold))


def _measure_handwriting_ratio(gray: np.ndarray) -> float:
    """
    Very rough estimate of handwriting vs. print ratio.
    Handwriting has more irregular stroke widths → higher morphological
    residual after closing with a small rectangular kernel.
    Returns 0–1 (0 = likely print, 1 = likely handwriting).
    """
    try:
        import cv2
        small = _downscale(gray, max_side=600)
        _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if binary.sum() < 1000:
            return 0.0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        residual = closed.astype(np.int32) - opened.astype(np.int32)
        ratio = float(np.clip(residual, 0, 255).sum()) / (255.0 * binary.sum() + 1e-9)
        return float(np.clip(ratio * 4.0, 0.0, 1.0))
    except Exception:  # noqa: BLE001
        return 0.0


def _count_checkboxes(gray: np.ndarray) -> int:
    """Count square-ish blobs in the 8–45px range (checkbox heuristic)."""
    try:
        import cv2
        small = _downscale(gray, max_side=1000)
        inv = cv2.bitwise_not(small)
        _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if 8 <= cw <= 45 and 8 <= ch <= 45:
                ar = cw / (ch + 1e-6)
                if 0.6 <= ar <= 1.6:
                    count += 1
        return count
    except Exception:  # noqa: BLE001
        return 0


def _count_hlines(gray: np.ndarray) -> int:
    """Count horizontal rule lines (form separators)."""
    try:
        import cv2
        small = _downscale(gray, max_side=1000)
        inv = cv2.bitwise_not(small)
        _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        min_len = max(20, int(small.shape[1] * 0.15))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
        horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for c in contours if cv2.boundingRect(c)[2] >= min_len)
    except Exception:  # noqa: BLE001
        return 0


def _detect_barcode_heuristic(gray: np.ndarray) -> bool:
    """
    Morphology-based barcode presence heuristic.
    Dense vertical-stripe pattern → likely barcode.
    """
    try:
        import cv2
        small = _downscale(gray, max_side=600)
        inv = cv2.bitwise_not(small)
        _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Erode horizontally: removes text strokes, keeps dense vertical bars
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        eroded = cv2.erode(binary, kernel, iterations=1)
        density = float(eroded.sum()) / (255.0 * eroded.size)
        return density > 0.02
    except Exception:  # noqa: BLE001
        return False


def _estimate_multi_column(gray: np.ndarray) -> float:
    """
    Estimate probability that the page has multiple columns.
    Strategy: vertical projection profile — clear valleys in the middle
    of the page indicate column gaps.
    Returns 0–1.
    """
    try:
        import cv2
        small = _downscale(gray, max_side=600)
        _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = binary.shape
        # Consider only the middle 60% of the page height (skip header/footer)
        mid = binary[int(h * 0.2):int(h * 0.8), :]
        col_proj = mid.sum(axis=0).astype(np.float64)
        if col_proj.max() < 1:
            return 0.0
        col_proj /= col_proj.max()
        # Find valleys in the middle third of the width
        mid_lo = w // 3
        mid_hi = 2 * w // 3
        mid_proj = col_proj[mid_lo:mid_hi]
        valley_score = float(1.0 - mid_proj.mean())
        return float(np.clip(valley_score * 1.5, 0.0, 1.0))
    except Exception:  # noqa: BLE001
        return 0.0


# ── image analysis ────────────────────────────────────────────────────────────


def analyze_image(
    source,                          # PIL.Image.Image | Path | str
    *,
    fast: bool = False,              # skip slower heuristics (skew, handwriting)
) -> DifficultyResult:
    """
    Analyze a raster image and return a DifficultyResult.

    Parameters
    ----------
    source : PIL.Image | Path | str
        Image to analyze.
    fast : bool
        Skip slower heuristics (skew, handwriting). Useful when latency
        budget is very tight.
    """
    t0 = time.perf_counter()
    from PIL import Image as _PILImage

    if isinstance(source, (str, Path)):
        img = _PILImage.open(source)
    else:
        img = source

    feat = ImageFeatures()

    # Basic geometry
    feat.width, feat.height = img.size
    feat.aspect_ratio = feat.height / (feat.width + 1e-6)

    # DPI hint from EXIF / PIL info
    try:
        dpi_info = img.info.get("dpi") or img.info.get("jfif_density")
        if dpi_info and isinstance(dpi_info, (tuple, list)):
            feat.dpi_hint = int(max(dpi_info))
        elif isinstance(dpi_info, (int, float)):
            feat.dpi_hint = int(dpi_info)
    except Exception:  # noqa: BLE001
        pass

    gray = _gray_array(img)
    small = _downscale(gray, max_side=800)

    # Luminance stats
    luma_f = small.astype(np.float32)
    feat.luma_mean = float(luma_f.mean())
    feat.luma_std = float(luma_f.std())

    # Contrast (RMS contrast = std / 255)
    feat.contrast_rms = feat.luma_std / 255.0

    # Blur
    feat.blur_laplacian_var = _measure_blur(small)

    # Noise
    feat.noise_sigma = _measure_noise(small)

    # Is photo (phone photo has high luma std)
    feat.is_photo = feat.luma_std > 28.0

    # Text density
    feat.text_density = _measure_text_density(small)

    # Barcode
    feat.barcode_present = _detect_barcode_heuristic(small)

    # Checkboxes
    feat.checkbox_count = _count_checkboxes(small)

    # Horizontal lines
    feat.hline_count = _count_hlines(small)

    if not fast:
        # Skew (more expensive — projection profile search)
        feat.skew_deg = _measure_skew(small)
        # Handwriting ratio
        feat.handwriting_ratio = _measure_handwriting_ratio(small)

    elapsed = (time.perf_counter() - t0) * 1000.0

    result = _score_image(feat)
    result.image_features = feat
    result.analysis_ms = elapsed
    return result


def _score_image(feat: ImageFeatures) -> DifficultyResult:
    """Map ImageFeatures → DifficultyResult."""
    score = 0.0
    reasons: list[str] = []

    # ── resolution penalty ─────────────────────────────────────────────────
    px = feat.width * feat.height
    if px < 250_000:
        penalty = 30.0 * (1.0 - px / 250_000.0)
        score += penalty
        reasons.append(f"low_resolution({feat.width}x{feat.height})")
    elif px > 25_000_000:
        score += 5.0
        reasons.append("very_high_resolution")

    # ── contrast ───────────────────────────────────────────────────────────
    if feat.contrast_rms < 0.12:
        penalty = 20.0 * (1.0 - feat.contrast_rms / 0.12)
        score += penalty
        reasons.append(f"low_contrast(rms={feat.contrast_rms:.3f})")

    # ── blur ───────────────────────────────────────────────────────────────
    if feat.blur_laplacian_var < 50:
        penalty = 25.0 * (1.0 - feat.blur_laplacian_var / 50.0)
        score += penalty
        reasons.append(f"blurry(lap_var={feat.blur_laplacian_var:.1f})")
    elif feat.blur_laplacian_var < 150:
        score += 8.0 * (1.0 - feat.blur_laplacian_var / 150.0)

    # ── noise ──────────────────────────────────────────────────────────────
    if feat.noise_sigma > 8.0:
        penalty = min(15.0, (feat.noise_sigma - 8.0) * 1.2)
        score += penalty
        reasons.append(f"noisy(sigma={feat.noise_sigma:.1f})")

    # ── skew ───────────────────────────────────────────────────────────────
    skew_abs = abs(feat.skew_deg)
    if skew_abs > 2.0:
        penalty = min(15.0, (skew_abs - 2.0) * 3.0)
        score += penalty
        reasons.append(f"skewed({feat.skew_deg:.1f}°)")

    # ── text density ───────────────────────────────────────────────────────
    if feat.text_density < 0.01:
        score += 10.0
        reasons.append("very_low_text_density")
    elif feat.text_density > 0.50:
        score += 8.0
        reasons.append("very_high_text_density")

    # ── photo ──────────────────────────────────────────────────────────────
    if feat.is_photo:
        score += 10.0
        reasons.append("photo_capture")

    # ── barcode ────────────────────────────────────────────────────────────
    if feat.barcode_present:
        score += 5.0
        reasons.append("barcode_detected")

    # ── handwriting ────────────────────────────────────────────────────────
    if feat.handwriting_ratio > 0.4:
        penalty = feat.handwriting_ratio * 20.0
        score += penalty
        reasons.append(f"handwriting(ratio={feat.handwriting_ratio:.2f})")
    elif feat.handwriting_ratio > 0.15:
        score += feat.handwriting_ratio * 10.0

    score = float(np.clip(score, 0.0, 100.0))

    # ── document type ──────────────────────────────────────────────────────
    doc_type = _classify_image_doc_type(feat)
    layout_type = _classify_image_layout(feat)
    quality = _quality_from_score(score)

    return DifficultyResult(
        difficulty_score=score,
        document_type=doc_type,
        layout_type=layout_type,
        expected_ocr_quality=quality,
        reasons=reasons,
    )


def _classify_image_doc_type(feat: ImageFeatures) -> DocumentType:
    if feat.is_photo and feat.aspect_ratio > 1.8:
        return DocumentType.PHOTO_DOCUMENT
    if feat.aspect_ratio > 1.6 and feat.text_density > 0.04:
        return DocumentType.RECEIPT
    if feat.checkbox_count >= 4 or feat.hline_count >= 5:
        return DocumentType.FORM
    if feat.text_density > 0.08 and not feat.is_photo:
        return DocumentType.DENSE_SCAN
    if feat.text_density > 0.03 and feat.aspect_ratio < 1.5:
        return DocumentType.LETTER
    return DocumentType.UNKNOWN


def _classify_image_layout(feat: ImageFeatures) -> LayoutType:
    if feat.aspect_ratio > 1.8 and feat.text_density > 0.04:
        return LayoutType.RECEIPT_STRIP
    if feat.checkbox_count >= 4 or feat.hline_count >= 6:
        return LayoutType.FORM_GRID
    if feat.text_density < 0.02:
        return LayoutType.SPARSE
    return LayoutType.SINGLE_COLUMN


def _quality_from_score(score: float) -> OcrQuality:
    if score < 10:
        return OcrQuality.EXCELLENT
    if score < 25:
        return OcrQuality.GOOD
    if score < 45:
        return OcrQuality.FAIR
    if score < 65:
        return OcrQuality.POOR
    return OcrQuality.VERY_POOR


# ── PDF page analysis ─────────────────────────────────────────────────────────


def analyze_pdf_page(
    pdf_path: Path | str,
    page_index: int = 0,
    *,
    probe_dpi: int = 72,
    fast: bool = False,
) -> DifficultyResult:
    """
    Analyze a single PDF page using a low-res probe render.

    Parameters
    ----------
    pdf_path : Path | str
        Path to the PDF file.
    page_index : int
        0-based page index.
    probe_dpi : int
        DPI for the probe render (low DPI → fast). Default 72.
    fast : bool
        Skip image-level heuristics (skew, handwriting, checkboxes).
    """
    t0 = time.perf_counter()
    pdf_path = Path(pdf_path)

    feat = PdfPageFeatures(page_index=page_index)

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        feat.page_count = doc.page_count
        page = doc[page_index]

        # Geometry
        rect = page.rect
        feat.width_pt = float(rect.width)
        feat.height_pt = float(rect.height)
        feat.rotation_deg = float(page.rotation)

        # Embedded text (digital PDF check)
        text_blocks = page.get_text("blocks")
        char_count = sum(len(b[4]) for b in text_blocks if isinstance(b[4], str))
        feat.char_count = char_count
        feat.has_embedded_text = char_count > 30

        # Image coverage
        image_list = page.get_images(full=True)
        total_img_area = 0.0
        page_area = feat.width_pt * feat.height_pt
        for img_info in image_list[:8]:  # limit to avoid slow pages
            try:
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                for r in rects:
                    total_img_area += r.width * r.height
            except Exception:  # noqa: BLE001
                pass
        feat.image_coverage = float(np.clip(total_img_area / (page_area + 1e-6), 0.0, 1.0))

        doc.close()

    except Exception as exc:  # noqa: BLE001
        logger.debug("PDF metadata extraction failed for %s[%d]: %s", pdf_path, page_index, exc)

    # Low-res probe render for image heuristics
    img_result: DifficultyResult | None = None
    try:
        import fitz
        from PIL import Image as _PILImage
        import io

        doc = fitz.open(pdf_path)
        page = doc[page_index]
        mat = fitz.Matrix(probe_dpi / 72.0, probe_dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        doc.close()
        png_bytes = pix.tobytes("png")
        probe_img = _PILImage.open(io.BytesIO(png_bytes))
        img_result = analyze_image(probe_img, fast=fast)
        feat.probe_luma_std = img_result.image_features.luma_std if img_result.image_features else None

        # Multi-column estimate from probe
        gray = _gray_array(probe_img)
        feat.multi_column_score = _estimate_multi_column(gray)

    except Exception as exc:  # noqa: BLE001
        logger.debug("PDF probe render failed for %s[%d]: %s", pdf_path, page_index, exc)

    elapsed = (time.perf_counter() - t0) * 1000.0

    result = _score_pdf_page(feat, img_result)
    result.pdf_page_features = feat
    result.analysis_ms = elapsed
    return result


def _score_pdf_page(feat: PdfPageFeatures, img_result: DifficultyResult | None) -> DifficultyResult:
    """Combine PDF metadata features with probe image analysis."""
    score = 0.0
    reasons: list[str] = []

    # If we have image-level analysis, start from that score
    if img_result is not None:
        score = img_result.difficulty_score * 0.8  # weight probe analysis
        reasons.extend(img_result.reasons)

    # ── rotation penalty ───────────────────────────────────────────────────
    rot = feat.rotation_deg % 360
    if rot not in (0, 360):
        # 90°/270° = sideways; 180° = upside down
        penalty = 20.0 if rot in (90, 270) else (15.0 if rot == 180 else 10.0)
        score += penalty
        reasons.append(f"page_rotated({rot}°)")

    # ── digital PDF bonus ──────────────────────────────────────────────────
    if feat.has_embedded_text:
        # Digital PDF with text layer — much easier for extraction
        score = max(0.0, score - 30.0)
        reasons.append("digital_pdf_text_layer")
    elif feat.image_coverage > 0.7:
        # Scanned image-only PDF
        score = min(100.0, score + 10.0)
        reasons.append("scanned_image_pdf")

    # ── multi-column ───────────────────────────────────────────────────────
    if feat.multi_column_score > 0.6:
        score += 8.0
        reasons.append(f"multi_column(score={feat.multi_column_score:.2f})")

    # ── page count (batch complexity) ─────────────────────────────────────
    if feat.page_count > 10:
        score = min(100.0, score + 5.0)
        reasons.append(f"large_pdf({feat.page_count}_pages)")

    score = float(np.clip(score, 0.0, 100.0))

    # Merge document/layout type from image probe if available
    if img_result is not None:
        doc_type = _refine_pdf_doc_type(feat, img_result.document_type)
        layout_type = _refine_pdf_layout(feat, img_result.layout_type)
    else:
        doc_type = _pdf_doc_type_from_meta(feat)
        layout_type = LayoutType.UNKNOWN

    quality = _quality_from_score(score)

    return DifficultyResult(
        difficulty_score=score,
        document_type=doc_type,
        layout_type=layout_type,
        expected_ocr_quality=quality,
        reasons=reasons,
    )


def _refine_pdf_doc_type(feat: PdfPageFeatures, img_type: DocumentType) -> DocumentType:
    if feat.has_embedded_text:
        # Digital PDF is almost certainly invoice or letter
        area_sq_in = (feat.width_pt / 72) * (feat.height_pt / 72)
        if area_sq_in < 50:
            return DocumentType.RECEIPT
        return DocumentType.INVOICE
    if feat.image_coverage > 0.8 and img_type == DocumentType.UNKNOWN:
        return DocumentType.DENSE_SCAN
    return img_type


def _pdf_doc_type_from_meta(feat: PdfPageFeatures) -> DocumentType:
    if feat.has_embedded_text:
        return DocumentType.INVOICE
    if feat.image_coverage > 0.8:
        return DocumentType.DENSE_SCAN
    return DocumentType.UNKNOWN


def _refine_pdf_layout(feat: PdfPageFeatures, img_layout: LayoutType) -> LayoutType:
    if feat.multi_column_score > 0.6:
        return LayoutType.MULTI_COLUMN
    return img_layout


# ── batch API ─────────────────────────────────────────────────────────────────


def analyze_images_batch(
    sources: list,
    *,
    fast: bool = True,
) -> list[DifficultyResult]:
    """Analyze multiple images sequentially (no heavy ML, CPU-only)."""
    return [analyze_image(src, fast=fast) for src in sources]


def analyze_pdf_pages(
    pdf_path: Path | str,
    *,
    probe_dpi: int = 72,
    fast: bool = True,
    max_pages: int | None = None,
) -> list[DifficultyResult]:
    """Analyze all pages of a PDF."""
    pdf_path = Path(pdf_path)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        n = doc.page_count
        doc.close()
    except Exception:  # noqa: BLE001
        return []

    n = n if max_pages is None else min(n, max_pages)
    return [
        analyze_pdf_page(pdf_path, i, probe_dpi=probe_dpi, fast=fast)
        for i in range(n)
    ]


# ── benchmark ─────────────────────────────────────────────────────────────────


def benchmark(
    image_path: Path | str | None = None,
    pdf_path: Path | str | None = None,
    *,
    n_image: int = 20,
    n_pdf_page: int = 10,
) -> dict[str, Any]:
    """
    Run a timing benchmark and return statistics.

    Returns
    -------
    dict with keys:
        image_mean_ms, image_max_ms, image_min_ms (if image_path given)
        pdf_mean_ms, pdf_max_ms, pdf_min_ms       (if pdf_path given)
    """
    import statistics

    results: dict[str, Any] = {}

    if image_path is not None:
        from PIL import Image as _PILImage
        img = _PILImage.open(image_path)
        timings = []
        for _ in range(n_image):
            r = analyze_image(img, fast=False)
            timings.append(r.analysis_ms)
        results["image_samples"] = n_image
        results["image_mean_ms"] = round(statistics.mean(timings), 2)
        results["image_min_ms"] = round(min(timings), 2)
        results["image_max_ms"] = round(max(timings), 2)
        results["image_p95_ms"] = round(sorted(timings)[int(0.95 * len(timings))], 2)
        logger.info(
            "DifficultyAnalyzer image benchmark: mean=%.1f ms min=%.1f ms max=%.1f ms",
            results["image_mean_ms"],
            results["image_min_ms"],
            results["image_max_ms"],
        )

    if pdf_path is not None:
        timings = []
        for i in range(n_pdf_page):
            try:
                r = analyze_pdf_page(pdf_path, page_index=0, fast=False)
                timings.append(r.analysis_ms)
            except Exception:  # noqa: BLE001
                break
        if timings:
            results["pdf_samples"] = len(timings)
            results["pdf_mean_ms"] = round(statistics.mean(timings), 2)
            results["pdf_min_ms"] = round(min(timings), 2)
            results["pdf_max_ms"] = round(max(timings), 2)
            results["pdf_p95_ms"] = round(sorted(timings)[int(0.95 * len(timings))], 2)
            logger.info(
                "DifficultyAnalyzer PDF benchmark: mean=%.1f ms min=%.1f ms max=%.1f ms",
                results["pdf_mean_ms"],
                results["pdf_min_ms"],
                results["pdf_max_ms"],
            )

    return results
