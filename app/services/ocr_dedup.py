"""
Universal OCR duplicate-detection policy (all document types).

Removes **double detections** (tile overlap, engine echo) while keeping real
form layout where the same label appears in separate columns.

Spatial rule: fuzzy / near-text merge only when box centers are close on X.
Far-apart boxes with similar text (two form columns) are never merged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.services.layout.dedup import bbox_iou
from app.services.ocr_quality import normalized_levenshtein

# Bump when dedup / noise logic changes (invalidates OCR disk cache entries).
OCR_DEDUP_CACHE_REVISION = "v7"

# Form field markers to keep (exact transcription of the printed form).
_FORM_FIELD_KEEP = frozenset({
    "zł", "zl", "zt", "gr", "pln", "…", "..", "...",
})
_AMOUNT_FRAGMENT_RE = re.compile(
    r"^\d{1,3}\s+[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{1,3}$",
)
_SHORT_ALPHA_RE = re.compile(r"^[A-Za-ząćęłńóśźżĄĆĘŁŃÓŚŹŻ]{1,2}$")

# Short tokens that are real on PL documents (do not drop).
_ALLOWED_SHORT = frozenset({
    "pl", "up", "nr", "np", "ul", "do", "od", "na", "po", "we", "i", "w", "z",
})


@dataclass(frozen=True)
class OcrDedupPolicy:
    overlap_iou: float = 0.40
    max_adjacent_gap_px: int = 5
    min_vertical_overlap: float = 0.50
    containment_ratio: float = 0.82
    # Tile echo: merge near-identical text only when centers are this close on X
    max_repeat_center_distance_px: int = 180
    max_repeat_center_distance_ratio: float = 0.08
    tile_echo_text_similarity: float = 0.90

    @classmethod
    def from_settings(cls) -> OcrDedupPolicy:
        from app.core.config import settings

        return cls(
            overlap_iou=float(settings.ocr_dedup_overlap_iou),
            max_adjacent_gap_px=int(settings.ocr_dedup_max_adjacent_gap_px),
            min_vertical_overlap=float(settings.ocr_dedup_min_vertical_overlap),
            containment_ratio=float(settings.ocr_dedup_containment_ratio),
            max_repeat_center_distance_px=int(settings.ocr_dedup_max_repeat_center_px),
            max_repeat_center_distance_ratio=float(
                settings.ocr_dedup_max_repeat_center_ratio,
            ),
            tile_echo_text_similarity=float(settings.ocr_dedup_tile_echo_similarity),
        )


def _is_form_field_marker(text: str) -> bool:
    """Dots and currency unit labels printed on PL forms — always keep."""
    t = (text or "").strip()
    if not t:
        return False
    if t.casefold() in _FORM_FIELD_KEEP:
        return True
    if t in ("…", "..", "..."):
        return True
    if len(t) <= 4 and all(c in ".…" for c in t):
        return True
    return False


def is_spurious_ocr_fragment(text: str, score: float) -> bool:
    """
    Drop unknown OCR garbage on forms — keep printed field markers.

    **Keep:** ``zł`` / ``zt``, ``gr``, dot leaders ``…`` / ``..``
    **Drop:** hallucinated fragments like ``16 Z``, ``FZ``, ``COO``

    When ``ocr_noise_classifier_enabled`` is True, delegates to the full
    multi-score noise classifier (with adaptive tier thresholds).
    """
    from app.core.config import settings

    if settings.ocr_noise_classifier_enabled:
        from app.services.ocr_noise_classifier import is_garbage_ocr
        return is_garbage_ocr(text, score)

    return _legacy_spurious_ocr_fragment(text, score)


def _legacy_spurious_ocr_fragment(text: str, score: float) -> bool:
    """Original spurious-fragment heuristic (pre noise classifier)."""
    from app.core.config import settings

    if not settings.ocr_filter_spurious_fragments:
        return False

    t = (text or "").strip()
    if not t:
        return True
    if _is_form_field_marker(t):
        return False
    if _AMOUNT_FRAGMENT_RE.match(t) and score < settings.ocr_noise_amount_max_score:
        return True
    fold = t.casefold()
    if _SHORT_ALPHA_RE.match(t):
        if fold in _ALLOWED_SHORT:
            return False
        if score < settings.ocr_noise_short_max_score:
            return True
    if (
        len(t) == 3
        and t.isalpha()
        and t.isupper()
        and score < settings.ocr_noise_low_conf_threshold
    ):
        return True
    if (
        len(t) <= settings.ocr_noise_max_short_len
        and score < settings.ocr_noise_low_conf_threshold
        and sum(c.isalpha() for c in t) <= 3
        and fold not in _ALLOWED_SHORT
    ):
        return True
    return False


def filter_detection_triplets(
    parts: Sequence[str],
    bboxes: Sequence[tuple[int, int, int, int]],
    scores: Sequence[float],
) -> tuple[list[str], list[tuple[int, int, int, int]], list[float]]:
    """Remove spurious fragments before line merge / dedup."""
    out_p: list[str] = []
    out_b: list[tuple[int, int, int, int]] = []
    out_s: list[float] = []
    for txt, bbox, sc in zip(parts, bboxes, scores):
        if is_spurious_ocr_fragment(txt, float(sc)):
            continue
        out_p.append(txt)
        out_b.append(bbox)
        out_s.append(float(sc))
    return out_p, out_b, out_s


def collapse_redundant_form_marker_lines(
    lines: list,
    separator: str,
) -> list:
    """Drop a line that repeats only ``zt`` / ``gr`` / dots already on the line above."""
    if len(lines) < 2:
        return lines
    out = [lines[0]]
    for ln in lines[1:]:
        text = (getattr(ln, "text", "") or "").strip()
        if not text:
            out.append(ln)
            continue
        parts = [
            p.strip()
            for p in text.split(separator)
            if p.strip()
        ]
        if parts and all(_is_form_field_marker(p) for p in parts):
            prev = (getattr(out[-1], "text", "") or "")
            prev_parts = {
                p.strip()
                for p in prev.split(separator)
                if p.strip()
            }
            marker_parts = {p for p in parts if _is_form_field_marker(p)}
            if marker_parts and marker_parts.issubset(prev_parts):
                continue
        out.append(ln)
    return out


def texts_exact_same(a: str, b: str) -> bool:
    """Identical text; case-insensitive, diacritics significant (ą ≠ a)."""
    return (a or "").strip().casefold() == (b or "").strip().casefold()


def texts_tile_echo_same(a: str, b: str, policy: OcrDedupPolicy) -> bool:
    """Near-identical OCR of the same glyph (tile strip overlap), not a new field."""
    if texts_exact_same(a, b):
        return True
    ac = (a or "").strip().casefold()
    bc = (b or "").strip().casefold()
    if len(ac) < 4 or len(bc) < 4:
        return False
    sim = 1.0 - normalized_levenshtein(ac, bc)
    return sim >= policy.tile_echo_text_similarity


def is_substring_echo(longer: str, shorter: str) -> bool:
    """Shorter text is a prefix echo of longer (e.g. 'adresowana do' + 'adresowana')."""
    l = (longer or "").strip().casefold()
    s = (shorter or "").strip().casefold()
    if len(s) < 4 or len(s) >= len(l):
        return False
    if not l.startswith(s):
        return False
    return (len(l) - len(s)) / len(l) < 0.55


def polish_diacritic_score(text: str) -> int:
    pl = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
    return sum(1 for c in (text or "") if c in pl)


def pick_better_text(a: str, b: str, score_a: float = 0.0, score_b: float = 0.0) -> str:
    da, db = polish_diacritic_score(a), polish_diacritic_score(b)
    if da != db:
        return a if da > db else b
    if len(b) > len(a):
        return b
    if len(a) > len(b):
        return a
    return a if score_a >= score_b else b


def center_distance_x(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    return abs((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2)


def horizontal_gap_px(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    if a[2] <= b[0]:
        return int(b[0] - a[2])
    if b[2] <= a[0]:
        return int(a[0] - b[2])
    return 0


def vertical_overlap_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    iy0 = max(a[1], b[1])
    iy1 = min(a[3], b[3])
    if iy1 <= iy0:
        return 0.0
    inter = iy1 - iy0
    ha = max(1, a[3] - a[1])
    hb = max(1, b[3] - b[1])
    return inter / min(ha, hb)


def _box_area(b: tuple[int, int, int, int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def containment_ratio(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> float:
    ix0 = max(inner[0], outer[0])
    iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2])
    iy1 = min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = _box_area(inner)
    return inter / area if area > 0 else 0.0


def _spatially_close(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    policy: OcrDedupPolicy,
) -> bool:
    if bbox_iou(bbox_a, bbox_b) >= policy.overlap_iou:
        return True
    inner, outer = (
        (bbox_a, bbox_b) if _box_area(bbox_a) <= _box_area(bbox_b) else (bbox_b, bbox_a)
    )
    if containment_ratio(inner, outer) >= policy.containment_ratio:
        return True
    gap = horizontal_gap_px(bbox_a, bbox_b)
    if gap <= policy.max_adjacent_gap_px:
        if vertical_overlap_ratio(bbox_a, bbox_b) >= policy.min_vertical_overlap:
            return True
    return False


def _centers_far_apart(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    policy: OcrDedupPolicy,
    image_width: int | None,
) -> bool:
    if image_width is None or image_width <= 0:
        return False
    limit = max(
        policy.max_repeat_center_distance_px,
        int(image_width * policy.max_repeat_center_distance_ratio),
    )
    return center_distance_x(bbox_a, bbox_b) > limit


def is_duplicate_detection(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    text_a: str,
    text_b: str,
    policy: OcrDedupPolicy | None = None,
    *,
    image_width: int | None = None,
) -> bool:
    """
    True when two detections are the same OCR artifact (safe to keep one).

    Far-apart boxes: only exact-text spatial duplicates merge (rare).
    Close boxes: exact, tile-echo (~90% char match), or prefix substring echo.
    """
    pol = policy or OcrDedupPolicy.from_settings()
    if not _spatially_close(bbox_a, bbox_b, pol):
        return False
    if _centers_far_apart(bbox_a, bbox_b, pol, image_width):
        return texts_exact_same(text_a, text_b)
    if texts_exact_same(text_a, text_b):
        return True
    if texts_tile_echo_same(text_a, text_b, pol):
        return True
    if is_substring_echo(text_a, text_b) or is_substring_echo(text_b, text_a):
        return True
    return False


def merge_line_detections(
    parts: Sequence[str],
    bboxes: Sequence[tuple[int, int, int, int]],
    scores: Sequence[float],
    separator: str,
    policy: OcrDedupPolicy | None = None,
    *,
    image_width: int | None = None,
) -> str:
    """Join left-to-right; drop geometric / tile-echo duplicates."""
    pol = policy or OcrDedupPolicy.from_settings()
    parts, bboxes, scores = filter_detection_triplets(parts, bboxes, scores)
    if not parts:
        return ""
    if len(parts) != len(bboxes):
        return separator.join(p for p in parts if p)

    kept: list[tuple[str, tuple[int, int, int, int], float]] = []
    for txt, bbox, sc in zip(parts, bboxes, scores):
        t = (txt or "").strip()
        if not t:
            continue
        merged = False
        for i, (kt, kb, ks) in enumerate(kept):
            if not is_duplicate_detection(
                kb, bbox, kt, t, pol, image_width=image_width,
            ):
                continue
            kept[i] = (pick_better_text(kt, t, ks, sc), kb, max(ks, sc))
            merged = True
            break
        if not merged:
            kept.append((t, bbox, float(sc)))

    kept.sort(key=lambda row: row[1][0])
    return separator.join(row[0] for row in kept)


def dedupe_raw_detections(
    boxes: list,
    txts: list[str],
    scores: list[float],
    policy: OcrDedupPolicy | None = None,
    *,
    image_width: int | None = None,
) -> tuple[list, list[str], list[float]]:
    """Remove duplicate raw RapidOCR boxes before line clustering."""
    pol = policy or OcrDedupPolicy.from_settings()
    if len(boxes) < 2:
        return boxes, txts, scores

    def _bbox_tuple(bb) -> tuple[int, int, int, int]:
        import numpy as np

        arr = np.asarray(bb, dtype=np.float32)
        return (
            int(arr[:, 0].min()),
            int(arr[:, 1].min()),
            int(arr[:, 0].max()),
            int(arr[:, 1].max()),
        )

    tuples = [_bbox_tuple(b) for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
    keep: list[int] = []

    for i in order:
        if is_spurious_ocr_fragment(txts[i], float(scores[i])):
            continue
        duplicate = False
        for j in keep:
            if is_duplicate_detection(
                tuples[i], tuples[j], txts[i], txts[j], pol, image_width=image_width,
            ):
                duplicate = True
                break
        if not duplicate:
            keep.append(i)

    keep.sort()
    return (
        [boxes[i] for i in keep],
        [txts[i] for i in keep],
        [scores[i] for i in keep],
    )
