"""Consensus merge across OCR engine outputs."""

from __future__ import annotations

from collections import Counter

from app.services.ocr_quality import normalized_levenshtein
from app.services.ocr_validators import find_sensitive_fields, regex_validation_failed


def pick_best_candidate(
    candidates: list[tuple[str, float, str]],
) -> tuple[str, float, str]:
    """
    candidates: (text, quality_score, engine_name)
    Returns (text, score, engine).
    """
    if not candidates:
        return "", 0.0, "none"
    valid = [(t, s, e) for t, s, e in candidates if t.strip()]
    if not valid:
        t, s, e = candidates[0]
        return t, s, e
    best = max(valid, key=lambda x: x[1])
    return best[0], best[1], best[2]


def majority_vote_text(candidates: list[str]) -> str:
    """Simple majority for full-line strings; fallback to longest high-quality."""
    cleaned = [c.strip() for c in candidates if c and c.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    counts = Counter(cleaned)
    top = counts.most_common()
    if len(top) == 1 or top[0][1] > 1:
        return top[0][0]
    # Prefer candidate passing sensitive-field checksums
    for text in cleaned:
        failed, _ = regex_validation_failed(text)
        if not failed and find_sensitive_fields(text):
            return text
    # Close strings — pick shortest edit distance cluster center
    best = cleaned[0]
    best_avg = 999.0
    for anchor in cleaned:
        avg = sum(normalized_levenshtein(anchor, o) for o in cleaned) / len(cleaned)
        if avg < best_avg:
            best_avg = avg
            best = anchor
    return best
