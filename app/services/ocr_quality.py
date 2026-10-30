"""Per-line OCR quality scoring and disagreement detection."""

from __future__ import annotations

import re
import unicodedata

from app.core.config import settings
from app.services.ocr_validators import find_sensitive_fields
from app.services.semantic_validators.registry import semantic_validation_failed


def normalized_levenshtein(a: str, b: str) -> float:
    """0 = identical, 1 = completely different."""
    a = (a or "").strip()
    b = (b or "").strip()
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    la, lb = len(a), len(b)
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[lb] / max(la, 1)


def char_validity_score(text: str) -> float:
    """Penalty for garbage / excessive punctuation."""
    if not text.strip():
        return 0.0
    good = 0
    bad = 0
    for ch in text:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N") or ch in ".,;:-/()+%":
            good += 1
        elif cat.startswith("C"):
            bad += 2
        else:
            bad += 1
    repeat_penalty = 0
    for m in re.finditer(r"(.)\1{4,}", text):
        repeat_penalty += len(m.group(0)) * 0.5
    total = good + bad + repeat_penalty
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, good / total))


def regex_match_score(text: str) -> float:
    """1.0 when sensitive fields validate; lower when checksum fails."""
    fields = find_sensitive_fields(text)
    if not fields:
        return 0.85
    ok = sum(1 for f in fields if f.valid)
    return ok / len(fields)


def ocr_agreement_score(rapid_text: str, tess_text: str) -> float:
    return 1.0 - normalized_levenshtein(rapid_text, tess_text)


def line_quality_score(
    *,
    mean_conf: float,
    rapid_text: str,
    tess_text: str | None = None,
) -> float:
    tess_text = tess_text or ""
    w_conf = settings.line_score_weight_conf
    w_char = settings.line_score_weight_char
    w_regex = settings.line_score_weight_regex
    w_agree = settings.line_score_weight_agreement
    agree = ocr_agreement_score(rapid_text, tess_text) if tess_text else 0.85
    return max(
        0.0,
        min(
            1.0,
            w_conf * mean_conf
            + w_char * char_validity_score(rapid_text)
            + w_regex * regex_match_score(rapid_text)
            + w_agree * agree,
        ),
    )


def classify_line_escalation(
    *,
    mean_conf: float,
    min_conf: float,
    rapid_text: str,
    tess_text: str,
    quality_score: float,
) -> tuple[str, str | None]:
    """
    Returns (tier, reason): ``none`` | ``light`` | ``hard``.

    * light — Tesseract disagreement / consensus only (szybko)
    * hard — niska pewność / checksum / pusto → crop VLM lub multi-variant
    """
    if not rapid_text.strip():
        return "hard", "empty_line"
    failed, reason = semantic_validation_failed(rapid_text)
    if failed:
        return "hard", reason
    if mean_conf < settings.ocr_line_escalate_mean_conf:
        return "hard", f"low_mean_conf ({mean_conf:.3f})"
    if min_conf < settings.ocr_line_escalate_min_conf:
        return "hard", f"low_min_conf ({min_conf:.3f})"
    if quality_score < settings.ocr_line_escalate_quality_below:
        return "hard", f"low_quality_score ({quality_score:.3f})"

    dist = normalized_levenshtein(rapid_text, tess_text) if tess_text.strip() else 0.0
    if tess_text.strip():
        if dist > settings.ocr_line_disagreement_threshold:
            if mean_conf < settings.ocr_line_escalate_mean_conf_light:
                return "hard", f"ocr_disagreement ({dist:.2f})"
            return "light", f"ocr_disagreement ({dist:.2f})"
        if (
            dist > settings.ocr_line_disagreement_light
            and mean_conf < settings.ocr_line_escalate_mean_conf_light
        ):
            return "light", f"ocr_disagreement_mild ({dist:.2f})"

    return "none", None


def line_needs_escalation(
    *,
    mean_conf: float,
    min_conf: float,
    rapid_text: str,
    tess_text: str,
    quality_score: float,
) -> tuple[bool, str | None]:
    tier, reason = classify_line_escalation(
        mean_conf=mean_conf,
        min_conf=min_conf,
        rapid_text=rapid_text,
        tess_text=tess_text,
        quality_score=quality_score,
    )
    return tier != "none", reason
