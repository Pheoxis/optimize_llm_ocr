"""
OCR noise classifier — detect garbage tokens BEFORE line merge.

Runs on individual OCR detection boxes (text + confidence) and returns
multi-dimensional scores plus a garbage / keep decision.

Categories detected
-------------------
  * barcode garbage       — bar-like symbol runs, vertical-stroke noise
  * malformed words       — impossible consonant clusters, broken morphology
  * impossible PL words   — no Polish vowels / invalid diacritic patterns
  * impossible EN words   — no vowels in long Latin tokens
  * OCR hallucinations    — low confidence + invalid character mix
  * repeated garbage      — ``[?]`` runs, repeated single-char tokens
  * tile overlap artifacts — ultra-short fragments with symbol pollution
  * symbol pollution      — excessive punctuation / control characters

Scores (each in [0, 1])
-----------------------
  noise_score       — higher = more likely garbage
  language_score    — higher = looks like valid PL/EN text
  structure_score   — higher = matches document structure (dates, IDs, amounts)
  dictionary_score  — higher = tokens found in lightweight PL/EN lexicon

Protected (never dropped)
-------------------------
  PESEL, NIP, REGON, IBAN, invoice IDs, tracking numbers, form markers
  (``zł``, ``gr``, dot leaders), and short allowed tokens (``nr``, ``ul``).

Adaptive thresholds
-------------------
  EASY   — stricter filtering (lower noise threshold → more garbage removed)
  MEDIUM — default
  HARD   — softer filtering (harder docs may produce odd but real tokens)

Usage
-----
    from app.services.ocr_noise_classifier import classify_token, is_garbage_ocr

    result = classify_token("16 Z", score=0.70)
    if result.is_garbage:
        ...  # drop before merge

    # Batch filter (drop-in for filter_detection_triplets)
    parts, bboxes, scores = filter_noisy_detections(parts, bboxes, scores)
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ── thread-local difficulty tier (set by pipeline before OCR) ─────────────────

_tier_state = threading.local()


def set_noise_tier(tier: str | None) -> None:
    """Set current document difficulty tier for adaptive thresholds."""
    _tier_state.tier = (tier or "medium").casefold()


def get_noise_tier() -> str:
    return getattr(_tier_state, "tier", None) or "medium"


def clear_noise_tier() -> None:
    if hasattr(_tier_state, "tier"):
        del _tier_state.tier


# ── enums / dataclasses ─────────────────────────────────────────────────────


class NoiseReason(str, Enum):
    BARCODE_GARBAGE = "barcode_garbage"
    MALFORMED_WORD = "malformed_word"
    IMPOSSIBLE_PL = "impossible_pl_word"
    IMPOSSIBLE_EN = "impossible_en_word"
    HALLUCINATION = "ocr_hallucination"
    REPEATED_GARBAGE = "repeated_garbage"
    TILE_ARTIFACT = "tile_overlap_artifact"
    SYMBOL_POLLUTION = "symbol_pollution"
    LOW_CONF_FRAGMENT = "low_conf_fragment"
    EMPTY = "empty"


@dataclass
class NoiseClassification:
    """Per-token noise analysis result."""

    text: str
    ocr_score: float
    is_garbage: bool
    noise_score: float          # 0–1, higher = more garbage-like
    language_score: float       # 0–1, higher = valid PL/EN
    structure_score: float      # 0–1, higher = structured field / ID
    dictionary_score: float     # 0–1, higher = lexicon match
    reasons: list[str] = field(default_factory=list)
    protected: bool = False
    tier: str = "medium"
    threshold_used: float = 0.55

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text[:80],
            "is_garbage": self.is_garbage,
            "noise_score": round(self.noise_score, 3),
            "language_score": round(self.language_score, 3),
            "structure_score": round(self.structure_score, 3),
            "dictionary_score": round(self.dictionary_score, 3),
            "reasons": self.reasons,
            "protected": self.protected,
            "tier": self.tier,
            "threshold_used": round(self.threshold_used, 3),
        }


# ── protected patterns (never classify as garbage) ────────────────────────────

_PL_VOWELS = frozenset("aąeęioóuyy")
_EN_VOWELS = frozenset("aeiouy")

_FORM_MARKERS = frozenset({"zł", "zl", "zt", "gr", "pln", "…", "..", "..."})

_ALLOWED_SHORT = frozenset({
    "pl", "up", "nr", "np", "ul", "do", "od", "na", "po", "we", "i", "w", "z",
    "fv", "fp", "lp", "jm", "qr", "ksef",
})

# Reuse semantic-validator patterns for protected identifiers
_PESEL_RE = re.compile(r"^\d{11}$")
_NIP_RE = re.compile(r"^\d{3}[-\s]?\d{2}[-\s]?\d{2}[-\s]?\d{3}$|^\d{10}$")
_REGON_RE = re.compile(r"^\d{9}$|^\d{14}$")
_IBAN_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$", re.IGNORECASE)
_INVOICE_ID_RE = re.compile(
    r"^(?:"
    r"(?:FV|FA|FP|PRO|KOR)\s*[/\-\s]?\d{1,6}(?:[/\-\s]\d{1,6}){1,3}"
    r"|\d{1,4}/\d{1,4}/(?:19|20)\d{2}"
    r"|[A-Z]{2,4}[-/]?\d{4,12}"
    r")$",
    re.IGNORECASE,
)
_TRACKING_RE = re.compile(
    r"^[A-Z0-9]{10,30}$",   # UPS/FedEx/Poczta Polska style
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"^\d{4}[-/.]\d{2}[-/.]\d{2}$|^\d{2}[-/.]\d{2}[-/.]\d{4}$",
)
_AMOUNT_RE = re.compile(
    r"^\d{1,3}(?:[ \s]\d{3})*(?:[,.]\d{2})?\s*(?:zł|zl|pln|gr)?$",
    re.IGNORECASE,
)
_BARCODE_GARBAGE_RE = re.compile(r"^[|Il1!]{4,}$|^[█▌▐░]{3,}$")
_SYMBOL_HEAVY_RE = re.compile(r"^[^\w\sąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{2,}$")
_HALLUCINATION_RE = re.compile(r"\[\?+\]|^(\?\s*)+$|^[?]{2,}$")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{4,}")
_CONSONANT_RUN_RE = re.compile(
    r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{5,}",
)
_MALFORMED_MIXED_RE = re.compile(
    r"^\d{1,3}\s+[A-Za-z]{1,3}$",   # "16 Z"
)
_TILE_SHORT_GARBAGE_RE = re.compile(r"^[A-Z]{2,3}$")


def _is_form_field_marker_legacy(text: str) -> bool:
    """Dots and currency unit labels — always keep."""
    t = (text or "").strip()
    if not t:
        return False
    if t.casefold() in _FORM_MARKERS:
        return True
    if t in ("…", "..", "..."):
        return True
    if len(t) <= 4 and all(c in ".…" for c in t):
        return True
    return False


# ── adaptive thresholds ───────────────────────────────────────────────────────


def adaptive_noise_threshold(tier: str | None = None) -> float:
    """
    Return the noise_score cutoff for classifying a token as garbage.

    EASY   → lower threshold (stricter — filter more aggressively)
    HARD   → higher threshold (softer — keep borderline tokens)
    """
    from app.core.config import settings

    t = (tier or get_noise_tier()).casefold()
    if t == "easy":
        return float(settings.ocr_noise_threshold_easy)
    if t == "hard":
        return float(settings.ocr_noise_threshold_hard)
    return float(settings.ocr_noise_threshold_medium)


# ── protected check ───────────────────────────────────────────────────────────


def is_protected_token(text: str) -> bool:
    """
    True when the token must never be dropped (IDs, form markers, amounts).
    """
    t = (text or "").strip()
    if not t:
        return False
    fold = t.casefold()
    if fold in _FORM_MARKERS or fold in _ALLOWED_SHORT:
        return True
    if len(t) <= 4 and all(c in ".…" for c in t):
        return True
    digits = re.sub(r"\D", "", t)
    if _PESEL_RE.match(digits) or _PESEL_RE.match(t):
        return True
    if _NIP_RE.match(t) or (len(digits) == 10 and digits.isdigit()):
        return True
    if _REGON_RE.match(digits) or _REGON_RE.match(t):
        return True
    if _IBAN_RE.match(t.replace(" ", "")):
        return True
    if _INVOICE_ID_RE.match(t):
        return True
    if _TRACKING_RE.match(t.replace(" ", "").replace("-", "")):
        return True
    if _DATE_RE.match(t):
        return True
    if _AMOUNT_RE.match(t):
        return True
    return False


# ── score helpers ─────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    return [w for w in re.split(r"[\s/\\|,;]+", text) if w]


def _has_pl_diacritic(text: str) -> bool:
    pl = set("ąćęłńóśźż")
    return any(c in pl for c in text)


def _vowel_ratio(word: str, vowels: frozenset[str]) -> float:
    alpha = [c.casefold() for c in word if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c in vowels) / len(alpha)


def score_dictionary(text: str) -> float:
    """Fraction of alpha tokens (≥2 chars) found in PL/EN lexicon."""
    from app.services.ocr_lexicons import ALL_KNOWN, EN_WORDS, PL_WORDS

    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    hits = 0
    checked = 0
    for tok in tokens:
        fold = tok.casefold().strip(".,;:!?()")
        if len(fold) < 2:
            continue
        if not any(c.isalpha() for c in fold):
            continue
        checked += 1
        if fold in ALL_KNOWN:
            hits += 1
        elif any(fold.startswith(w) and len(fold) <= len(w) + 2 for w in PL_WORDS | EN_WORDS):
            hits += 0.5
    if checked == 0:
        return 0.5 if re.search(r"\d", text) else 0.0
    return min(1.0, hits / checked)


def score_language(text: str) -> float:
    """
    Heuristic PL/EN language plausibility in [0, 1].
    """
    t = (text or "").strip()
    if not t:
        return 0.0

    dict_s = score_dictionary(t)
    if dict_s >= 0.5:
        return min(1.0, 0.5 + dict_s * 0.5)

    tokens = _tokenize(t)
    alpha_tokens = [tok for tok in tokens if sum(c.isalpha() for c in tok) >= 2]
    if not alpha_tokens:
        if _MALFORMED_MIXED_RE.match(t):
            return 0.15
        return 0.6 if re.fullmatch(r"[\d\s.,\-/]+", t) else 0.2

    pl_viable = 0
    en_viable = 0
    for tok in alpha_tokens:
        fold = tok.casefold()
        pl_vr = _vowel_ratio(fold, _PL_VOWELS)
        en_vr = _vowel_ratio(fold, _EN_VOWELS)
        if pl_vr >= 0.25 or _has_pl_diacritic(fold):
            pl_viable += 1
        if en_vr >= 0.20:
            en_viable += 1

    n = len(alpha_tokens)
    best = max(pl_viable, en_viable) / n
    if _has_pl_diacritic(t):
        best = min(1.0, best + 0.15)
    return best


def score_structure(text: str) -> float:
    """Higher when text matches document field patterns (IDs, dates, amounts)."""
    t = (text or "").strip()
    if not t:
        return 0.0
    if is_protected_token(t):
        return 1.0
    score = 0.0
    if _DATE_RE.search(t):
        score = max(score, 0.85)
    if _AMOUNT_RE.match(t):
        score = max(score, 0.80)
    if re.search(r"\d{2,}", t) and re.search(r"[A-Za-ząćęłńóśźżĄĆĘŁŃÓŚŹŻ]{2,}", t):
        score = max(score, 0.55)   # mixed alphanumeric field
    if re.fullmatch(r"[\d\s\-./]+", t) and len(re.sub(r"\D", "", t)) >= 4:
        score = max(score, 0.65)   # numeric field
    if len(t) >= 8 and t.isupper() and t.isalpha():
        score = max(score, 0.50)   # header label
    return score


def _symbol_pollution_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            continue
        if ch in ".,;:-/()+%@#&'":
            continue
        bad += 1
    if total == 0:
        return 1.0
    return bad / total


def _detect_reasons(text: str, ocr_score: float) -> tuple[list[str], float]:
    """
    Return (reasons, raw_noise_component) from individual detectors.
    """
    t = (text or "").strip()
    reasons: list[str] = []
    noise_parts: list[float] = []

    if not t:
        return [NoiseReason.EMPTY.value], 1.0

    # Barcode garbage
    if _BARCODE_GARBAGE_RE.match(t) or (len(t) >= 6 and _symbol_pollution_ratio(t) > 0.7):
        reasons.append(NoiseReason.BARCODE_GARBAGE.value)
        noise_parts.append(0.85)

    # Symbol pollution
    sym_ratio = _symbol_pollution_ratio(t)
    if sym_ratio >= 0.45:
        reasons.append(NoiseReason.SYMBOL_POLLUTION.value)
        noise_parts.append(min(1.0, sym_ratio + 0.2))

    # Hallucination patterns
    if _HALLUCINATION_RE.search(t) or _REPEAT_CHAR_RE.search(t):
        reasons.append(NoiseReason.HALLUCINATION.value)
        noise_parts.append(0.90)

    # Malformed mixed (16 Z)
    if _MALFORMED_MIXED_RE.match(t):
        reasons.append(NoiseReason.MALFORMED_WORD.value)
        noise_parts.append(0.80)

    # Tile artifact — short uppercase gibberish
    if _TILE_SHORT_GARBAGE_RE.match(t) and ocr_score < 0.82:
        reasons.append(NoiseReason.TILE_ARTIFACT.value)
        noise_parts.append(0.75)

    # Consonant runs
    if _CONSONANT_RUN_RE.search(t) and len(t) >= 4:
        reasons.append(NoiseReason.MALFORMED_WORD.value)
        noise_parts.append(0.70)

    # Impossible PL / EN words (long alpha tokens)
    for tok in _tokenize(t):
        alpha = sum(c.isalpha() for c in tok)
        if alpha < 4:
            continue
        fold = tok.casefold()
        if _has_pl_diacritic(t) or _has_pl_diacritic(tok):
            if _vowel_ratio(fold, _PL_VOWELS) < 0.15:
                reasons.append(NoiseReason.IMPOSSIBLE_PL.value)
                noise_parts.append(0.75)
                break
        elif _vowel_ratio(fold, _EN_VOWELS) < 0.12:
            reasons.append(NoiseReason.IMPOSSIBLE_EN.value)
            noise_parts.append(0.72)
            break

    # Repeated garbage (same 1-2 char token repeated)
    toks = _tokenize(t)
    if len(toks) >= 2 and len(set(t.casefold() for t in toks)) == 1 and len(toks[0]) <= 2:
        reasons.append(NoiseReason.REPEATED_GARBAGE.value)
        noise_parts.append(0.78)

    # Low confidence short fragment
    if len(t) <= 4 and ocr_score < 0.75 and sym_ratio > 0.2:
        reasons.append(NoiseReason.LOW_CONF_FRAGMENT.value)
        noise_parts.append(0.65)

    # Pure symbol runs
    if _SYMBOL_HEAVY_RE.match(t):
        reasons.append(NoiseReason.SYMBOL_POLLUTION.value)
        noise_parts.append(0.88)

    raw = max(noise_parts) if noise_parts else 0.0
    return reasons, raw


def compute_noise_score(
    text: str,
    ocr_score: float,
    *,
    language_score: float,
    structure_score: float,
    dictionary_score: float,
) -> float:
    """
    Composite noise score in [0, 1] combining detector signals and OCR confidence.
    """
    reasons, raw = _detect_reasons(text, ocr_score)

    # Base from detectors
    noise = raw

    # Low OCR confidence pushes toward garbage
    if ocr_score < 0.60:
        noise = max(noise, 0.55 * (1.0 - ocr_score))

    # Good language / dictionary / structure reduce noise
    mitigation = (
        language_score * 0.30
        + dictionary_score * 0.35
        + structure_score * 0.35
    )
    noise = max(0.0, noise - mitigation * 0.55)

    # Very short unknown fragments
    t = (text or "").strip()
    if len(t) <= 3 and dictionary_score < 0.2 and structure_score < 0.3:
        noise = max(noise, 0.60 if ocr_score < 0.80 else 0.45)

    return min(1.0, noise)


# ── public API ────────────────────────────────────────────────────────────────


def classify_token(
    text: str,
    ocr_score: float = 1.0,
    *,
    tier: str | None = None,
) -> NoiseClassification:
    """
    Classify a single OCR detection token.

    Parameters
    ----------
    text       : OCR text for one detection box
    ocr_score  : RapidOCR confidence in [0, 1]
    tier       : difficulty tier override (``easy`` / ``medium`` / ``hard``)
    """
    from app.core.config import settings

    if not settings.ocr_noise_classifier_enabled:
        from app.services.ocr_dedup import _legacy_spurious_ocr_fragment
        legacy = _legacy_spurious_ocr_fragment(text, ocr_score)
        return NoiseClassification(
            text=text or "",
            ocr_score=ocr_score,
            is_garbage=legacy,
            noise_score=0.8 if legacy else 0.1,
            language_score=0.5,
            structure_score=0.5,
            dictionary_score=0.5,
            reasons=["legacy_spurious"] if legacy else [],
            tier=tier or get_noise_tier(),
            threshold_used=0.5,
        )

    t = (text or "").strip()
    current_tier = (tier or get_noise_tier()).casefold()
    threshold = adaptive_noise_threshold(current_tier)

    if is_protected_token(t):
        lang = score_language(t)
        struct = score_structure(t)
        dict_s = score_dictionary(t)
        return NoiseClassification(
            text=t,
            ocr_score=ocr_score,
            is_garbage=False,
            noise_score=0.0,
            language_score=lang,
            structure_score=max(struct, 1.0),
            dictionary_score=dict_s,
            reasons=[],
            protected=True,
            tier=current_tier,
            threshold_used=threshold,
        )

    lang = score_language(t)
    struct = score_structure(t)
    dict_s = score_dictionary(t)
    reasons, raw = _detect_reasons(t, ocr_score)
    noise = compute_noise_score(
        t, ocr_score,
        language_score=lang,
        structure_score=struct,
        dictionary_score=dict_s,
    )

    is_garbage = False
    # Hard-drop patterns that are never real document content
    if NoiseReason.MALFORMED_WORD.value in reasons and raw >= 0.75:
        is_garbage = True
    elif NoiseReason.HALLUCINATION.value in reasons:
        is_garbage = True
    elif NoiseReason.BARCODE_GARBAGE.value in reasons and raw >= 0.80:
        is_garbage = True
    elif noise >= threshold:
        if struct >= 0.75:
            is_garbage = False   # protected structure
        elif noise >= threshold + 0.12:
            is_garbage = True    # very noisy regardless
        elif dict_s < 0.25 and lang < 0.35:
            is_garbage = True
        elif struct < 0.40:
            is_garbage = True

    # Form markers (always keep)
    if t.casefold() in _FORM_MARKERS or _is_form_field_marker_legacy(t):
        is_garbage = False

    return NoiseClassification(
        text=t,
        ocr_score=ocr_score,
        is_garbage=is_garbage,
        noise_score=noise,
        language_score=lang,
        structure_score=struct,
        dictionary_score=dict_s,
        reasons=reasons,
        protected=False,
        tier=current_tier,
        threshold_used=threshold,
    )


def is_garbage_ocr(
    text: str,
    ocr_score: float = 1.0,
    *,
    tier: str | None = None,
) -> bool:
    """Return True when the token should be dropped before merge."""
    from app.core.config import settings
    if not settings.ocr_filter_spurious_fragments and not settings.ocr_noise_classifier_enabled:
        return False
    return classify_token(text, ocr_score, tier=tier).is_garbage


def filter_noisy_detections(
    parts: Sequence[str],
    bboxes: Sequence[tuple[int, int, int, int]],
    scores: Sequence[float],
    *,
    tier: str | None = None,
) -> tuple[list[str], list[tuple[int, int, int, int]], list[float], dict[str, Any]]:
    """
    Remove garbage detections before line merge / dedup.

    Returns filtered (parts, bboxes, scores) plus summary metadata.
    """
    out_p: list[str] = []
    out_b: list[tuple[int, int, int, int]] = []
    out_s: list[float] = []
    removed = 0
    reason_counts: dict[str, int] = {}

    for txt, bbox, sc in zip(parts, bboxes, scores):
        cls = classify_token(txt or "", float(sc), tier=tier)
        if cls.is_garbage:
            removed += 1
            for r in cls.reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
            continue
        out_p.append(txt)
        out_b.append(bbox)
        out_s.append(float(sc))

    meta = {
        "noise_filtered": removed,
        "noise_reasons": reason_counts,
        "tier": tier or get_noise_tier(),
    }
    return out_p, out_b, out_s, meta


def classify_batch(
    texts: Sequence[str],
    scores: Sequence[float],
    *,
    tier: str | None = None,
) -> list[NoiseClassification]:
    """Classify a sequence of OCR tokens."""
    return [classify_token(t, float(s), tier=tier) for t, s in zip(texts, scores)]
