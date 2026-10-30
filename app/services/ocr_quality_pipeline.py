"""
Global OCR quality scoring pipeline.

Computes a single **quality_score** in [0, 100] for any OCR output and
determines whether the result is trustworthy or needs recovery actions.

Dimensions analysed
-------------------
  confidence            RapidOCR mean / min scores, low-confidence ratio
  malformed_words       noise-classifier garbage token density
  duplicated_lines      exact / near-duplicate line detection
  overlap_artifacts     tile overlap & pre-merge dedup signals from meta
  semantic_validation   PESEL / NIP / IBAN / date checksum failures
  garbage_density       overall OCR noise fraction
  layout_integrity      reading-order / column / block health
  language_consistency  PL/EN script & lexicon coherence
  entropy_anomaly       character-entropy outlier detection

Quality tiers
-------------
  90–100  EXCELLENT   — accept as-is
  75–89   ACCEPTABLE  — minor issues, no action required
  50–74   SUSPICIOUS  — optional light retry / verification
   0–49   POOR        — retry preprocessing / OCR / segmentation / MiniCPM

Tracing
-------
``QualityPipelineSession`` (thread-local) accumulates across retry attempts:
  * retry_history       — every assessment + action taken
  * degradation_trace   — score drops between attempts
  * escalation_trace    — engine / tier escalations

Usage
-----
    from app.services.ocr_quality_pipeline import (
        assess_ocr_output,
        attach_quality_to_meta,
        QualityPipelineSession,
        use_quality_session,
    )

    with use_quality_session() as session:
        text, conf, meta = ocr_image(path)
        attach_quality_to_meta(meta, text)
        if meta["quality"]["needs_recovery"]:
            ...  # plan_recovery_actions(meta["quality"])
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ── tier enum ─────────────────────────────────────────────────────────────────


class QualityTier(str, Enum):
    EXCELLENT = "excellent"       # 90–100
    ACCEPTABLE = "acceptable"     # 75–89
    SUSPICIOUS = "suspicious"     # 50–74
    POOR = "poor"                 # 0–49  → retry / escalate


class RecoveryAction(str, Enum):
    PREPROCESS_RETRY = "preprocess_retry"
    OCR_RETRY = "ocr_retry"
    SEGMENTATION_RETRY = "segmentation_retry"
    MINICPM_ESCALATION = "minicpm_escalation"
    TESSERACT_VERIFY = "tesseract_verify"
    NONE = "none"


_TIER_THRESHOLDS = (
    (90.0, QualityTier.EXCELLENT),
    (75.0, QualityTier.ACCEPTABLE),
    (50.0, QualityTier.SUSPICIOUS),
    (0.0, QualityTier.POOR),
)


# ── dimension weights (must sum to 1.0) ───────────────────────────────────────

_DIMENSION_WEIGHTS: dict[str, float] = {
    "confidence": 0.20,
    "malformed_words": 0.12,
    "duplicated_lines": 0.10,
    "overlap_artifacts": 0.08,
    "semantic_validation": 0.12,
    "garbage_density": 0.10,
    "layout_integrity": 0.10,
    "language_consistency": 0.10,
    "entropy_anomaly": 0.08,
}


# ── dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class RecoveryPlan:
    """Recommended recovery action for a low-quality OCR result."""

    action: RecoveryAction
    reason: str
    priority: int = 1   # lower = try first

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "priority": self.priority,
        }


@dataclass
class QualityAssessment:
    """Complete quality analysis for one OCR output."""

    quality_score: float                    # 0–100 composite
    tier: QualityTier
    dimensions: dict[str, float]            # each dimension 0–100 (higher = better)
    issues: list[str] = field(default_factory=list)
    recommended_actions: list[RecoveryPlan] = field(default_factory=list)
    needs_recovery: bool = False
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_score": round(self.quality_score, 1),
            "quality_tier": self.tier.value,
            "dimensions": {k: round(v, 1) for k, v in self.dimensions.items()},
            "issues": self.issues,
            "needs_recovery": self.needs_recovery,
            "recommended_actions": [a.to_dict() for a in self.recommended_actions],
            "analysis_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class QualityPipelineSession:
    """
    Tracks quality across multiple OCR attempts within one job/page.

    Attach via ``use_quality_session()`` at the start of processing.
    """

    retry_history: list[dict[str, Any]] = field(default_factory=list)
    degradation_trace: list[dict[str, Any]] = field(default_factory=list)
    escalation_trace: list[dict[str, Any]] = field(default_factory=list)
    attempts: int = 0
    best_score: float = 0.0
    best_text: str = ""
    best_tier: str = QualityTier.POOR.value

    def record_assessment(
        self,
        assessment: QualityAssessment,
        *,
        text: str = "",
        engine: str | None = None,
        action_taken: str | None = None,
    ) -> None:
        self.attempts += 1
        prev_best = self.best_score
        entry = {
            "attempt": self.attempts,
            "quality_score": round(assessment.quality_score, 1),
            "tier": assessment.tier.value,
            "engine": engine,
            "action_taken": action_taken,
            "issues": assessment.issues[:6],
            "text_chars": len(text),
            "ts": round(time.time(), 3),
        }
        self.retry_history.append(entry)

        if assessment.quality_score > self.best_score:
            self.best_score = assessment.quality_score
            self.best_text = text
            self.best_tier = assessment.tier.value
        elif prev_best > 0 and assessment.quality_score < prev_best - 1.0:
            self.degradation_trace.append({
                "attempt": self.attempts,
                "score_before": round(prev_best, 1),
                "score_after": round(assessment.quality_score, 1),
                "delta": round(assessment.quality_score - prev_best, 1),
                "reason": "score_regression",
            })

    def record_degradation(
        self,
        score_before: float,
        score_after: float,
        *,
        reason: str,
        engine: str | None = None,
    ) -> None:
        self.degradation_trace.append({
            "attempt": self.attempts,
            "score_before": round(score_before, 1),
            "score_after": round(score_after, 1),
            "delta": round(score_after - score_before, 1),
            "reason": reason,
            "engine": engine,
            "ts": round(time.time(), 3),
        })

    def record_escalation(
        self,
        *,
        from_engine: str,
        to_engine: str,
        reason: str,
        lines_escalated: int = 0,
    ) -> None:
        self.escalation_trace.append({
            "attempt": self.attempts,
            "from_engine": from_engine,
            "to_engine": to_engine,
            "reason": reason,
            "lines_escalated": lines_escalated,
            "ts": round(time.time(), 3),
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "best_score": round(self.best_score, 1),
            "best_tier": self.best_tier,
            "retry_history": self.retry_history,
            "degradation_trace": self.degradation_trace,
            "escalation_trace": self.escalation_trace,
        }


# ── thread-local session ──────────────────────────────────────────────────────

_session_state = threading.local()


def current_quality_session() -> QualityPipelineSession | None:
    return getattr(_session_state, "session", None)


def _set_session(session: QualityPipelineSession | None) -> None:
    _session_state.session = session


@contextmanager
def use_quality_session(session: QualityPipelineSession | None = None) -> Iterator[QualityPipelineSession]:
    """Activate a quality session for the current thread (nestable)."""
    sess = session or QualityPipelineSession()
    prev = current_quality_session()
    _set_session(sess)
    try:
        yield sess
    finally:
        _set_session(prev)


# ── tier mapping ──────────────────────────────────────────────────────────────


def tier_for_score(score: float) -> QualityTier:
    from app.core.config import settings
    thresholds = (
        (float(settings.ocr_quality_threshold_excellent), QualityTier.EXCELLENT),
        (float(settings.ocr_quality_threshold_acceptable), QualityTier.ACCEPTABLE),
        (float(settings.ocr_quality_threshold_suspicious), QualityTier.SUSPICIOUS),
        (0.0, QualityTier.POOR),
    )
    for min_score, tier in thresholds:
        if score >= min_score:
            return tier
    return QualityTier.POOR


# ── dimension scorers (each returns 0–100, higher = better) ───────────────────


def _score_confidence(meta: dict[str, Any]) -> tuple[float, list[str]]:
    issues: list[str] = []
    mean = float(meta.get("mean_score") or meta.get("confidence") or 0.0)
    low_ratio = float(meta.get("low_conf_ratio") or 0.0)
    min_sc = float(meta.get("min_score") or mean)

    if mean <= 0 and "mean_score" not in meta:
        return 70.0, issues   # no signal — neutral

    base = mean * 70.0 + (1.0 - low_ratio) * 20.0 + min(min_sc, 1.0) * 10.0
    score = max(0.0, min(100.0, base))

    if mean < 0.65:
        issues.append(f"low_mean_conf={mean:.2f}")
    if low_ratio > 0.25:
        issues.append(f"high_low_conf_ratio={low_ratio:.2f}")
    if min_sc < 0.40:
        issues.append(f"very_low_min_conf={min_sc:.2f}")

    return score, issues


def _tokenize_lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _score_malformed_and_garbage(
    text: str,
    lines: list | None,
) -> tuple[float, float, list[str]]:
    """Returns (malformed_score, garbage_density_score, issues)."""
    from app.services.ocr_noise_classifier import classify_token

    issues: list[str] = []
    garbage = 0
    total = 0

    line_texts: list[tuple[str, float]] = []
    if lines:
        for ln in lines:
            txt = (getattr(ln, "text", None) or str(ln)).strip()
            sc = float(getattr(ln, "mean_score", 0.85) or 0.85)
            if txt:
                line_texts.append((txt, sc))
    else:
        for txt in _tokenize_lines(text):
            line_texts.append((txt, 0.85))

    for txt, sc in line_texts:
        total += 1
        if classify_token(txt, sc).is_garbage:
            garbage += 1
        for tok in re.split(r"[\s/\\|,;]+", txt):
            tok = tok.strip()
            if tok and any(c.isalnum() for c in tok):
                total += 1
                if classify_token(tok, sc).is_garbage:
                    garbage += 1

    if total == 0:
        return 50.0, 50.0, issues

    ratio = garbage / total
    score = max(0.0, (1.0 - ratio) * 100.0)

    if ratio > 0.15:
        issues.append(f"garbage_token_ratio={ratio:.2f}")
    if ratio > 0.30:
        issues.append("high_garbage_density")

    return score, score, issues


def _score_duplicated_lines(text: str) -> tuple[float, list[str]]:
    issues: list[str] = []
    lines = _tokenize_lines(text)
    if len(lines) < 2:
        return 100.0, issues

    normalized = [re.sub(r"\s+", " ", ln.casefold()) for ln in lines]
    counts = Counter(normalized)
    dup_lines = sum(c - 1 for c in counts.values() if c > 1)
    ratio = dup_lines / len(lines)
    score = max(0.0, (1.0 - ratio * 2.0) * 100.0)   # dup ratio weighted 2×

    if dup_lines >= 2:
        issues.append(f"duplicate_lines={dup_lines}")
    return score, issues


def _score_overlap_artifacts(meta: dict[str, Any]) -> tuple[float, list[str]]:
    issues: list[str] = []
    reading = meta.get("reading_order") or {}
    layout = meta.get("layout_meta") or reading

    overlap = int(layout.get("overlap_removed") or 0)
    noise_filt = int(layout.get("noise_filtered") or layout.get("filtered_garbage") or 0)
    n_boxes = int(meta.get("n_boxes") or 0)

    if n_boxes <= 0 and overlap == 0 and noise_filt == 0:
        return 85.0, issues

    artifact_ratio = (overlap + noise_filt) / max(1, n_boxes)
    score = max(0.0, min(100.0, (1.0 - artifact_ratio * 3.0) * 100.0))

    if overlap >= 3:
        issues.append(f"overlap_removed={overlap}")
    if noise_filt >= 5:
        issues.append(f"noise_filtered={noise_filt}")
    return score, issues


def _score_semantic_validation(text: str) -> tuple[float, list[str]]:
    from app.core.config import settings

    issues: list[str] = []
    if not settings.ocr_semantic_validation_enabled:
        return 90.0, issues

    from app.services.semantic_validators.registry import validate_text
    from app.services.semantic_validators.types import ValidationSeverity

    result = validate_text(text)
    errors = [
        i for i in result.issues
        if not i.valid and i.severity == ValidationSeverity.ERROR
    ]
    warnings = [
        i for i in result.issues
        if not i.valid and i.severity == ValidationSeverity.WARNING
    ]

    if not result.issues:
        return 95.0, issues

    penalty = len(errors) * 25 + len(warnings) * 8
    score = max(0.0, 100.0 - penalty)

    for err in errors[:3]:
        issues.append(f"semantic_{err.kind}:{err.reason or 'invalid'}")
    return score, issues


def _score_layout_integrity(meta: dict[str, Any]) -> tuple[float, list[str]]:
    issues: list[str] = []
    reading = meta.get("reading_order") or {}
    layout = meta.get("layout_meta") or reading

    if not layout:
        return 80.0, issues

    score = 90.0
    strategy = layout.get("reading_order_strategy", "single_column")
    cols = int(layout.get("detected_columns") or 1)
    blocks = int(layout.get("block_count") or 1)
    merges = int(layout.get("merge_operations") or 0)
    garbage = int(layout.get("filtered_garbage") or layout.get("noise_filtered") or 0)

    if garbage > 10:
        score -= 15
        issues.append(f"layout_garbage_filtered={garbage}")
    if merges > 50:
        score -= 10
        issues.append(f"excessive_merges={merges}")
    if cols > 3:
        score -= 10
        issues.append(f"many_columns={cols}")
    if blocks > 8:
        score -= 5
        issues.append(f"many_blocks={blocks}")

    # Unknown strategy is fine; log for debug
    if strategy not in ("single_column", "multi_column", "form"):
        issues.append(f"unknown_strategy={strategy}")

    return max(0.0, min(100.0, score)), issues


def _score_language_consistency(text: str) -> tuple[float, list[str]]:
    from app.services.ocr_lexicons import EN_WORDS, PL_WORDS

    issues: list[str] = []
    if not text or len(text) < 20:
        return 75.0, issues

    pl_chars = set("ąćęłńóśźż")
    has_pl = any(c in pl_chars for c in text.casefold())
    tokens = [
        t.casefold().strip(".,;:!?()")
        for t in re.split(r"[\s/\\|,;]+", text)
        if len(t) >= 3 and t.isalpha()
    ]
    if not tokens:
        return 70.0, issues

    pl_hits = sum(1 for t in tokens if t in PL_WORDS)
    en_hits = sum(1 for t in tokens if t in EN_WORDS)
    known = pl_hits + en_hits
    known_ratio = known / len(tokens)

    # Inconsistent: PL diacritics expected but almost no PL lexicon hits
    if has_pl and pl_hits == 0 and len(tokens) > 5:
        issues.append("pl_diacritics_but_no_lexicon_hits")
        return max(0.0, 40.0 + known_ratio * 30.0), issues

    score = 50.0 + known_ratio * 50.0
    if has_pl and pl_hits > 0:
        score = min(100.0, score + 10.0)

    if known_ratio < 0.10 and len(tokens) > 8:
        issues.append(f"low_lexicon_coverage={known_ratio:.2f}")

    return max(0.0, min(100.0, score)), issues


def _char_entropy(text: str) -> float:
    """Shannon entropy (bits) over non-whitespace characters."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    counts = Counter(chars)
    n = len(chars)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _score_entropy_anomaly(text: str) -> tuple[float, list[str]]:
    issues: list[str] = []
    if len(text) < 30:
        return 85.0, issues

    ent = _char_entropy(text)
    # Normal Latin document text: ~3.8–4.6 bits
    # Too low → repetitive garbage; too high → random OCR noise
    if ent < 2.8:
        issues.append(f"entropy_too_low={ent:.2f}")
        return max(0.0, ent / 2.8 * 60.0), issues
    if ent > 5.2:
        issues.append(f"entropy_too_high={ent:.2f}")
        return max(0.0, 100.0 - (ent - 5.2) * 40.0), issues
    return 95.0, issues


# ── recovery planning ─────────────────────────────────────────────────────────


def plan_recovery_actions(
    assessment: QualityAssessment,
    meta: dict[str, Any] | None = None,
) -> list[RecoveryPlan]:
    """Recommend recovery actions based on dimension scores and tier."""
    if assessment.tier in (QualityTier.EXCELLENT, QualityTier.ACCEPTABLE):
        return []

    dims = assessment.dimensions
    plans: list[RecoveryPlan] = []

    if dims.get("confidence", 100) < 65:
        plans.append(RecoveryPlan(
            RecoveryAction.OCR_RETRY,
            "low OCR confidence — retry with server model",
            priority=1,
        ))

    if dims.get("garbage_density", 100) < 55 or dims.get("malformed_words", 100) < 55:
        plans.append(RecoveryPlan(
            RecoveryAction.PREPROCESS_RETRY,
            "high garbage density — retry with enhanced preprocessing",
            priority=2,
        ))

    if dims.get("layout_integrity", 100) < 60 or dims.get("duplicated_lines", 100) < 55:
        plans.append(RecoveryPlan(
            RecoveryAction.SEGMENTATION_RETRY,
            "layout corruption or duplicate lines — retry segmentation",
            priority=3,
        ))

    if dims.get("semantic_validation", 100) < 70:
        plans.append(RecoveryPlan(
            RecoveryAction.TESSERACT_VERIFY,
            "semantic validator failures — run Tesseract line verification",
            priority=4,
        ))

    if assessment.tier == QualityTier.POOR:
        plans.append(RecoveryPlan(
            RecoveryAction.MINICPM_ESCALATION,
            f"quality_score={assessment.quality_score:.0f} — escalate suspicious lines to MiniCPM",
            priority=5,
        ))
    elif assessment.tier == QualityTier.SUSPICIOUS and assessment.quality_score < 65:
        plans.append(RecoveryPlan(
            RecoveryAction.MINICPM_ESCALATION,
            f"quality_score={assessment.quality_score:.0f} — optional crop VLM escalation",
            priority=6,
        ))

    plans.sort(key=lambda p: p.priority)
    return plans


# ── main assessment ───────────────────────────────────────────────────────────


def assess_ocr_output(
    text: str,
    meta: dict[str, Any] | None = None,
    *,
    lines: list | None = None,
) -> QualityAssessment:
    """
    Compute global quality score (0–100) for an OCR result.

    Parameters
    ----------
    text  : final OCR text
    meta  : OCR engine metadata (mean_score, reading_order, n_boxes, …)
    lines : optional list of FastOCRLine objects for per-line analysis
    """
    from app.core.config import settings

    t0 = time.perf_counter()
    meta = meta or {}
    all_issues: list[str] = []
    dimensions: dict[str, float] = {}

    # ── compute each dimension ────────────────────────────────────────────────
    conf_score, conf_issues = _score_confidence(meta)
    dimensions["confidence"] = conf_score
    all_issues.extend(conf_issues)

    mal_score, garb_score, garb_issues = _score_malformed_and_garbage(text, lines)
    dimensions["malformed_words"] = mal_score
    dimensions["garbage_density"] = garb_score
    all_issues.extend(garb_issues)

    dup_score, dup_issues = _score_duplicated_lines(text)
    dimensions["duplicated_lines"] = dup_score
    all_issues.extend(dup_issues)

    overlap_score, overlap_issues = _score_overlap_artifacts(meta)
    dimensions["overlap_artifacts"] = overlap_score
    all_issues.extend(overlap_issues)

    sem_score, sem_issues = _score_semantic_validation(text)
    dimensions["semantic_validation"] = sem_score
    all_issues.extend(sem_issues)

    layout_score, layout_issues = _score_layout_integrity(meta)
    dimensions["layout_integrity"] = layout_score
    all_issues.extend(layout_issues)

    lang_score, lang_issues = _score_language_consistency(text)
    dimensions["language_consistency"] = lang_score
    all_issues.extend(lang_issues)

    ent_score, ent_issues = _score_entropy_anomaly(text)
    dimensions["entropy_anomaly"] = ent_score
    all_issues.extend(ent_issues)

    # ── weighted composite ────────────────────────────────────────────────────
    weights = _load_weights(settings)
    composite = sum(
        dimensions.get(dim, 75.0) * weights.get(dim, 0.0)
        for dim in weights
    )
    quality_score = max(0.0, min(100.0, composite))

    tier = tier_for_score(quality_score)
    needs_recovery = tier in (QualityTier.SUSPICIOUS, QualityTier.POOR)

    assessment = QualityAssessment(
        quality_score=quality_score,
        tier=tier,
        dimensions=dimensions,
        issues=all_issues[:20],
        needs_recovery=needs_recovery,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )
    assessment.recommended_actions = plan_recovery_actions(assessment, meta)

    logger.debug(
        "OCR quality: score=%.1f tier=%s issues=%d recovery=%s (%.1f ms)",
        quality_score, tier.value, len(all_issues), needs_recovery,
        assessment.elapsed_ms,
    )
    return assessment


def _load_weights(settings) -> dict[str, float]:
    """Load dimension weights from settings when present, else defaults."""
    custom = {}
    for dim in _DIMENSION_WEIGHTS:
        key = f"ocr_quality_weight_{dim}"
        if hasattr(settings, key):
            custom[dim] = float(getattr(settings, key))
    if custom:
        total = sum(custom.values()) or 1.0
        return {k: v / total for k, v in custom.items()}
    return dict(_DIMENSION_WEIGHTS)


# ── integration helpers ───────────────────────────────────────────────────────


def attach_quality_to_meta(
    meta: dict[str, Any],
    text: str,
    *,
    lines: list | None = None,
    record_session: bool = True,
) -> QualityAssessment:
    """
    Run quality assessment and attach results to OCR metadata dict.

    Adds keys: ``quality``, ``quality_score``, ``quality_tier``,
    ``quality_needs_recovery``, ``quality_actions``.
    When a ``QualityPipelineSession`` is active, records the attempt.
    """
    from app.core.config import settings

    if not settings.ocr_quality_pipeline_enabled:
        return QualityAssessment(
            quality_score=75.0,
            tier=QualityTier.ACCEPTABLE,
            dimensions={},
        )

    # Merge layout_meta into meta for scoring if present on fast result path
    if "reading_order" in meta and "layout_meta" not in meta:
        meta = {**meta, "layout_meta": meta["reading_order"]}

    assessment = assess_ocr_output(text, meta, lines=lines)

    meta["quality"] = assessment.to_dict()
    meta["quality_score"] = round(assessment.quality_score, 1)
    meta["quality_tier"] = assessment.tier.value
    meta["quality_needs_recovery"] = assessment.needs_recovery
    if assessment.recommended_actions:
        meta["quality_actions"] = [a.to_dict() for a in assessment.recommended_actions]

    session = current_quality_session()
    if record_session and session is not None:
        session.record_assessment(
            assessment,
            text=text,
            engine=str(meta.get("engine", "")),
        )
        meta["quality_session"] = {
            "attempts": session.attempts,
            "best_score": round(session.best_score, 1),
            "retry_history": session.retry_history[-5:],
            "degradation_trace": session.degradation_trace,
            "escalation_trace": session.escalation_trace,
        }

    return assessment


def should_escalate_for_quality(assessment: QualityAssessment) -> bool:
    """True when quality tier requires escalation (POOR or low SUSPICIOUS)."""
    from app.core.config import settings
    if not settings.ocr_quality_pipeline_enabled:
        return False
    if assessment.tier == QualityTier.POOR:
        return True
    if assessment.tier == QualityTier.SUSPICIOUS:
        return assessment.quality_score < float(settings.ocr_quality_escalate_below)
    return False


def quality_summary(meta: dict[str, Any]) -> str:
    """One-line human-readable quality summary for logs."""
    score = meta.get("quality_score", "?")
    tier = meta.get("quality_tier", "?")
    recovery = meta.get("quality_needs_recovery", False)
    flag = " ⚠ recovery" if recovery else ""
    return f"quality={score} ({tier}){flag}"
