"""
Crop-level vision escalation (replaces full-page VLM fallback).

Flow: RapidOCR → line scoring → suspicious/bad line crops → vision OCR → merge.

Preserves original line order, supports bbox padding, merged multi-line crops,
and a batch interface for future parallel VLM calls.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image

from app.core.config import settings
from app.services.pipeline_profile import current_profile
from app.services.fast_ocr import FastOCRLine, FastOCRResult
from app.services.ocr_consensus import majority_vote_text, pick_best_candidate
from app.services.ocr_line_verification import (
    LineVerificationMetrics,
    log_verification_metrics,
    verify_line_crop,
)
from app.services.ocr_lines import line_dicts_from_line_smart_debug
from app.services.ocr_preprocess_variants import generate_crop_variants
from app.services.ocr_quality import classify_line_escalation, line_quality_score
from app.services.vision_llm import VisionLLMError, ocr_crop_vision

logger = logging.getLogger(__name__)

_QMARK_RUN_RE = re.compile(r"(?:\[\?\]\s*){40,}")
_TRACKING_DIGITS_RE = re.compile(r"\(\d{2,}\)\d{8,}|\b\d{12,}\b")


def _looks_like_unhelpful_vlm(text: str) -> bool:
    """
    Guardrail: reject pathological VLM outputs like thousands of repeated "[?]".

    This is *not* semantic autocorrect — we only discard obviously uninformative
    filler that harms exact transcription output.
    """
    t = (text or "").strip()
    if not t:
        return True
    if len(t) > 800 and _QMARK_RUN_RE.search(t):
        return True
    # Mostly unreadable tokens: keep a short "[?]" or a few, but not spam.
    if len(t) > 200:
        q = t.count("[?]")
        if q >= 40 and (q * 3) / max(len(t), 1) > 0.35:
            return True
    return False


def _vlm_degrades_rapid(rapid: str, vlm: str) -> bool:
    """Reject VLM output that drops structured digits or is far shorter than Rapid."""
    rapid_s = (rapid or "").strip()
    vlm_s = (vlm or "").strip()
    if not rapid_s or not vlm_s:
        return False
    if _looks_like_unhelpful_vlm(vlm_s):
        return True
    for match in _TRACKING_DIGITS_RE.finditer(rapid_s):
        if match.group(0) not in vlm_s:
            return True
    if len(rapid_s) >= 15 and len(vlm_s) < max(8, int(len(rapid_s) * 0.4)):
        return True
    return False


@dataclass(frozen=True)
class CropBatchItem:
    """One region to send to vision OCR (single line or merged adjacent lines)."""

    line_indices: tuple[int, ...]
    bbox: tuple[int, int, int, int]
    rapid_texts: tuple[str, ...]


@dataclass
class LineEscalationResult:
    """Per visual line after scoring and optional crop escalation."""

    line_index: int
    text: str
    confidence: float
    engine: str
    bbox: tuple[int, int, int, int]
    quality_score: float
    escalated: bool
    disagreement_score: float = 0.0
    suspicious: bool = False
    rapid_text: str = ""
    tesseract_text: str = ""
    tess_skipped: bool = False
    escalation_hint: str | None = None
    preprocessing_variant: str | None = None
    escalation_reason: str | None = None
    crop_batch_size: int = 1


@dataclass
class CropEscalationResult:
    """Merged page/crop output after crop-level escalation."""

    text: str
    mean_confidence: float | None
    line_results: list[LineEscalationResult] = field(default_factory=list)
    engines_used: set[str] = field(default_factory=set)
    crop_vlm_calls: int = 0
    elapsed_sec: float = 0.0
    verification_metrics: LineVerificationMetrics | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def crop_line_image(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    pad: int | None = None,
) -> Image.Image:
    from app.services.ocr_line_verification import _crop_line

    return _crop_line(image, bbox, pad if pad is not None else settings.ocr_line_crop_padding_px)


def union_bbox(
    bboxes: list[tuple[int, int, int, int]],
    *,
    pad: int | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Merge line boxes into one crop rectangle (with padding)."""
    if not bboxes:
        return (0, 0, 0, 0)
    pad = settings.ocr_line_crop_padding_px if pad is None else pad
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    if image_size:
        w, h = image_size
        return (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(w, x1 + pad),
            min(h, y1 + pad),
        )
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def _tesseract_line(
    crop: Image.Image,
    lang: str | None,
    *,
    rapid_text: str | None = None,
) -> str:
    from app.services.ocr_line_verification import tesseract_line_crop

    return tesseract_line_crop(crop, lang, rapid_text=rapid_text)


def _try_crop_vlm(
    crop: Image.Image,
    fallback_text: str,
    fallback_score: float,
) -> tuple[str, float, str, bool]:
    def _call() -> tuple[str, float | None, dict]:
        return ocr_crop_vision(crop)

    try:
        from app.core.config import settings as _s
        from app.services.gpu_scheduler import QueueFullError, get_scheduler

        if _s.gpu_scheduler_enabled:
            from app.services.dynamic_gpu_scheduler import TaskPriority

            result = get_scheduler().submit_escalation(
                _call,
                timeout_ms=_s.gpu_scheduler_escalation_timeout_ms,
                fallback=None,
                priority=TaskPriority.ESCALATION,
            )
            if result is None:
                # Timeout or queue full — keep RapidOCR result
                return fallback_text, fallback_score, "fast_ocr", False
            vlm_text, _, _ = result
        else:
            vlm_text, _, _ = ocr_crop_vision(crop)
    except VisionLLMError as exc:
        logger.debug("crop VLM failed: %s", exc)
        return fallback_text, fallback_score, "fast_ocr", False
    except Exception as exc:  # noqa: BLE001
        logger.debug("crop VLM scheduler error: %s", exc)
        return fallback_text, fallback_score, "fast_ocr", False

    if not vlm_text.strip():
        return fallback_text, fallback_score, "fast_ocr", False
    if _looks_like_unhelpful_vlm(vlm_text):
        return fallback_text, fallback_score, "fast_ocr", False
    if fallback_text.strip() and _vlm_degrades_rapid(fallback_text, vlm_text):
        return fallback_text, fallback_score, "fast_ocr", False
    vlm_score = line_quality_score(
        mean_conf=0.9,
        rapid_text=vlm_text,
        tess_text=fallback_text,
    )
    if vlm_score >= fallback_score or not fallback_text.strip():
        return vlm_text.strip(), vlm_score, "vision_llm", True
    return fallback_text, fallback_score, "fast_ocr", True


def _retry_crop_multi_variant(
    crop: Image.Image,
    lang: str,
    *,
    gpu_id: int | None,
) -> tuple[str, float, str, str | None]:
    from app.services.fast_ocr import ocr_pil_fast

    candidates: list[tuple[str, float, str]] = []
    best_variant: str | None = None
    for name, variant in generate_crop_variants(crop):
        try:
            fast = ocr_pil_fast(variant, gpu_id=gpu_id, preprocess_variant=name)
            tess = _tesseract_line(variant, lang, rapid_text=fast.text)
            q = line_quality_score(
                mean_conf=fast.mean_score,
                rapid_text=fast.text,
                tess_text=tess,
            )
            candidates.append((fast.text.strip(), q, "fast_ocr"))
            if tess.strip():
                q_t = line_quality_score(mean_conf=0.75, rapid_text=tess, tess_text=fast.text)
                candidates.append((tess.strip(), q_t, "tesseract"))
            if candidates and q >= candidates[-1][1]:
                best_variant = name
        except Exception as exc:  # noqa: BLE001
            logger.debug("variant %s failed: %s", name, exc)
    if not candidates:
        return "", 0.0, "none", None
    text, score, engine = pick_best_candidate(candidates)
    return text, score, engine, best_variant


def escalate_line_crop(
    crop: Image.Image,
    lang: str,
    *,
    gpu_id: int | None,
    allow_vlm: bool,
    rapid_text: str = "",
) -> tuple[str, float, str, str | None, bool]:
    """Heavy path: crop VLM first (if enabled), else multi-preprocess variants."""
    vlm_used = False
    rapid_fallback = (rapid_text or "").strip()
    if allow_vlm and settings.ocr_line_vlm_without_multi_variant:
        text, score, engine, vlm_used = _try_crop_vlm(
            crop, rapid_fallback, line_quality_score(mean_conf=0.5, rapid_text=rapid_fallback, tess_text=""),
        )
        if text.strip() and not (
            rapid_fallback and text.strip() != rapid_fallback and _vlm_degrades_rapid(rapid_fallback, text)
        ):
            return text, score, engine, "vlm_crop", vlm_used

    text, score, engine, variant = _retry_crop_multi_variant(crop, lang, gpu_id=gpu_id)
    if allow_vlm and (not text.strip() or score < settings.ocr_line_escalate_quality_below):
        fb = text.strip() or rapid_fallback
        text, score, engine, vlm_used = _try_crop_vlm(crop, fb, score)
        if vlm_used and engine == "vision_llm":
            return text, score, engine, variant, True
    if text.strip():
        return text, score, engine, variant, vlm_used
    return "", score, engine, variant, vlm_used


def split_crop_text_for_lines(vlm_text: str, n_lines: int) -> list[str]:
    """Map multi-line VLM output back to *n_lines* slots (preserve count)."""
    if n_lines <= 0:
        return []
    stripped = vlm_text.strip()
    if not stripped:
        return [""] * n_lines
    parts = [p.strip() for p in stripped.splitlines() if p.strip()]
    if len(parts) == n_lines:
        return parts
    if n_lines == 1:
        return [stripped]
    if len(parts) > n_lines:
        return parts[: n_lines - 1] + [" ".join(parts[n_lines - 1 :])]
    out = parts + [""] * (n_lines - len(parts))
    return out[:n_lines]


def _line_needs_crop_vlm(
    *,
    tier: str,
    line_mean_score: float,
    page_quality_poor: bool,
) -> tuple[bool, str]:
    """Whether this line should be sent through crop-level vision OCR."""
    if tier == "hard":
        return True, "hard"
    if tier == "light" and settings.ocr_line_multi_variant_on_light:
        return True, "light"
    if page_quality_poor and line_mean_score < settings.ocr_line_escalate_mean_conf:
        return True, "page_quality_poor"
    return False, "none"


def _select_tesseract_verification_indices(
    lines: list[FastOCRLine],
    *,
    budget_override: int | None = None,
) -> set[int]:
    """
    Auto budget for line-level Tesseract checks.

    Keep verification deterministic and hands-off for mixed client uploads:
    - 0 budget => verify all lines
    - otherwise verify only the lowest-confidence lines (up to budget)
    """
    budget = int(
        budget_override
        if budget_override is not None
        else settings.ocr_line_tesseract_max_lines_per_page
    )
    if budget <= 0 or len(lines) <= budget:
        return set(range(len(lines)))
    ranked = sorted(
        range(len(lines)),
        key=lambda idx: (
            float(lines[idx].mean_score),
            float(lines[idx].min_score),
            len((lines[idx].text or "").strip()),
            idx,
        ),
    )
    return set(ranked[:budget])


def _is_noise_line(
    line: FastOCRLine,
    *,
    min_chars: int,
    max_mean_conf: float,
) -> bool:
    """Tiny / very low-confidence detections that VLM would only hallucinate over."""
    text = (line.text or "").strip()
    if not text:
        return True
    if len(text) < min_chars and float(line.mean_score) < max_mean_conf:
        return True
    return False


def _enforce_vlm_budget(
    lines: list[FastOCRLine],
    escalation_flags: list[tuple[bool, str]],
    *,
    max_calls: int,
    skip_noise: bool,
    noise_min_chars: int,
    noise_max_mean_conf: float,
    disagreements: list[float] | None = None,
    suspicious_flags: list[bool] | None = None,
) -> list[tuple[bool, str]]:
    """Cap VLM escalation and skip noise lines, preserving line order."""
    if not lines:
        return escalation_flags
    out = list(escalation_flags)

    if skip_noise:
        for idx, line in enumerate(lines):
            if not out[idx][0]:
                continue
            if _is_noise_line(
                line,
                min_chars=noise_min_chars,
                max_mean_conf=noise_max_mean_conf,
            ):
                out[idx] = (False, "noise_skip")

    if max_calls <= 0:
        return out

    candidates = [i for i, (needs, _) in enumerate(out) if needs]
    if len(candidates) <= max_calls:
        return out
    def _vlm_sort_key(i: int) -> tuple:
        disagree = disagreements[i] if disagreements and i < len(disagreements) else 0.0
        suspicious = (
            suspicious_flags[i]
            if suspicious_flags and i < len(suspicious_flags)
            else False
        )
        # Worst lines first; suspicious + high disagreement beat mild low-conf.
        priority = 0 if suspicious and disagree >= 0.85 else 1
        return (
            priority,
            float(lines[i].mean_score),
            float(lines[i].min_score),
            -disagree,
            -len((lines[i].text or "").strip()),
        )

    candidates.sort(key=_vlm_sort_key)
    for i in candidates[max_calls:]:
        out[i] = (False, "vlm_budget_exceeded")
    return out


def build_crop_batch(
    lines: list[FastOCRLine],
    escalation_flags: list[tuple[bool, str]],
    *,
    image_size: tuple[int, int],
) -> list[CropBatchItem]:
    """
    Group consecutive lines flagged for escalation into multi-line crops when enabled.
    """
    items: list[CropBatchItem] = []
    i = 0
    merge = settings.ocr_crop_merge_adjacent_lines
    while i < len(lines):
        needs, _ = escalation_flags[i]
        if not needs:
            i += 1
            continue
        indices = [i]
        bboxes = [lines[i].bbox]
        texts = [lines[i].text]
        j = i + 1
        if merge:
            while j < len(lines) and escalation_flags[j][0]:
                indices.append(j)
                bboxes.append(lines[j].bbox)
                texts.append(lines[j].text)
                j += 1
        else:
            j = i + 1
        bbox = union_bbox(bboxes, image_size=image_size)
        items.append(CropBatchItem(
            line_indices=tuple(indices),
            bbox=bbox,
            rapid_texts=tuple(texts),
        ))
        i = j
    return items


CropVlmFn = Callable[[Image.Image], tuple[str, float | None, dict]]


def process_crop_batch(
    image: Image.Image,
    batch: list[CropBatchItem],
    lang: str,
    *,
    gpu_id: int | None,
    allow_vlm: bool,
    vlm_fn: CropVlmFn | None = None,
) -> dict[int, tuple[str, float, str, str | None, bool]]:
    """
    Run crop escalation on batch items.

    Returns map ``line_index → (text, score, engine, variant, vlm_used)``.
    Respects ``ocr_crop_batch_size`` for chunking (sequential today; hook for
    true multi-image VLM batching later).
    """
    if not batch:
        return {}

    batch_size = max(1, settings.ocr_crop_batch_size)
    out: dict[int, tuple[str, float, str, str | None, bool]] = {}
    rgb = image.convert("RGB")

    for chunk_start in range(0, len(batch), batch_size):
        chunk = batch[chunk_start : chunk_start + batch_size]
        for item in chunk:
            crop = crop_line_image(rgb, item.bbox)
            n = len(item.line_indices)
            if n > 1 and allow_vlm:
                try:
                    _vlm_callable = vlm_fn or ocr_crop_vision
                    from app.core.config import settings as _s2
                    if _s2.gpu_scheduler_enabled and vlm_fn is None:
                        from app.services.gpu_scheduler import get_scheduler as _gs
                        _raw = _gs().submit_escalation(
                            _vlm_callable, crop,
                            timeout_ms=_s2.gpu_scheduler_escalation_timeout_ms,
                            fallback=None,
                        )
                        if _raw is None:
                            raise VisionLLMError("escalation timeout/queue-full")
                        vlm_text, _, _ = _raw
                    else:
                        vlm_text, _, _ = _vlm_callable(crop)
                    if _looks_like_unhelpful_vlm(vlm_text):
                        raise VisionLLMError("unhelpful VLM output")
                    split = split_crop_text_for_lines(vlm_text, n)
                    for idx, line_i in enumerate(item.line_indices):
                        text = split[idx] if idx < len(split) else ""
                        score = line_quality_score(mean_conf=0.9, rapid_text=text, tess_text="")
                        out[line_i] = (text, score, "vision_llm", "vlm_multi_line_crop", True)
                    continue
                except VisionLLMError as exc:
                    logger.debug("multi-line crop VLM failed, per-line fallback: %s", exc)

            rapid_fallback = " ".join(t for t in item.rapid_texts if t).strip()
            text, score, engine, variant, vlm_used = escalate_line_crop(
                crop,
                lang,
                gpu_id=gpu_id,
                allow_vlm=allow_vlm,
                rapid_text=rapid_fallback,
            )
            if n == 1:
                out[item.line_indices[0]] = (text, score, engine, variant, vlm_used)
            else:
                split = split_crop_text_for_lines(text, n)
                for idx, line_i in enumerate(item.line_indices):
                    line_text = split[idx] if idx < len(split) else text
                    out[line_i] = (line_text, score, engine, variant, vlm_used)
    return out


def apply_crop_escalation(
    image: Image.Image,
    fast: FastOCRResult,
    lang: str | None = None,
    *,
    allow_vlm: bool = True,
    gpu_id: int | None = None,
    page_quality_poor: bool = False,
    escalation_reason: str | None = None,
) -> CropEscalationResult:
    """
    Score each Rapid line, escalate only bad/suspicious crops, merge in order.
    """
    from app.services.ocr_lang_routing import resolve_tesseract_lang

    profile = current_profile()
    lines = fast.lines
    lang_sample = (fast.text or "").strip() or "\n".join(
        (ln.text or "").strip() for ln in (lines or [])[:32]
    )
    language = lang if lang is not None else resolve_tesseract_lang(lang_sample)
    t0 = time.time()
    engines: set[str] = {"fast_ocr"}
    rgb = image.convert("RGB")
    pad = settings.ocr_line_crop_padding_px

    verification_enabled = settings.ocr_line_verification_enabled
    tess_budget_override: int | None = None
    profile_allow_vlm = True
    max_vlm_calls = 0
    skip_noise = False
    noise_min_chars = 3
    noise_max_mean_conf = 0.55

    if profile is not None:
        if profile.line_verification_enabled is not None:
            verification_enabled = profile.line_verification_enabled
        if profile.tesseract_budget_per_page is not None:
            tess_budget_override = int(profile.tesseract_budget_per_page)
        if profile.allow_crop_vlm is not None:
            profile_allow_vlm = bool(profile.allow_crop_vlm)
        max_vlm_calls = int(profile.max_crop_vlm_calls)
        skip_noise = bool(profile.skip_noise_lines)
        noise_min_chars = int(profile.noise_min_chars)
        noise_max_mean_conf = float(profile.noise_max_mean_conf)

    allow_vlm = allow_vlm and profile_allow_vlm
    verify_with_tesseract = _select_tesseract_verification_indices(
        lines, budget_override=tess_budget_override,
    )

    if not lines:
        crop_vlm_calls = 0
        text = ""
        if allow_vlm:
            try:
                from app.core.config import settings as _s3
                if _s3.gpu_scheduler_enabled:
                    from app.services.gpu_scheduler import get_scheduler as _gs2
                    _r = _gs2().submit_escalation(
                        ocr_crop_vision, rgb,
                        timeout_ms=_s3.gpu_scheduler_escalation_timeout_ms,
                        fallback=None,
                    )
                    if _r is not None:
                        text, _, _ = _r
                        crop_vlm_calls = 1
                        engines.add("vision_llm")
                else:
                    text, _, _ = ocr_crop_vision(rgb)
                    crop_vlm_calls = 1
                    engines.add("vision_llm")
            except VisionLLMError as exc:
                logger.warning("empty page crop VLM failed: %s", exc)
        elapsed = time.time() - t0
        return CropEscalationResult(
            text=text.strip(),
            mean_confidence=fast.mean_score if text.strip() else None,
            engines_used=engines,
            crop_vlm_calls=crop_vlm_calls,
            elapsed_sec=elapsed,
            meta={
                "pipeline": "crop_escalation",
                "empty_detection": True,
                "crop_vlm_calls": crop_vlm_calls,
                "escalation_reason": escalation_reason,
            },
        )

    escalation_flags: list[tuple[bool, str]] = []
    line_states: list[dict[str, Any]] = []

    tess_ran = 0
    suspicious_count = 0
    disagreements: list[float] = []

    for idx, line in enumerate(lines):
        crop = crop_line_image(rgb, line.bbox, pad) if any(line.bbox) else rgb
        if verification_enabled:
            force_skip = idx not in verify_with_tesseract
            verified = verify_line_crop(
                crop,
                line,
                lang=None,
                line_index=idx,
                force_skip_tesseract=force_skip,
            )
            tess = verified.tesseract_text
            disagree = verified.disagreement
            tier = verified.escalation_tier
            reason = verified.escalation_hint
            q_score = verified.quality_score
            suspicious = verified.suspicious
            tess_skipped = verified.tess_skipped
            if not tess_skipped:
                tess_ran += 1
                engines.add("tesseract")
            if suspicious:
                suspicious_count += 1
            disagreements.append(disagree)
        else:
            from app.services.ocr_postprocess import polish_ocr_needs_verification

            skip_tess = (
                line.mean_score >= settings.ocr_line_skip_tesseract_above
                and not polish_ocr_needs_verification(line.text)
            )
            tess = "" if skip_tess else _tesseract_line(crop, None, rapid_text=line.text)
            tess_skipped = skip_tess
            if tess and not skip_tess:
                tess_ran += 1
                engines.add("tesseract")
            q_score = line_quality_score(
                mean_conf=line.mean_score,
                rapid_text=line.text,
                tess_text=tess,
            )
            disagree = 0.0
            tier, reason = classify_line_escalation(
                mean_conf=line.mean_score,
                min_conf=line.min_score,
                rapid_text=line.text,
                tess_text=tess,
                quality_score=q_score,
            )
            suspicious = False

        needs, mode = _line_needs_crop_vlm(
            tier=tier,
            line_mean_score=line.mean_score,
            page_quality_poor=page_quality_poor,
        )

        if (
            suspicious
            and disagree >= 0.85
            and len((tess or "").strip()) < max(4, int(len(line.text or "") * 0.2))
        ):
            tier = "hard"
            reason = reason or "suspicious_high_disagreement"
            needs, mode = _line_needs_crop_vlm(
                tier=tier,
                line_mean_score=line.mean_score,
                page_quality_poor=page_quality_poor,
            )

        escalation_flags.append((needs, mode))
        line_states.append({
            "line": line,
            "tess": tess,
            "disagree": disagree,
            "tier": tier,
            "reason": reason,
            "q_score": q_score,
            "suspicious": suspicious,
            "tess_skipped": tess_skipped,
            "escalation_mode": mode,
        })

    escalation_flags = _enforce_vlm_budget(
        lines,
        escalation_flags,
        max_calls=max_vlm_calls,
        skip_noise=skip_noise,
        noise_min_chars=noise_min_chars,
        noise_max_mean_conf=noise_max_mean_conf,
        disagreements=disagreements,
        suspicious_flags=[bool(s["suspicious"]) for s in line_states],
    )
    batch = build_crop_batch(lines, escalation_flags, image_size=rgb.size)
    escalated_map = process_crop_batch(
        rgb, batch, language, gpu_id=gpu_id, allow_vlm=allow_vlm,
    )
    crop_vlm_calls = sum(1 for v in escalated_map.values() if v[4])

    line_results: list[LineEscalationResult] = []
    for idx, state in enumerate(line_states):
        line = state["line"]
        tess: str = state["tess"]
        tier: str = state["tier"]
        reason: str | None = state["reason"]
        q_score: float = state["q_score"]
        suspicious: bool = state["suspicious"]
        tess_skipped: bool = state["tess_skipped"]
        disagree: float = state["disagree"]
        mode: str = state["escalation_mode"]

        final_text = line.text.strip()
        final_conf = line.mean_score
        engine = "fast_ocr"
        variant_name: str | None = None
        escalated = False

        if idx in escalated_map:
            escalated = True
            final_text, final_conf, engine, variant_name, vlm_used = escalated_map[idx]
            if vlm_used and _vlm_degrades_rapid(line.text, final_text):
                final_text = line.text.strip()
                engine = "fast_ocr"
                vlm_used = False
                escalated = False
            engines.add(engine)
            if vlm_used:
                engines.add("vision_llm")
            elif settings.ocr_consensus_enabled and tess.strip():
                final_text = majority_vote_text([line.text, tess, final_text])
                engine = "consensus"
        elif tier == "light" and allow_vlm and settings.ocr_consensus_enabled and tess.strip():
            final_text = majority_vote_text([line.text, tess])
            engine = "consensus"
            escalated = True
        elif (
            settings.ocr_consensus_enabled
            and tess.strip()
            and disagree > settings.ocr_line_disagreement_light
            and len(tess.strip()) >= max(4, int(len(line.text or "") * 0.2))
        ):
            final_text = majority_vote_text([line.text, tess])
            engine = "consensus"

        from app.services.ocr_postprocess import polish_line_output

        final_text = polish_line_output(final_text, tess)

        line_results.append(LineEscalationResult(
            line_index=idx,
            text=final_text,
            confidence=round(final_conf, 4),
            engine=engine,
            bbox=line.bbox,
            quality_score=round(q_score, 4),
            escalated=escalated,
            disagreement_score=round(disagree, 4),
            suspicious=suspicious,
            rapid_text=line.text.strip(),
            tesseract_text=tess,
            tess_skipped=tess_skipped,
            escalation_hint=reason,
            preprocessing_variant=variant_name,
            escalation_reason=reason or (mode if escalated else None),
            crop_batch_size=len(batch),
        ))

    v_metrics = LineVerificationMetrics(
        total_lines=len(lines),
        tess_ran=tess_ran,
        tess_skipped=len(lines) - tess_ran,
        suspicious_count=suspicious_count,
        mean_disagreement=(
            sum(disagreements) / len(disagreements) if disagreements else 0.0
        ),
        max_disagreement=max(disagreements) if disagreements else 0.0,
        suspicious_threshold=settings.ocr_line_suspicious_threshold,
    )
    if verification_enabled:
        log_verification_metrics(v_metrics, label="crop_escalation")

    layout_meta: dict[str, Any] | None = None
    if profile is not None and profile.layout_reconstruction_enabled and line_results:
        try:
            from app.services.layout import LayoutReconstructionConfig, reconstruct_layout
            from app.services.layout.reading_order import lines_from_reading_order

            layout_cfg = LayoutReconstructionConfig(
                enable_barcodes=profile.layout_detect_barcodes,
                enable_form_detection=profile.layout_detect_forms,
                enable_columns=profile.layout_detect_columns,
                enable_handwriting_flag=profile.layout_flag_handwriting,
                enable_duplicate_suppression=profile.layout_dedup_enabled,
            )
            items = [(r.text, r.bbox, r.confidence) for r in line_results]
            layout = reconstruct_layout(items, rgb, config=layout_cfg)
            ordered = layout.ordered_indices
            if ordered:
                # Update suppressed/handwriting flags on line_results
                suppressed_set = set(layout.suppressed_indices)
                for box in layout.boxes:
                    if box.index >= len(line_results):
                        continue
                    if box.is_handwriting:
                        line_results[box.index].escalation_hint = (
                            (line_results[box.index].escalation_hint or "")
                            + ";handwriting"
                        ).strip(";")
                reordered_lines = [line_results[i] for i in ordered]
                merged_lines = lines_from_reading_order(
                    layout.boxes,
                    layout.ordered_indices,
                    separator=settings.fast_line_separator,
                )
                merged_text = "\n".join(line for line in merged_lines if line.strip())
                line_results = reordered_lines
                layout_meta = {
                    "features": layout.features.to_dict(),
                    "barcode_regions": layout.barcode_regions,
                    "decoded_barcodes": layout.decoded_barcodes,
                    "suppressed_indices": layout.suppressed_indices,
                    "ordered_count": len(ordered),
                    "input_count": len(items),
                }
                engines.add("layout")
        except Exception as exc:  # noqa: BLE001
            logger.warning("layout reconstruction failed: %s", exc)
            merged_text = "\n".join(r.text for r in line_results if r.text)
    else:
        merged_text = "\n".join(r.text for r in line_results if r.text)

    mean_conf = (
        sum(r.confidence for r in line_results) / len(line_results)
        if line_results else fast.mean_score
    )
    elapsed = time.time() - t0

    debug_payload = [
        {
            "text": r.text,
            "confidence": r.confidence,
            "engine": r.engine,
            "bbox": list(r.bbox),
            "quality_score": r.quality_score,
            "escalated": r.escalated,
            "disagreement_score": r.disagreement_score,
            "suspicious": r.suspicious,
            "rapid_text": r.rapid_text,
            "tesseract_text": r.tesseract_text,
            "tess_skipped": r.tess_skipped,
            "escalation_hint": r.escalation_hint,
            "preprocessing_variant": r.preprocessing_variant,
            "escalation_reason": r.escalation_reason,
        }
        for r in line_results
    ]

    return CropEscalationResult(
        text=merged_text,
        mean_confidence=round(mean_conf, 4),
        line_results=line_results,
        engines_used=engines,
        crop_vlm_calls=crop_vlm_calls,
        elapsed_sec=elapsed,
        verification_metrics=v_metrics,
        meta={
            "engine": "crop_escalation",
            "pipeline": "crop_escalation",
            "model": f"PP-OCRv5/{settings.fast_lang_type}/{settings.fast_model_type}",
            "mean_score": round(mean_conf, 4),
            "n_lines": len(line_results),
            "n_boxes": fast.n_boxes,
            "lines": line_dicts_from_line_smart_debug(debug_payload, page=1),
            "line_verification": v_metrics.to_dict(),
            "line_verification_tess_budget": int(
                tess_budget_override
                if tess_budget_override is not None
                else settings.ocr_line_tesseract_max_lines_per_page
            ),
            "line_verification_tess_selected": len(verify_with_tesseract),
            "pipeline_profile": profile.name if profile is not None else None,
            "max_crop_vlm_calls": max_vlm_calls,
            "layout": layout_meta,
            "suspicious_line_count": suspicious_count,
            "crop_vlm_calls": crop_vlm_calls,
            "crop_batch_count": len(batch),
            "elapsed_sec": round(elapsed, 3),
            "engines": sorted(engines),
            "gpu_id": gpu_id,
            "page_quality_poor": page_quality_poor,
            "escalation_reason": escalation_reason,
            "page_vision_escalation": False,
        },
    )

