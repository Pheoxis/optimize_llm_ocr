"""
Automatic pipeline profiles per source (JPG / PDF / ZIP / ZIP member).

Goal: hands-off optimal latency/quality trade-off without the client tuning env.
A thread-local *active profile* tightens or relaxes verification / VLM use
inside ``apply_crop_escalation`` without mutating global settings.

Profiles never *replace* settings; they only narrow them (faster path).
Falling back to settings keeps existing behavior when no profile is set.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

__all__ = [
    "PipelineProfile",
    "IMAGE_PROFILE",
    "PDF_CROP_PROFILE",
    "PDF_LINE_PROFILE",
    "ZIP_MEMBER_PROFILE",
    "PREMIUM_PROFILE",
    "PROFILES",
    "use_profile",
    "current_profile",
    "profile_for_source",
]


@dataclass(frozen=True)
class PipelineProfile:
    """Per-source pipeline tuning. ``None`` fields fall back to ``settings``."""

    name: str
    line_verification_enabled: bool | None = None
    tesseract_budget_per_page: int | None = None  # 0 = unlimited (verify all)
    skip_high_conf_tesseract: bool | None = None
    allow_crop_vlm: bool | None = None
    max_crop_vlm_calls: int = 0  # 0 = unlimited
    merge_adjacent_crops: bool | None = None
    skip_noise_lines: bool = True  # drop tiny low-conf detections from escalation
    noise_min_chars: int = 3
    noise_max_mean_conf: float = 0.55
    # Form-aware layout reconstruction
    layout_reconstruction_enabled: bool = False
    layout_detect_barcodes: bool = True
    layout_detect_forms: bool = True
    layout_detect_columns: bool = True
    layout_flag_handwriting: bool = True
    layout_dedup_enabled: bool = True


# Single photo / phone receipt: keep quality, cap VLM, allow verification
IMAGE_PROFILE = PipelineProfile(
    name="image",
    line_verification_enabled=True,
    tesseract_budget_per_page=10,
    skip_high_conf_tesseract=True,
    allow_crop_vlm=True,
    max_crop_vlm_calls=2,
    merge_adjacent_crops=True,
    layout_reconstruction_enabled=True,
)

# Clean rendered PDF crop (RapidOCR is already accurate here): skip verification + VLM
PDF_CROP_PROFILE = PipelineProfile(
    name="pdf_crop",
    line_verification_enabled=False,
    tesseract_budget_per_page=0,
    skip_high_conf_tesseract=True,
    allow_crop_vlm=False,
    max_crop_vlm_calls=0,
    merge_adjacent_crops=True,
    layout_reconstruction_enabled=True,
    layout_detect_forms=False,
    layout_detect_columns=True,
    layout_flag_handwriting=False,
    layout_dedup_enabled=True,
)

# PDF page promoted to line-smart (bad crop): light verification, tight VLM
PDF_LINE_PROFILE = PipelineProfile(
    name="pdf_line",
    line_verification_enabled=True,
    tesseract_budget_per_page=8,
    skip_high_conf_tesseract=True,
    allow_crop_vlm=True,
    max_crop_vlm_calls=2,
    merge_adjacent_crops=True,
    layout_reconstruction_enabled=True,
)

# ZIP member: same dedup/layout rules as other sources; skip heavy CV + VLM
ZIP_MEMBER_PROFILE = PipelineProfile(
    name="zip_member",
    line_verification_enabled=False,
    tesseract_budget_per_page=0,
    skip_high_conf_tesseract=True,
    allow_crop_vlm=False,
    max_crop_vlm_calls=0,
    merge_adjacent_crops=True,
    layout_reconstruction_enabled=True,
    layout_detect_barcodes=False,
    layout_detect_forms=False,
    layout_detect_columns=True,
    layout_flag_handwriting=False,
    layout_dedup_enabled=True,
)

# Adaptive routing tier-3 (score 80-100): all features enabled, generous VLM budget.
# This mirrors IMAGE_PROFILE but removes VLM cap and enables layout reconstruction
# with all detectors active.  Tesseract budget is unlimited (0).
PREMIUM_PROFILE = PipelineProfile(
    name="premium",
    line_verification_enabled=True,
    tesseract_budget_per_page=0,       # unlimited
    skip_high_conf_tesseract=False,    # verify every suspicious line
    allow_crop_vlm=True,
    max_crop_vlm_calls=0,              # unlimited — very hard pages need it
    merge_adjacent_crops=True,
    layout_reconstruction_enabled=True,
    layout_detect_barcodes=True,
    layout_detect_forms=True,
    layout_detect_columns=True,
    layout_flag_handwriting=True,
    layout_dedup_enabled=True,
)

PROFILES: dict[str, PipelineProfile] = {
    p.name: p
    for p in (IMAGE_PROFILE, PDF_CROP_PROFILE, PDF_LINE_PROFILE, ZIP_MEMBER_PROFILE, PREMIUM_PROFILE)
}


_state = threading.local()


def current_profile() -> PipelineProfile | None:
    return getattr(_state, "profile", None)


def _set(profile: PipelineProfile | None) -> None:
    _state.profile = profile


@contextmanager
def use_profile(profile: PipelineProfile | None) -> Iterator[PipelineProfile | None]:
    """Activate a profile for the calling thread (nestable)."""
    prev = current_profile()
    _set(profile)
    try:
        yield profile
    finally:
        _set(prev)


def profile_for_source(
    source_kind: str,
    *,
    in_zip: bool = False,
    pdf_crop_line_smart: bool = False,
) -> PipelineProfile:
    """
    Map source kind → optimal profile.

    ``source_kind``: ``"image"`` | ``"pdf"`` | ``"zip"``
    ``in_zip``: True when called from a ZIP member worker.
    ``pdf_crop_line_smart``: True when a PDF crop was promoted to line-smart.
    """
    if in_zip:
        return ZIP_MEMBER_PROFILE
    if pdf_crop_line_smart:
        return PDF_LINE_PROFILE
    if source_kind == "pdf":
        return PDF_CROP_PROFILE
    return IMAGE_PROFILE
