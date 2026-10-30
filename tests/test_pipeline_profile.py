"""Per-source pipeline profiles + VLM/noise budget enforcement."""

from __future__ import annotations

from unittest.mock import patch

from PIL import Image

from app.core.config import settings
from app.services.crop_escalation import (
    _enforce_vlm_budget,
    _is_noise_line,
    apply_crop_escalation,
)
from app.services.fast_ocr import FastOCRLine, FastOCRResult
from app.services.pipeline_profile import (
    IMAGE_PROFILE,
    PDF_CROP_PROFILE,
    PDF_LINE_PROFILE,
    ZIP_MEMBER_PROFILE,
    current_profile,
    profile_for_source,
    use_profile,
)


def test_profile_for_source_routing() -> None:
    assert profile_for_source("image").name == IMAGE_PROFILE.name
    assert profile_for_source("pdf").name == PDF_CROP_PROFILE.name
    assert profile_for_source("pdf", pdf_crop_line_smart=True).name == PDF_LINE_PROFILE.name
    assert profile_for_source("image", in_zip=True).name == ZIP_MEMBER_PROFILE.name
    assert profile_for_source("pdf", in_zip=True).name == ZIP_MEMBER_PROFILE.name


def test_use_profile_context_manager() -> None:
    assert current_profile() is None
    with use_profile(IMAGE_PROFILE):
        assert current_profile() is IMAGE_PROFILE
        with use_profile(PDF_CROP_PROFILE):
            assert current_profile() is PDF_CROP_PROFILE
        assert current_profile() is IMAGE_PROFILE
    assert current_profile() is None


def test_is_noise_line_filters_short_low_conf() -> None:
    line = FastOCRLine("CO", 0.5, 0.4, 1, (0, 0, 10, 10))
    assert _is_noise_line(line, min_chars=3, max_mean_conf=0.55)


def test_is_noise_line_keeps_real_text() -> None:
    line = FastOCRLine("Hello world", 0.4, 0.3, 1, (0, 0, 10, 10))
    assert not _is_noise_line(line, min_chars=3, max_mean_conf=0.55)


def test_enforce_vlm_budget_skips_noise_and_caps() -> None:
    lines = [
        FastOCRLine("CO", 0.5, 0.4, 1, (0, 0, 10, 10)),
        FastOCRLine("Faktura VAT 1/2026", 0.3, 0.2, 1, (0, 0, 200, 20)),
        FastOCRLine("Adres dostawy", 0.4, 0.3, 1, (0, 0, 200, 20)),
        FastOCRLine("Pozycja 1", 0.6, 0.5, 1, (0, 0, 100, 20)),
    ]
    flags = [(True, "hard")] * 4
    out = _enforce_vlm_budget(
        lines,
        flags,
        max_calls=2,
        skip_noise=True,
        noise_min_chars=3,
        noise_max_mean_conf=0.55,
    )
    kept = [i for i, (needs, _) in enumerate(out) if needs]
    assert 0 not in kept
    assert len(kept) == 2
    assert 1 in kept and 2 in kept


def test_apply_crop_escalation_image_profile_caps_vlm() -> None:
    lines = [
        FastOCRLine(f"line{i}", 0.30 + i * 0.01, 0.2, 1, (0, i * 12, 100, i * 12 + 10))
        for i in range(8)
    ]
    fast = FastOCRResult(
        text="\n".join(line.text for line in lines),
        lines=lines,
        mean_score=0.35,
        min_score=0.2,
        low_conf_ratio=1.0,
        n_boxes=8,
        elapsed_sec=0.1,
    )
    image = Image.new("RGB", (200, 200), color=(255, 255, 255))
    captured: dict[str, int] = {"calls": 0}

    def _fake_process_crop_batch(_image, batch, _lang, **_kwargs):
        out = {}
        for item in batch:
            captured["calls"] += 1
            for line_i in item.line_indices:
                out[line_i] = ("FIX", 0.95, "vision_llm", "vlm_crop", True)
        return out

    with (
        patch(
            "app.services.crop_escalation.process_crop_batch",
            side_effect=_fake_process_crop_batch,
        ),
        patch.object(settings, "ocr_line_verification_enabled", False),
        patch.object(settings, "ocr_consensus_enabled", False),
        use_profile(IMAGE_PROFILE),
    ):
        result = apply_crop_escalation(
            image, fast, "eng", allow_vlm=True, page_quality_poor=True,
        )

    assert captured["calls"] <= IMAGE_PROFILE.max_crop_vlm_calls
    assert result.crop_vlm_calls <= IMAGE_PROFILE.max_crop_vlm_calls
    assert result.meta.get("pipeline_profile") == IMAGE_PROFILE.name


def test_apply_crop_escalation_pdf_crop_profile_disables_vlm() -> None:
    lines = [
        FastOCRLine("low conf line", 0.20, 0.1, 1, (0, 0, 100, 10)),
    ]
    fast = FastOCRResult(
        text="low conf line",
        lines=lines,
        mean_score=0.2,
        min_score=0.1,
        low_conf_ratio=1.0,
        n_boxes=1,
        elapsed_sec=0.1,
    )
    image = Image.new("RGB", (200, 200), color=(255, 255, 255))

    captured: dict[str, bool] = {"allow_vlm": True}

    def _fake_process_crop_batch(_image, _batch, _lang, **kwargs):
        captured["allow_vlm"] = bool(kwargs.get("allow_vlm"))
        return {}

    with (
        patch(
            "app.services.crop_escalation.process_crop_batch",
            side_effect=_fake_process_crop_batch,
        ),
        patch.object(settings, "ocr_line_verification_enabled", True),
        patch.object(settings, "ocr_consensus_enabled", False),
        patch(
            "app.services.ocr_line_verification.tesseract_line_crop",
            return_value="",
        ),
        use_profile(PDF_CROP_PROFILE),
    ):
        result = apply_crop_escalation(
            image, fast, "eng", allow_vlm=True, page_quality_poor=True,
        )

    assert captured["allow_vlm"] is False
    assert result.crop_vlm_calls == 0
    assert result.meta.get("pipeline_profile") == PDF_CROP_PROFILE.name
    assert result.meta["line_verification"]["tess_ran"] == 0
