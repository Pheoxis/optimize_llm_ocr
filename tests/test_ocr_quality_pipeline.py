"""Tests for global OCR quality pipeline."""

from __future__ import annotations

from app.services.ocr_quality_pipeline import (
    QualityPipelineSession,
    QualityTier,
    assess_ocr_output,
    attach_quality_to_meta,
    plan_recovery_actions,
    should_escalate_for_quality,
    tier_for_score,
    use_quality_session,
)


def test_tier_thresholds() -> None:
    assert tier_for_score(95) == QualityTier.EXCELLENT
    assert tier_for_score(80) == QualityTier.ACCEPTABLE
    assert tier_for_score(60) == QualityTier.SUSPICIOUS
    assert tier_for_score(30) == QualityTier.POOR


def test_excellent_invoice_text() -> None:
    text = (
        "FAKTURA VAT\n"
        "SPRZEDAWCA: ABC sp. z o.o.\n"
        "NIP: 123-456-78-90\n"
        "NABYWCA: XYZ Corp\n"
        "Kwota netto: 1 000,00 zł\n"
        "Kwota brutto: 1 230,00 zł\n"
    )
    meta = {"mean_score": 0.95, "min_score": 0.88, "low_conf_ratio": 0.05, "n_boxes": 20}
    assessment = assess_ocr_output(text, meta)
    assert assessment.quality_score >= 75
    assert assessment.tier in (QualityTier.EXCELLENT, QualityTier.ACCEPTABLE)


def test_poor_garbage_text() -> None:
    text = "\n".join([
        "16 Z FZ COO",
        "|||| [?][?][?]",
        "przesynç xyzqwt",
        "16 Z FZ COO",
    ] * 3)
    meta = {"mean_score": 0.45, "min_score": 0.20, "low_conf_ratio": 0.55, "n_boxes": 8}
    assessment = assess_ocr_output(text, meta)
    assert assessment.quality_score < 75
    assert assessment.tier in (QualityTier.SUSPICIOUS, QualityTier.POOR)
    assert assessment.needs_recovery
    assert len(assessment.recommended_actions) >= 1


def test_duplicate_lines_penalized() -> None:
    text = "kwota pobrania\nkwota pobrania\nkwota pobrania\ninne należności"
    meta = {"mean_score": 0.90, "low_conf_ratio": 0.05, "n_boxes": 10}
    assessment = assess_ocr_output(text, meta)
    assert assessment.dimensions["duplicated_lines"] < 80


def test_recovery_plan_for_poor() -> None:
    text = "16 Z\nFZ gr COO\n[?][?][?]\n||||"
    meta = {"mean_score": 0.40, "low_conf_ratio": 0.60, "n_boxes": 5}
    assessment = assess_ocr_output(text, meta)
    actions = plan_recovery_actions(assessment, meta)
    assert len(actions) >= 1


def test_should_escalate_for_quality() -> None:
    text = "garbage " * 20
    meta = {"mean_score": 0.30, "low_conf_ratio": 0.70, "n_boxes": 4}
    assessment = assess_ocr_output(text, meta)
    assert should_escalate_for_quality(assessment) or assessment.tier == QualityTier.SUSPICIOUS


def test_attach_quality_to_meta() -> None:
    meta: dict = {"mean_score": 0.92, "engine": "fast_ocr", "n_boxes": 15}
    text = "Paragon\nRAZEM: 50,00 zł"
    with use_quality_session() as session:
        attach_quality_to_meta(meta, text)
        assert "quality_score" in meta
        assert "quality_tier" in meta
        assert "quality" in meta
        assert session.attempts >= 1
        assert len(session.retry_history) >= 1


def test_session_degradation_trace() -> None:
    session = QualityPipelineSession()
    from app.services.ocr_quality_pipeline import QualityAssessment

    good = QualityAssessment(quality_score=85, tier=QualityTier.ACCEPTABLE, dimensions={})
    bad = QualityAssessment(quality_score=40, tier=QualityTier.POOR, dimensions={})
    session.record_assessment(good, text="good text")
    session.record_assessment(bad, text="bad text")
    assert len(session.degradation_trace) >= 1
    assert session.best_score == 85.0


def test_session_escalation_trace() -> None:
    session = QualityPipelineSession()
    session.record_escalation(
        from_engine="fast_ocr",
        to_engine="minicpm",
        reason="crop_vlm",
        lines_escalated=3,
    )
    assert len(session.escalation_trace) == 1
    assert session.escalation_trace[0]["lines_escalated"] == 3


def test_assessment_to_dict() -> None:
    text = "Test dokumentu"
    meta = {"mean_score": 0.88, "n_boxes": 5}
    assessment = assess_ocr_output(text, meta)
    d = assessment.to_dict()
    for key in ("quality_score", "quality_tier", "dimensions", "issues", "analysis_ms"):
        assert key in d
