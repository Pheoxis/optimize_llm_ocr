"""Disk-backed OCR cache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.core.config import settings
from app.services.ocr_cache import (
    OcrCacheEntry,
    cache_key_sha1,
    clear_ocr_cache,
    get_or_compute,
    get_ocr_cache_metrics,
    image_bytes_for_cache,
    lookup,
    reset_ocr_cache_metrics,
    store,
)
from app.services.ocr_cache_integration import cached_ocr_pil_fast


def test_cache_key_stable() -> None:
    img = Image.new("RGB", (10, 10), color=(1, 2, 3))
    b = image_bytes_for_cache(img)
    k1 = cache_key_sha1(b, "fast_ocr:v1")
    k2 = cache_key_sha1(b, "fast_ocr:v1")
    k3 = cache_key_sha1(b, "fast_ocr:v2")
    assert k1 == k2
    assert k1 != k3


def test_store_lookup_and_metrics(tmp_path: Path) -> None:
    clear_ocr_cache()
    reset_ocr_cache_metrics()
    with (
        patch.object(settings, "ocr_cache_dir", tmp_path / "cache"),
        patch.object(settings, "ocr_cache_enabled", True),
        patch.object(settings, "ocr_cache_max_bytes", 10_000_000),
    ):
        key = "abc123"
        entry = OcrCacheEntry(
            text="hello",
            confidence=0.9,
            metadata={"engine": "test"},
            variant="v1",
        )
        store(key, entry)
        hit = lookup(key)
        assert hit is not None
        assert hit.text == "hello"
        m = get_ocr_cache_metrics()
        assert m.stores == 1
        assert m.hits == 1


def test_get_or_compute_hit(tmp_path: Path) -> None:
    clear_ocr_cache()
    reset_ocr_cache_metrics()
    img = Image.new("RGB", (12, 12), color=(255, 0, 0))
    img_bytes = image_bytes_for_cache(img)
    calls = {"n": 0}

    def compute() -> OcrCacheEntry:
        calls["n"] += 1
        return OcrCacheEntry(text="computed", confidence=0.5, metadata={})

    with (
        patch.object(settings, "ocr_cache_dir", tmp_path / "cache"),
        patch.object(settings, "ocr_cache_enabled", True),
        patch.object(settings, "ocr_cache_max_bytes", 10_000_000),
    ):
        e1, hit1 = get_or_compute(img_bytes, "variant_a", compute)
        e2, hit2 = get_or_compute(img_bytes, "variant_a", compute)

    assert e1.text == "computed"
    assert hit1 is False
    assert hit2 is True
    assert calls["n"] == 1
    assert get_ocr_cache_metrics().hits == 1


def test_eviction_when_over_limit(tmp_path: Path) -> None:
    clear_ocr_cache()
    reset_ocr_cache_metrics()
    with (
        patch.object(settings, "ocr_cache_dir", tmp_path / "cache"),
        patch.object(settings, "ocr_cache_enabled", True),
        patch.object(settings, "ocr_cache_max_bytes", 200),
    ):
        store("key_a", OcrCacheEntry(text="a" * 50, confidence=1.0, metadata={}))
        store("key_b", OcrCacheEntry(text="b" * 50, confidence=1.0, metadata={}))
        store("key_c", OcrCacheEntry(text="c" * 50, confidence=1.0, metadata={}))
        assert lookup("key_a") is None or lookup("key_b") is None
        assert get_ocr_cache_metrics().evictions >= 1


def test_cached_ocr_pil_fast_wrapper(tmp_path: Path) -> None:
    from app.services.fast_ocr import FastOCRResult

    clear_ocr_cache()
    reset_ocr_cache_metrics()
    img = Image.new("RGB", (8, 8), color=(0, 255, 0))
    calls = {"n": 0}

    def fake_compute() -> FastOCRResult:
        calls["n"] += 1
        return FastOCRResult(
            text="LINE",
            lines=[],
            mean_score=0.88,
            min_score=0.8,
            low_conf_ratio=0.0,
            n_boxes=1,
            elapsed_sec=0.01,
        )

    with (
        patch.object(settings, "ocr_cache_dir", tmp_path / "cache"),
        patch.object(settings, "ocr_cache_enabled", True),
        patch.object(settings, "ocr_cache_max_bytes", 10_000_000),
    ):
        r1 = cached_ocr_pil_fast(img, fake_compute, preprocess_variant="original")
        r2 = cached_ocr_pil_fast(img, fake_compute, preprocess_variant="original")

    assert r1.text == "LINE"
    assert r2.text == "LINE"
    assert calls["n"] == 1
