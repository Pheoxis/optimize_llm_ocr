"""
Disk-backed OCR result cache (repeated forms / identical crops).

Cache key: SHA1(image_bytes + preprocessing_variant).
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.json"
_LOCK_NAME = ".manifest.lock"


@dataclass
class OcrCacheEntry:
    """Cached OCR output for one image + preprocessing variant."""

    text: str
    confidence: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    engine: str | None = None
    lines: list[dict[str, Any]] | None = None
    min_score: float | None = None
    low_conf_ratio: float | None = None
    n_boxes: int | None = None
    variant: str = ""
    created_at: str = ""
    accessed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OcrCacheEntry:
        return cls(
            text=str(data.get("text", "") or ""),
            confidence=data.get("confidence"),
            metadata=dict(data.get("metadata") or {}),
            engine=data.get("engine"),
            lines=data.get("lines"),
            min_score=data.get("min_score"),
            low_conf_ratio=data.get("low_conf_ratio"),
            n_boxes=data.get("n_boxes"),
            variant=str(data.get("variant", "") or ""),
            created_at=str(data.get("created_at", "") or ""),
            accessed_at=str(data.get("accessed_at", "") or ""),
        )


@dataclass
class OcrCacheMetrics:
    """In-process cache hit/miss counters (aggregated across workers per process)."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "hit_rate": round(
                self.hits / max(self.hits + self.misses, 1),
                4,
            ),
        }


_metrics = OcrCacheMetrics()
_metrics_lock = threading.Lock()


def get_ocr_cache_metrics() -> OcrCacheMetrics:
    with _metrics_lock:
        return OcrCacheMetrics(
            hits=_metrics.hits,
            misses=_metrics.misses,
            stores=_metrics.stores,
            evictions=_metrics.evictions,
        )


def reset_ocr_cache_metrics() -> None:
    global _metrics
    with _metrics_lock:
        _metrics = OcrCacheMetrics()


def log_ocr_cache_metrics(*, label: str = "") -> None:
    m = get_ocr_cache_metrics()
    prefix = f"{label}: " if label else ""
    logger.info(
        "%sOCR cache — hits=%d misses=%d stores=%d evictions=%d hit_rate=%.2f",
        prefix,
        m.hits,
        m.misses,
        m.stores,
        m.evictions,
        m.hits / max(m.hits + m.misses, 1),
    )


def _bump(**kwargs: int) -> None:
    with _metrics_lock:
        for key, delta in kwargs.items():
            setattr(_metrics, key, getattr(_metrics, key) + delta)


def image_bytes_for_cache(image: Image.Image) -> bytes:
    """Stable RGB payload for cache keys (lossless PNG)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def file_bytes_for_cache(path: Path) -> bytes:
    return path.read_bytes()


def cache_key_sha1(image_bytes: bytes, preprocess_variant: str) -> str:
    digest = hashlib.sha1()
    digest.update(image_bytes)
    digest.update(preprocess_variant.encode("utf-8"))
    return digest.hexdigest()


def fast_ocr_preprocess_variant(preprocess_variant: str = "original") -> str:
    """Variant string including RapidOCR model fingerprint."""
    from app.services.ocr_dedup import OCR_DEDUP_CACHE_REVISION

    base = (
        f"fast_ocr:{settings.fast_ocr_version}:"
        f"{settings.fast_lang_type}:{settings.fast_model_type}:"
        f"dedup:{OCR_DEDUP_CACHE_REVISION}"
    )
    if preprocess_variant and preprocess_variant != "original":
        return f"{base}:{preprocess_variant}"
    return base


def vision_preprocess_variant(model: str, *, max_side: int) -> str:
    return f"vision_llm:{model}:side{max_side}"


def _cache_dir() -> Path:
    root = settings.ocr_cache_dir
    if not root.is_absolute():
        root = settings.project_root / root
    return root


def _entry_path(key: str) -> Path:
    return _cache_dir() / key[:2] / f"{key}.json"


def _manifest_path() -> Path:
    return _cache_dir() / _MANIFEST_NAME


def _lock_path() -> Path:
    return _cache_dir() / _LOCK_NAME


class _ManifestLock:
    def __enter__(self) -> None:
        self._cache_dir().mkdir(parents=True, exist_ok=True)
        self._fh = open(_lock_path(), "a+", encoding="utf-8")
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)

    def __exit__(self, *args: object) -> None:
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        self._fh.close()

    @staticmethod
    def _cache_dir() -> Path:
        return _cache_dir()


def _load_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if not path.is_file():
        return {"entries": {}, "total_bytes": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"entries": {}, "total_bytes": 0}
    if "entries" not in data:
        data["entries"] = {}
    data["total_bytes"] = int(data.get("total_bytes") or 0)
    return data


def _save_manifest(data: dict[str, Any]) -> None:
    path = _manifest_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _evict_if_needed(manifest: dict[str, Any]) -> int:
    limit = settings.ocr_cache_max_bytes
    entries: dict[str, dict[str, Any]] = manifest["entries"]
    evicted = 0
    while manifest["total_bytes"] > limit and entries:
        oldest_key = min(
            entries.keys(),
            key=lambda k: entries[k].get("accessed_at", entries[k].get("created_at", "")),
        )
        entry_meta = entries.pop(oldest_key)
        manifest["total_bytes"] = max(
            0,
            manifest["total_bytes"] - int(entry_meta.get("bytes", 0)),
        )
        path = _entry_path(oldest_key)
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
        parent = path.parent
        try:
            if parent != _cache_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass
        evicted += 1
    return evicted


def lookup(key: str) -> OcrCacheEntry | None:
    if not settings.ocr_cache_enabled:
        return None
    path = _entry_path(key)
    if not path.is_file():
        _bump(misses=1)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = OcrCacheEntry.from_dict(data)
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("OCR cache read failed %s: %s", key[:8], exc)
        _bump(misses=1)
        return None

    now = datetime.now(UTC).isoformat()
    with _ManifestLock():
        manifest = _load_manifest()
        meta = manifest["entries"].get(key)
        if meta is not None:
            meta["accessed_at"] = now
            meta["bytes"] = path.stat().st_size
            _save_manifest(manifest)

    entry.accessed_at = now
    _bump(hits=1)
    return entry


def store(key: str, entry: OcrCacheEntry) -> None:
    if not settings.ocr_cache_enabled:
        return

    now = datetime.now(UTC).isoformat()
    entry.created_at = entry.created_at or now
    entry.accessed_at = now

    path = _entry_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(entry.to_dict(), ensure_ascii=False)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    size = path.stat().st_size

    with _ManifestLock():
        manifest = _load_manifest()
        old = manifest["entries"].get(key)
        if old is not None:
            manifest["total_bytes"] = max(
                0,
                manifest["total_bytes"] - int(old.get("bytes", 0)),
            )
        manifest["entries"][key] = {
            "bytes": size,
            "created_at": entry.created_at,
            "accessed_at": entry.accessed_at,
            "variant": entry.variant,
        }
        manifest["total_bytes"] += size
        evicted = _evict_if_needed(manifest)
        _save_manifest(manifest)

    _bump(stores=1, evictions=evicted)


def get_or_compute(
    image_bytes: bytes,
    preprocess_variant: str,
    compute: Callable[[], OcrCacheEntry],
) -> tuple[OcrCacheEntry, bool]:
    """
    Return ``(entry, cache_hit)``.

    *compute* runs only on miss.
    """
    if not settings.ocr_cache_enabled:
        return compute(), False

    key = cache_key_sha1(image_bytes, preprocess_variant)
    hit = lookup(key)
    if hit is not None:
        return hit, True

    entry = compute()
    entry.variant = preprocess_variant
    store(key, entry)
    return entry, False


def clear_ocr_cache() -> None:
    """Remove all cached entries (tests / admin)."""
    root = _cache_dir()
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
