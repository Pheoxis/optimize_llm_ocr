"""
Fast OCR engine — RapidOCR (PP-OCRv5 Latin) on GPU.

Sub-second OCR for receipts, invoices, scanned forms.  Multilingual (PL/EN/ES
via the Latin recognition model).  Used as the primary engine in ``smart``
mode; the result is escalated to the vision LLM only when its quality gate
fails.

Hardware
--------
Designed for one CUDA GPU (RTX PRO 6000 in our case).  Falls back to CPU
automatically if onnxruntime cannot load CUDA libraries.

CUDA bootstrap
--------------
onnxruntime-gpu 1.26 ships against CUDA 12.x but our PyTorch is built
against CUDA 13.  The CUDA 12 runtime libs are however already present in
the venv under ``site-packages/nvidia/...`` (pulled by other packages).
We pre-load the few ``.so`` files onnxruntime needs with ``RTLD_GLOBAL``
so its dynamic linker finds them at import time.  No env vars required.
"""

from __future__ import annotations

import ctypes
import logging
import os
import site
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── CUDA-12 shared library bootstrap (must run BEFORE `import rapidocr`) ─────

_CUDA12_LIBS: tuple[str, ...] = (
    "nvidia/cublas/lib/libcublas.so.12",
    "nvidia/cublas/lib/libcublasLt.so.12",
    "nvidia/cuda_runtime/lib/libcudart.so.12",
    "nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
    "nvidia/cudnn/lib/libcudnn.so.9",
    "nvidia/cufft/lib/libcufft.so.11",
    "nvidia/curand/lib/libcurand.so.10",
    "nvidia/cusparse/lib/libcusparse.so.12",
    "nvidia/nccl/lib/libnccl.so.2",
)


def _nvidia_lib_dirs() -> list[str]:
    """Directories with CUDA 12 .so files shipped inside the venv (nvidia-* wheels)."""
    dirs: list[str] = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_root = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for pkg in os.listdir(nvidia_root):
            lib_dir = os.path.join(nvidia_root, pkg, "lib")
            if os.path.isdir(lib_dir):
                dirs.append(lib_dir)
    return dirs


def _prepend_ld_library_path(dirs: Iterable[str]) -> None:
    """Expose venv CUDA libs to onnxruntime's CUDA provider loader."""
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [d for d in dirs if d and os.path.isdir(d)]
    if not parts:
        return
    merged: list[str] = []
    for d in parts + ([existing] if existing else []):
        if d not in merged:
            merged.append(d)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


def _bootstrap_cuda12() -> bool:
    """Pre-load CUDA 12 shared libraries from venv site-packages.

    Sets ``LD_LIBRARY_PATH`` and ``RTLD_GLOBAL`` dlopen so
    ``libonnxruntime_providers_cuda.so`` finds libcublasLt/cudnn at runtime.

    Returns True if at least the core libs (cublasLt, cudart, cudnn) loaded.
    """
    lib_dirs = _nvidia_lib_dirs()
    _prepend_ld_library_path(lib_dirs)

    loaded: list[str] = []
    candidates: list[str] = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for rel in _CUDA12_LIBS:
            candidates.append(os.path.join(sp, rel))
    seen: set[str] = set()
    for path in candidates:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            loaded.append(os.path.basename(path))
        except OSError as exc:
            logger.debug("CUDA lib %s not loadable: %s", path, exc)
    core = {"libcublasLt.so.12", "libcudart.so.12", "libcudnn.so.9"}
    have_core = core.issubset(set(loaded))
    if have_core:
        logger.info("Loaded CUDA-12 libs for onnxruntime: %s", ", ".join(loaded))
    else:
        logger.warning(
            "CUDA-12 libs missing (%s); RapidOCR will run on CPU",
            core - set(loaded),
        )
    return have_core


_cuda_ready = _bootstrap_cuda12()


# ── lazy RapidOCR instance per GPU ───────────────────────────────────────────

_engines: dict[int, object] = {}
_engine_lock = threading.Lock()
_inference_locks: dict[int, threading.Lock] = {}


def _inference_lock(gpu_id: int) -> threading.Lock:
    """Serialize ONNX Runtime calls per GPU (ORT sessions are not thread-safe)."""
    with _engine_lock:
        if gpu_id not in _inference_locks:
            _inference_locks[gpu_id] = threading.Lock()
        return _inference_locks[gpu_id]


def cuda_runtime_status() -> dict:
    """Diagnostics: whether ONNX Runtime can use CUDA on this machine."""
    status: dict = {
        "bootstrap_ok": _cuda_ready,
        "fast_use_cuda": False,
        "ort_device": "unknown",
        "ort_providers": [],
        "nvidia_gpus": 0,
        "rapidocr_provider": None,
    }
    from app.core.config import settings

    status["fast_use_cuda"] = settings.fast_use_cuda
    try:
        from onnxruntime import get_available_providers, get_device

        status["ort_device"] = get_device()
        status["ort_providers"] = get_available_providers()
    except Exception as exc:  # noqa: BLE001
        status["ort_error"] = str(exc)

    try:
        from app.services.gpu import detect_cuda_device_count

        status["nvidia_gpus"] = detect_cuda_device_count()
    except Exception as exc:  # noqa: BLE001
        status["gpu_detect_error"] = str(exc)

    if _cuda_ready and settings.fast_use_cuda:
        try:
            eng = _get_engine(0)
            det = getattr(eng, "text_det", None)
            if det and hasattr(det, "session"):
                status["rapidocr_provider"] = det.session.session.get_providers()[0]
        except Exception as exc:  # noqa: BLE001
            status["rapidocr_error"] = str(exc)
    return status


def _build_engine(gpu_id: int):
    """Create a new RapidOCR instance bound to ``gpu_id``."""
    from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR

    from app.core.config import settings

    lang_map = {
        "latin": LangRec.LATIN,
        "eslav": LangRec.ESLAV,
        "en": LangRec.EN,
        "ch": LangRec.CH,
    }
    version_map = {
        "PPOCRV5": OCRVersion.PPOCRV5,
        "PPOCRV4": OCRVersion.PPOCRV4,
    }
    model_map = {
        "mobile": ModelType.MOBILE,
        "server": ModelType.SERVER,
    }

    use_cuda = settings.fast_use_cuda and _cuda_ready
    params = {
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": lang_map.get(settings.fast_lang_type, LangRec.LATIN),
        "Rec.model_type": model_map.get(settings.fast_model_type, ModelType.MOBILE),
        "Rec.ocr_version": version_map.get(settings.fast_ocr_version, OCRVersion.PPOCRV5),
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.model_type": model_map.get(settings.fast_model_type, ModelType.MOBILE),
        "Det.ocr_version": version_map.get(settings.fast_ocr_version, OCRVersion.PPOCRV5),
        "EngineConfig.onnxruntime.use_cuda": use_cuda,
        "EngineConfig.onnxruntime.gpu_id": gpu_id,
    }
    logger.info(
        "Initialising RapidOCR on GPU %d (cuda=%s, ver=%s, lang=%s)",
        gpu_id, use_cuda, settings.fast_ocr_version, settings.fast_lang_type,
    )
    return RapidOCR(params=params)


def _get_engine(gpu_id: int = 0):
    """Build (once per GPU) and return a configured RapidOCR instance."""
    with _engine_lock:
        if gpu_id not in _engines:
            _engines[gpu_id] = _build_engine(gpu_id)
        return _engines[gpu_id]


_server_engines: dict[int, object] = {}
_server_engine_lock = threading.Lock()


def _build_server_engine(gpu_id: int):
    """Like ``_build_engine`` but forces the more accurate ``server`` model."""
    from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR

    from app.core.config import settings

    lang_map = {
        "latin": LangRec.LATIN,
        "eslav": LangRec.ESLAV,
        "en": LangRec.EN,
        "ch": LangRec.CH,
    }
    version_map = {
        "PPOCRV5": OCRVersion.PPOCRV5,
        "PPOCRV4": OCRVersion.PPOCRV4,
    }
    use_cuda = settings.fast_use_cuda and _cuda_ready
    params = {
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": lang_map.get(settings.fast_lang_type, LangRec.LATIN),
        "Rec.model_type": ModelType.SERVER,
        "Rec.ocr_version": version_map.get(settings.fast_ocr_version, OCRVersion.PPOCRV5),
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.model_type": ModelType.SERVER,
        "Det.ocr_version": version_map.get(settings.fast_ocr_version, OCRVersion.PPOCRV5),
        "EngineConfig.onnxruntime.use_cuda": use_cuda,
        "EngineConfig.onnxruntime.gpu_id": gpu_id,
    }
    logger.info(
        "Initialising RapidOCR SERVER on GPU %d (cuda=%s, ver=%s, lang=%s)",
        gpu_id, use_cuda, settings.fast_ocr_version, settings.fast_lang_type,
    )
    return RapidOCR(params=params)


def _get_server_engine(gpu_id: int = 0):
    """Build (once per GPU) and return the accurate ``server`` RapidOCR engine."""
    with _server_engine_lock:
        if gpu_id not in _server_engines:
            _server_engines[gpu_id] = _build_server_engine(gpu_id)
        return _server_engines[gpu_id]


# ── public types ─────────────────────────────────────────────────────────────

@dataclass
class FastOCRLine:
    """A reconstructed visual row composed of one or more recognised boxes."""
    text: str
    mean_score: float
    min_score: float
    n_boxes: int
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x0, y0, x1, y1


@dataclass
class FastOCRResult:
    text: str                # final, line-merged text
    lines: list[FastOCRLine]
    mean_score: float        # mean over all individual word scores (0-1)
    min_score: float
    low_conf_ratio: float    # fraction of word scores < threshold
    n_boxes: int
    elapsed_sec: float
    # Block-aware reading order debug metadata (empty when advanced pipeline
    # is disabled or not applicable).  Keys: detected_columns, block_count,
    # merge_operations, overlap_removed, reading_order_strategy,
    # filtered_garbage.
    layout_meta: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.layout_meta is None:
            self.layout_meta = {}


# ── line reconstruction ──────────────────────────────────────────────────────

def _reconstruct_lines(
    boxes: Iterable, txts: list[str], scores: list[float], *,
    y_tolerance: int, separator: str, image_width: int | None = None,
) -> list[FastOCRLine]:
    """Group detected boxes that share approximately the same Y position into a
    single visual line.  Boxes inside each line are sorted left-to-right."""
    items: list[dict] = []
    for bb, txt, sc in zip(boxes, txts, scores):
        arr = np.asarray(bb, dtype=np.float32)
        y_top = float(arr[:, 1].min())
        y_bot = float(arr[:, 1].max())
        x_left = float(arr[:, 0].min())
        x_right = float(arr[:, 0].max())
        items.append({
            "y_mid": (y_top + y_bot) * 0.5,
            "x_left": x_left,
            "x_right": x_right,
            "h": y_bot - y_top,
            "txt": txt,
            "sc": float(sc),
            "bbox": (int(x_left), int(y_top), int(x_right), int(y_bot)),
        })
    if not items:
        return []

    items.sort(key=lambda r: r["y_mid"])
    median_h = float(np.median([r["h"] for r in items]))
    tol = max(float(y_tolerance), median_h * 0.5)

    lines: list[list[dict]] = []
    bucket = [items[0]]
    bucket_y = items[0]["y_mid"]
    for it in items[1:]:
        if abs(it["y_mid"] - bucket_y) <= tol:
            bucket.append(it)
        else:
            bucket.sort(key=lambda r: r["x_left"])
            lines.append(bucket)
            bucket = [it]
            bucket_y = it["y_mid"]
    bucket.sort(key=lambda r: r["x_left"])
    lines.append(bucket)

    from app.services.ocr_dedup import (
        OcrDedupPolicy,
        collapse_redundant_form_marker_lines,
        filter_detection_triplets,
        merge_line_detections,
    )

    policy = OcrDedupPolicy.from_settings()
    out: list[FastOCRLine] = []
    for line in lines:
        parts = [it["txt"] for it in line]
        bboxes = [it["bbox"] for it in line]
        scs = [it["sc"] for it in line]
        parts, bboxes, scs = filter_detection_triplets(parts, bboxes, scs)
        if not parts:
            continue
        x0 = int(min(b[0] for b in bboxes))
        x1 = int(max(b[2] for b in bboxes))
        y0 = int(min(b[1] for b in bboxes))
        y1 = int(max(b[3] for b in bboxes))
        merged = merge_line_detections(
            parts, bboxes, scs, separator, policy, image_width=image_width,
        )
        if not merged.strip():
            continue
        out.append(FastOCRLine(
            text=merged,
            mean_score=sum(scs) / len(scs),
            min_score=min(scs),
            n_boxes=len(parts),
            bbox=(x0, y0, x1, y1),
        ))
    out = collapse_redundant_form_marker_lines(out, separator)
    return out


# ── tall-page tiling ─────────────────────────────────────────────────────────

def _run_rapid_on_image(engine, image: Image.Image) -> tuple[list, list[str], list[float]]:
    """Run RapidOCR on a PIL image; return (boxes, texts, scores)."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.convert("RGB").save(tmp_path, format="PNG")
        raw = engine(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    txts = list(raw.txts) if raw.txts else []
    scores = [float(s) for s in raw.scores] if raw.scores else []
    boxes = list(raw.boxes) if raw.boxes is not None else []
    return boxes, txts, scores


def _offset_boxes(boxes: list, y_offset: int) -> list:
    out = []
    for bb in boxes:
        arr = np.asarray(bb, dtype=np.float32).copy()
        arr[:, 1] += y_offset
        out.append(arr)
    return out


def _ocr_tiled(image: Image.Image, engine) -> tuple[list, list[str], list[float]]:
    """Vertically tile a tall image, merge detections with global Y coordinates."""
    from app.core.config import settings

    w, h = image.size
    overlap = int(h * settings.fast_tile_overlap_ratio)
    strip_h = max(settings.fast_tile_min_height, h // 2)
    step = max(1, strip_h - overlap)

    all_boxes: list = []
    all_txts: list[str] = []
    all_scores: list[float] = []

    y0 = 0
    while y0 < h:
        y1 = min(h, y0 + strip_h)
        strip = image.crop((0, y0, w, y1))
        boxes, txts, scores = _run_rapid_on_image(engine, strip)
        if boxes:
            all_boxes.extend(_offset_boxes(boxes, y0))
            all_txts.extend(txts)
            all_scores.extend(scores)
        if y1 >= h:
            break
        y0 += step

    return all_boxes, all_txts, all_scores


def _result_from_boxes(
    boxes: list,
    txts: list[str],
    scores: list[float],
    *,
    elapsed_sec: float,
    image_width: int = 0,
    image_height: int = 0,
) -> FastOCRResult:
    from app.core.config import settings
    from app.services.ocr_dedup import OcrDedupPolicy, dedupe_raw_detections

    # Raw-box level dedup (tile echo at the detection stage).
    pol = OcrDedupPolicy.from_settings()
    boxes, txts, scores = dedupe_raw_detections(
        boxes, txts, scores, pol, image_width=image_width or None,
    )

    layout_meta: dict = {}

    if settings.advanced_reading_order_enabled and boxes:
        # ── Block-aware reading order pipeline ────────────────────────────────
        try:
            from app.services.ocr_reading_order import reconstruct_reading_order
            ror = reconstruct_reading_order(
                boxes, txts, scores,
                y_tolerance=settings.fast_line_y_tolerance,
                separator=settings.fast_line_separator,
                image_width=image_width or None,
                image_height=image_height or None,
            )
            lines = ror.lines
            text = ror.text
            layout_meta = ror.debug_meta
            if layout_meta:
                logger.debug(
                    "reading_order: %s  cols=%d  blocks=%d  "
                    "overlap_removed=%d  merges=%d  garbage=%d",
                    layout_meta.get("reading_order_strategy"),
                    layout_meta.get("detected_columns", 1),
                    layout_meta.get("block_count", 0),
                    layout_meta.get("overlap_removed", 0),
                    layout_meta.get("merge_operations", 0),
                    layout_meta.get("filtered_garbage", 0),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "advanced reading order failed, falling back to legacy: %s", exc,
            )
            lines = _reconstruct_lines(
                boxes, txts, scores,
                y_tolerance=settings.fast_line_y_tolerance,
                separator=settings.fast_line_separator,
                image_width=image_width or None,
            )
            text = "\n".join(line.text for line in lines)
    else:
        # ── Legacy y-sort + x-sort (fallback / disabled) ──────────────────────
        lines = _reconstruct_lines(
            boxes, txts, scores,
            y_tolerance=settings.fast_line_y_tolerance,
            separator=settings.fast_line_separator,
            image_width=image_width or None,
        )
        text = "\n".join(line.text for line in lines)

    if scores:
        mean_score = sum(scores) / len(scores)
        min_score = min(scores)
        low = sum(1 for s in scores if s < settings.smart_word_threshold)
        low_ratio = low / len(scores)
    else:
        mean_score = 0.0
        min_score = 0.0
        low_ratio = 1.0

    return FastOCRResult(
        text=text,
        lines=lines,
        mean_score=mean_score,
        min_score=min_score,
        low_conf_ratio=low_ratio,
        n_boxes=len(scores),
        elapsed_sec=elapsed_sec,
        layout_meta=layout_meta,
    )


# ── public API ───────────────────────────────────────────────────────────────

_PL_DIACRITICS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
_PL_HINTS = (
    "POLSKA", "POLSKI", "WARSZAW", "SZANOWNI", "UMOWA", "POWAZANI", "POWAŻANI",
    "UBEZPIECZ", "FAKTURA", "PARAGON", "SPRZEDA", "NABYWCA", "PŁATN", "PLATN",
    "NIP", "KRS", "REGON", "ULICA", "UL.", "POLISA", "POLISĘ",
)


def _looks_polish_garbled(text: str) -> bool:
    """Heuristic: page text is clearly Polish OCR garble — useful for triggering
    a server-model retry without relying on (often inflated) RapidOCR scores.

    Generic, content-agnostic checks:
      - text is long enough to evaluate,
      - has Polish keyword hints (so it should contain diacritics),
      - diacritic ratio is unusually low, OR a large share of lines look
        broken (low alpha ratio / many short noise tokens).
    """
    if not text or len(text) < 200:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    upper = text.upper()
    if not any(h in upper for h in _PL_HINTS):
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters < 200:
        return False
    diac = sum(1 for c in text if c in _PL_DIACRITICS)
    diac_ratio = diac / max(1, letters)

    broken = 0
    for ln in lines:
        alpha = sum(1 for c in ln if c.isalpha())
        if not alpha:
            continue
        if alpha < max(2, len(ln) // 4):
            broken += 1
            continue
        tokens = ln.split()
        weird = sum(
            1 for t in tokens
            if len(t) >= 3 and sum(c.isalpha() for c in t) < len(t) // 2
        )
        if weird >= 2:
            broken += 1
    broken_ratio = broken / max(1, len(lines))

    if diac_ratio < 0.012 and broken_ratio > 0.18:
        return True
    if broken_ratio > 0.30:
        return True
    if diac_ratio < 0.005:
        return True
    return False


def _ocr_pil_fast_uncached(
    image: Image.Image,
    *,
    gpu_id: int | None = None,
) -> FastOCRResult:
    from app.core.config import settings
    from app.services.gpu import allocate_gpu
    import time

    if gpu_id is None:
        gpu_id = allocate_gpu()
    engine = _get_engine(gpu_id)
    t0 = time.time()
    rgb = image.convert("RGB")
    with _inference_lock(gpu_id):
        if rgb.height >= settings.fast_tile_min_height:
            boxes, txts, scores = _ocr_tiled(rgb, engine)
        else:
            boxes, txts, scores = _run_rapid_on_image(engine, rgb)
    elapsed = time.time() - t0

    primary = _result_from_boxes(
        boxes, txts, scores,
        elapsed_sec=elapsed,
        image_width=rgb.width,
        image_height=rgb.height,
    )

    needs_retry = (
        settings.fast_server_retry_enabled
        and settings.fast_model_type != "server"
        and primary.n_boxes >= settings.fast_server_retry_min_boxes
        and (
            primary.mean_score < settings.fast_server_retry_min_score
            or primary.low_conf_ratio > settings.fast_server_retry_low_conf_ratio
            or (
                settings.fast_server_retry_on_garbled_text
                and _looks_polish_garbled(primary.text)
            )
        )
    )
    if not needs_retry:
        return primary

    try:
        server_engine = _get_server_engine(gpu_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Server-model engine init failed: %s", exc)
        return primary

    t1 = time.time()
    try:
        with _inference_lock(gpu_id):
            if rgb.height >= settings.fast_tile_min_height:
                s_boxes, s_txts, s_scores = _ocr_tiled(rgb, server_engine)
            else:
                s_boxes, s_txts, s_scores = _run_rapid_on_image(server_engine, rgb)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Server-model retry failed: %s", exc)
        return primary
    server_elapsed = time.time() - t1

    server = _result_from_boxes(
        s_boxes, s_txts, s_scores,
        elapsed_sec=server_elapsed,
        image_width=rgb.width,
        image_height=rgb.height,
    )

    def _quality(r: FastOCRResult) -> float:
        return r.mean_score * max(1, len(r.text))

    if _quality(server) >= _quality(primary) * 1.02:
        logger.info(
            "fast_ocr: server retry adopted (primary mean=%.2f boxes=%d, "
            "server mean=%.2f boxes=%d)",
            primary.mean_score, primary.n_boxes,
            server.mean_score, server.n_boxes,
        )
        return server
    return primary


def ocr_pil_fast(
    image: Image.Image,
    *,
    gpu_id: int | None = None,
    preprocess_variant: str = "original",
) -> FastOCRResult:
    """Run RapidOCR on an in-memory PIL image (disk cache when enabled)."""
    from app.core.config import settings

    if not settings.ocr_cache_enabled:
        return _ocr_pil_fast_uncached(image, gpu_id=gpu_id)

    from app.services.ocr_cache_integration import cached_ocr_pil_fast

    return cached_ocr_pil_fast(
        image,
        lambda: _ocr_pil_fast_uncached(image, gpu_id=gpu_id),
        preprocess_variant=preprocess_variant,
    )


def ocr_image_fast(
    image_path: Path,
    *,
    gpu_id: int | None = None,
    preprocess_variant: str = "original",
) -> FastOCRResult:
    """Run RapidOCR + line reconstruction.  Raises RuntimeError on failure."""
    from app.core.config import settings

    if not settings.ocr_cache_enabled:
        with Image.open(image_path) as img:
            img.load()
            return _ocr_pil_fast_uncached(img, gpu_id=gpu_id)

    from app.services.ocr_cache_integration import cached_ocr_image_fast

    def _compute() -> FastOCRResult:
        with Image.open(image_path) as img:
            img.load()
            return _ocr_pil_fast_uncached(img.convert("RGB"), gpu_id=gpu_id)

    return cached_ocr_image_fast(
        image_path,
        _compute,
        preprocess_variant=preprocess_variant,
    )


def warmup(*, gpu_ids: list[int] | None = None) -> None:
    """Initialise RapidOCR on given GPUs (default: primary only) and run a dummy pass."""
    from app.services.gpu import get_primary_ocr_gpu_id

    if gpu_ids is None:
        gpu_ids = [get_primary_ocr_gpu_id()]

    from io import BytesIO
    from tempfile import NamedTemporaryFile

    buf = BytesIO()
    Image.new("RGB", (128, 64), color=(255, 255, 255)).save(buf, format="PNG")

    for gpu_id in gpu_ids:
        try:
            engine = _get_engine(gpu_id)
            with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(buf.getvalue())
                tmp_path = tmp.name
            try:
                with _inference_lock(gpu_id):
                    engine(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fast OCR warmup failed on GPU %d: %s", gpu_id, exc)


def warmup_gpus(gpu_ids: list[int]) -> None:
    """Warm all listed GPUs (deduped, sorted) — safe before parallel ZIP workers."""
    unique = sorted(set(gpu_ids))
    if unique:
        warmup(gpu_ids=unique)


def should_escalate(result: FastOCRResult) -> tuple[bool, str | None]:
    """Quality gate. Returns (escalate?, reason).

    Triggers escalation to the vision LLM when ANY of the following holds:
      • too few boxes detected,
      • global mean confidence below threshold,
      • too many low-confidence words,
      • the worst individual word score is below the worst-score threshold.
    """
    from app.core.config import settings

    if result.n_boxes < settings.smart_min_lines:
        return True, f"too_few_boxes ({result.n_boxes} < {settings.smart_min_lines})"
    if result.mean_score < settings.smart_min_mean_conf:
        return True, (
            f"mean_conf_low ({result.mean_score:.3f} < "
            f"{settings.smart_min_mean_conf})"
        )
    if result.low_conf_ratio > settings.smart_min_low_conf_ratio:
        return True, (
            f"low_conf_ratio ({result.low_conf_ratio:.2%} > "
            f"{settings.smart_min_low_conf_ratio:.0%})"
        )
    if result.min_score < settings.smart_min_worst_score:
        return True, (
            f"worst_word_low ({result.min_score:.3f} < "
            f"{settings.smart_min_worst_score})"
        )
    return False, None
