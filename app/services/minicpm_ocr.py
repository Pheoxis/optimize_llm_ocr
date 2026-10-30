"""
MiniCPM-V dedicated crop-level OCR escalation.

Design goals
------------
* Sub-second latency for single / few-line crops.
* Deterministic output (greedy decoding, temperature=0, seed=0).
* Polish + English support (MiniCPM-V-2 / 2.6 handles both natively).
* Strict OCR-only: no spelling correction, no semantic rewriting.
* Crop-only: full-page inference is explicitly refused.

Inference priority chain
------------------------
1. Direct transformers inference  — fastest (~300-800 ms for a warm model)
   Requires model weights on disk + ``transformers`` + ``torch`` with CUDA.
2. Ollama HTTP fallback            — ~2-6 s, uses crop-optimised num_ctx/num_predict.
3. Neither available              — raises ``MiniCPMError``; caller uses RapidOCR baseline.

Latency breakdown (MiniCPM-V-2 BF16, RTX PRO 6000 48 GB)
---------------------------------------------------------
  Image encode  ≈  50-120 ms   (512px crop)
  Token prefill ≈   80-150 ms
  Generation    ≈  100-300 ms  (15-40 tokens/line)
  ──────────────────────────────
  Total         ≈  230-570 ms  → sub-second ✓

Guardrails (same as existing pipeline)
---------------------------------------
* Hard crop-size limit: longest side clamped to ``minicpm_max_crop_side`` px.
* Full-page refusal: image area > ``_FULLPAGE_AREA_LIMIT`` raises ``CropSizeError``.
* Output sanitization: strip markdown, prefixes, chatter via ``sanitize_ocr_output``.
* Degradation guard: caller's ``_vlm_degrades_rapid`` decides whether to keep output.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import settings
from app.services.vision_llm import sanitize_ocr_output

logger = logging.getLogger(__name__)

# Maximum image area (px²) allowed for crop inference.
# Anything larger is almost certainly a full page — refuse it.
_FULLPAGE_AREA_LIMIT = 1_200 * 1_600  # ~A4 at 150 dpi


class MiniCPMError(RuntimeError):
    """Raised when MiniCPM-V inference fails and no fallback is available."""


class CropSizeError(MiniCPMError):
    """Image is too large to be a crop (full-page guard)."""


# ── OCR prompt ────────────────────────────────────────────────────────────────

_PROMPT_CROP_OCR = (
    "You are a precise OCR engine. "
    "Transcribe ALL visible text EXACTLY as printed.\n\n"
    "Rules:\n"
    "- DO NOT correct spelling or grammar\n"
    "- DO NOT add explanations or commentary\n"
    "- Preserve exact capitalization and punctuation\n"
    "- Preserve Polish diacritics: ą ć ę ł ń ó ś ź ż\n"
    "- Preserve duplicated words if they appear\n"
    "- For unreadable characters write [?]\n"
    "- Return ONLY the transcribed text, nothing else"
)

# Handwriting hint — injected when the crop is flagged as handwritten
_PROMPT_HW_SUFFIX = (
    "\n\nThis crop contains handwritten text. "
    "Transcribe character-by-character; do not guess unreadable parts."
)


# ── direct transformers backend ───────────────────────────────────────────────

_model_lock = threading.Lock()
_model: Any | None = None
_tokenizer: Any | None = None
_model_name_loaded: str | None = None
_model_available: bool | None = None  # None = not probed yet


def _probe_model_available() -> bool:
    """Check whether the configured model can be loaded (no GPU needed for probe)."""
    global _model_available
    if _model_available is not None:
        return _model_available

    with _model_lock:
        if _model_available is not None:
            return _model_available
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            _model_available = True
        except ImportError:
            logger.info(
                "MiniCPM-V direct inference unavailable: transformers/torch not installed"
            )
            _model_available = False
    return _model_available


def _load_model() -> tuple[Any, Any]:
    """
    Lazy-load MiniCPM-V model + tokenizer (once per process).

    Thread-safe; subsequent calls return the cached instance immediately.
    Raises ``MiniCPMError`` on any loading failure.
    """
    global _model, _tokenizer, _model_name_loaded

    if _model is not None:
        return _model, _tokenizer  # type: ignore[return-value]

    with _model_lock:
        if _model is not None:
            return _model, _tokenizer  # type: ignore[return-value]

        if not _probe_model_available():
            raise MiniCPMError("transformers / torch not installed")

        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = settings.minicpm_model_name
        dtype_str = settings.minicpm_dtype.lower()
        device = settings.minicpm_device

        logger.info(
            "Loading MiniCPM-V model %r  dtype=%s  device=%s …",
            model_name, dtype_str, device,
        )
        t0 = time.perf_counter()

        try:
            if dtype_str == "int4":
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_cfg,
                    device_map=device,
                )
            else:
                dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map=device,
                )

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            model.eval()
        except Exception as exc:
            raise MiniCPMError(f"Model load failed for {model_name!r}: {exc}") from exc

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "MiniCPM-V model loaded in %.0f ms  (name=%r  device=%s)",
            elapsed, model_name, device,
        )
        _model = model
        _tokenizer = tokenizer
        _model_name_loaded = model_name
        return _model, _tokenizer  # type: ignore[return-value]


def _resize_crop(img: Image.Image, max_side: int) -> Image.Image:
    """Resize crop so longest side ≤ max_side; preserve aspect ratio."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.LANCZOS,
    )


def _direct_infer(
    crop: Image.Image,
    prompt: str,
) -> tuple[str, dict]:
    """
    Run MiniCPM-V inference directly via transformers.

    Returns (text, metadata_dict).
    Raises ``MiniCPMError`` on failure.
    """
    model, tokenizer = _load_model()

    import torch

    rgb = crop.convert("RGB")
    rgb = _resize_crop(rgb, settings.minicpm_max_crop_side)

    msgs = [{"role": "user", "content": [rgb, prompt]}]

    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            raw = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,          # greedy
                temperature=0.0,
                max_new_tokens=settings.minicpm_max_tokens,
            )
    except Exception as exc:
        raise MiniCPMError(f"Inference error: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    text = sanitize_ocr_output(raw if isinstance(raw, str) else str(raw))

    meta: dict = {
        "engine": "minicpm_direct",
        "model": settings.minicpm_model_name,
        "crop_size": list(rgb.size),
        "elapsed_ms": round(elapsed_ms, 1),
        "max_tokens": settings.minicpm_max_tokens,
    }
    logger.debug(
        "MiniCPM-V direct inference: %.0f ms  crop=%s  text=%r",
        elapsed_ms, rgb.size, text[:60],
    )
    return text, meta


# ── Ollama fallback backend ───────────────────────────────────────────────────


def _ollama_crop_options() -> dict:
    """Crop-specific Ollama options (smaller context → faster)."""
    return {
        "temperature": 0.0,
        "top_p": 0.0,
        "top_k": 1,
        "seed": 0,
        "repeat_penalty": 1.0,
        "num_ctx": settings.minicpm_ollama_num_ctx,
        "num_predict": settings.minicpm_ollama_num_predict,
    }


def _ollama_infer(
    crop: Image.Image,
    prompt: str,
) -> tuple[str, dict]:
    """
    Run crop OCR via Ollama using the configured escalation model.

    Returns (text, metadata_dict).
    Raises ``MiniCPMError`` on failure.
    """
    import base64
    import io
    import json

    import requests

    rgb = crop.convert("RGB")
    rgb = _resize_crop(rgb, settings.minicpm_max_crop_side)

    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=settings.minicpm_jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    model = settings.vision_escalation_model or "minicpm-v"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "keep_alive": settings.vision_keep_alive,
        "options": _ollama_crop_options(),
    }

    timeout_sec = settings.minicpm_crop_timeout_ms / 1000.0
    t0 = time.perf_counter()
    try:
        from app.services.vision_llm import _get_session
        r = _get_session().post(
            f"{settings.ollama_url}/api/generate",
            json=payload,
            timeout=timeout_sec,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        raise MiniCPMError(f"Ollama crop request failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    raw_text = (data.get("response") or "").strip()
    text = sanitize_ocr_output(raw_text)

    meta: dict = {
        "engine": "minicpm_ollama",
        "model": model,
        "crop_size": list(rgb.size),
        "elapsed_ms": round(elapsed_ms, 1),
        "tokens": data.get("eval_count"),
    }
    logger.debug(
        "MiniCPM-V Ollama fallback: %.0f ms  crop=%s  text=%r",
        elapsed_ms, rgb.size, text[:60],
    )
    return text, meta


# ── public API ────────────────────────────────────────────────────────────────


def ocr_crop(
    crop: Image.Image,
    *,
    handwriting: bool = False,
    rapid_text: str = "",
) -> tuple[str, float | None, dict]:
    """
    OCR a single crop image using MiniCPM-V.

    Parameters
    ----------
    crop : PIL.Image
        Small region image (a text line or a few merged lines).
        Must not be a full page — raises ``CropSizeError`` otherwise.
    handwriting : bool
        Inject handwriting-specific prompt suffix.
    rapid_text : str
        RapidOCR baseline (used only for logging / debug).

    Returns
    -------
    (text, confidence, metadata)
        ``confidence`` is always ``None`` (model does not output probabilities).

    Raises
    ------
    CropSizeError
        Image area exceeds ``_FULLPAGE_AREA_LIMIT``.
    MiniCPMError
        Both inference paths failed.
    """
    w, h = crop.size
    if w * h > _FULLPAGE_AREA_LIMIT:
        raise CropSizeError(
            f"Image {w}×{h} px is too large for crop escalation "
            f"(limit {_FULLPAGE_AREA_LIMIT // 1_000:.0f} kpx²). "
            "Use crop-level escalation only."
        )

    prompt = _PROMPT_CROP_OCR
    if handwriting:
        prompt = prompt + _PROMPT_HW_SUFFIX

    errors: list[str] = []

    # ── 1. direct transformers inference ────────────────────────────────────
    if settings.minicpm_direct_enabled and _probe_model_available():
        try:
            text, meta = _direct_infer(crop, prompt)
            meta["path"] = "direct"
            if rapid_text:
                meta["rapid_baseline"] = rapid_text[:80]
            return text, None, meta
        except MiniCPMError as exc:
            errors.append(f"direct: {exc}")
            logger.warning("MiniCPM-V direct inference failed: %s", exc)

    # ── 2. Ollama fallback ───────────────────────────────────────────────────
    if settings.minicpm_ollama_fallback:
        try:
            text, meta = _ollama_infer(crop, prompt)
            meta["path"] = "ollama_fallback"
            meta["direct_errors"] = errors
            if rapid_text:
                meta["rapid_baseline"] = rapid_text[:80]
            return text, None, meta
        except MiniCPMError as exc:
            errors.append(f"ollama: {exc}")
            logger.warning("MiniCPM-V Ollama fallback failed: %s", exc)

    raise MiniCPMError(
        f"All MiniCPM-V inference paths failed: {'; '.join(errors)}"
    )


def warmup() -> bool:
    """
    Pre-load the model into VRAM so the first crop request has no cold-start.

    Returns True when direct inference is warm, False when using Ollama fallback.
    """
    if settings.minicpm_direct_enabled:
        try:
            _load_model()
            logger.info("MiniCPM-V direct model warm in VRAM (%s)", settings.minicpm_model_name)
            return True
        except MiniCPMError as exc:
            logger.warning("MiniCPM-V direct warmup failed: %s  — will use Ollama", exc)

    if settings.minicpm_ollama_fallback:
        # Ping Ollama to warm the model (small keepalive call)
        try:
            from app.services.vision_llm import _get_session
            import json as _json

            _get_session().post(
                f"{settings.ollama_url}/api/generate",
                json={
                    "model": settings.vision_escalation_model or "minicpm-v",
                    "prompt": "ok",
                    "stream": False,
                    "keep_alive": settings.vision_keep_alive,
                    "options": {"num_predict": 1, "num_ctx": 128},
                },
                timeout=30.0,
            )
            logger.info("MiniCPM-V Ollama model warmed (%s)", settings.vision_escalation_model)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MiniCPM-V Ollama warmup ping failed: %s", exc)

    return False


def is_model_loaded() -> bool:
    """True if the direct transformers model is currently in memory."""
    return _model is not None


def unload_model() -> None:
    """Release VRAM by deleting the loaded model (useful for memory pressure)."""
    global _model, _tokenizer, _model_name_loaded
    with _model_lock:
        if _model is not None:
            try:
                import torch
                del _model, _tokenizer
                _model = None
                _tokenizer = None
                _model_name_loaded = None
                torch.cuda.empty_cache()
                logger.info("MiniCPM-V model unloaded from VRAM")
            except Exception as exc:  # noqa: BLE001
                logger.warning("MiniCPM-V unload failed: %s", exc)


# ── output quality guardrails ─────────────────────────────────────────────────

_QMARK_SPAM_RE = __import__("re").compile(r"(?:\[\?\]\s*){20,}")
_REPEAT_TOKEN_RE = __import__("re").compile(r"\b(\w{3,})\b(?:\s+\1\b){4,}")


def is_minicpm_output_useful(
    text: str,
    *,
    rapid_baseline: str = "",
    min_chars: int = 2,
) -> bool:
    """
    Return False when the MiniCPM-V output should be discarded.

    Checks:
    * Empty or whitespace-only output.
    * [?] spam (model couldn't read anything).
    * Runaway token repetition (degenerate generation).
    * Output shorter than ``min_chars`` after stripping.
    """
    t = (text or "").strip()
    if len(t) < min_chars:
        return False
    if _QMARK_SPAM_RE.search(t):
        return False
    if _REPEAT_TOKEN_RE.search(t):
        return False
    # If we have a baseline: reject if VLM is suspiciously short
    if rapid_baseline.strip() and len(t) < max(3, len(rapid_baseline.strip()) // 3):
        return False
    return True
