"""
Vision-LLM OCR client (Ollama).

Used for phone photos and any image where Tesseract struggles.
Produces near-perfect 1:1 transcriptions in ~6–10 s on RTX PRO 6000
(with Flash Attention + KV cache q8_0 enabled in Ollama).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, Iterable

import requests
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


class VisionLLMError(RuntimeError):
    """Raised on Ollama API failure."""


# ── prompt (strict OCR-only — no semantics, no markdown) ─────────────────────

_PROMPT_OCR_STRICT = """You are an OCR engine performing exact transcription.

Rules:

* DO NOT correct spelling
* DO NOT interpret meaning
* DO NOT infer missing text
* preserve line breaks
* preserve capitalization
* preserve duplicated text
* preserve punctuation
* preserve Polish diacritics (ą ć ę ł ń ó ś ź ż)
* transcribe character-by-character
* if unreadable use [?]

Return ONLY OCR text."""

# Backward-compatible aliases
_PROMPT_OCR_EXACT = _PROMPT_OCR_STRICT
_PROMPT_LEGACY = _PROMPT_OCR_STRICT

# Model chatter / markdown wrappers to strip from output
_OCR_PREFIX_RE = re.compile(
    r"^(?:"
    r"here(?:'s| is) (?:the )?(?:ocr )?(?:text|output|transcription)[:\s]*"
    r"|ocr(?: text| output)?[:\s]*"
    r"|output[:\s]*"
    r"|transcription[:\s]*"
    r"|result[:\s]*"
    r")+",
    re.IGNORECASE,
)
_MARKDOWN_FENCE_RE = re.compile(r"^```(?:\w+)?\s*\n?", re.MULTILINE)
_MARKDOWN_FENCE_END_RE = re.compile(r"\n?```\s*$", re.MULTILINE)


# OCR-only sampling (fixed — not overridden by env; warmup uses separate options).
_OCR_VISION_TEMPERATURE = 0.0
_OCR_VISION_TOP_P = 0.0
_OCR_VISION_REPEAT_PENALTY = 1.0
_OCR_VISION_TOP_K = 1
_OCR_VISION_SEED = 0


def _ocr_vision_options() -> dict:
    """Deterministic Ollama options for vision OCR only (fixed sampling)."""
    return {
        "temperature": _OCR_VISION_TEMPERATURE,
        "top_p": _OCR_VISION_TOP_P,
        "top_k": _OCR_VISION_TOP_K,
        "seed": _OCR_VISION_SEED,
        "repeat_penalty": _OCR_VISION_REPEAT_PENALTY,
        "num_ctx": settings.vision_num_ctx,
        "num_predict": settings.vision_num_predict,
    }


# Backward-compatible alias (tests / imports)
_vision_options = _ocr_vision_options


def sanitize_ocr_output(raw: str) -> str:
    """
    Keep only OCR text: drop markdown fences and common explanatory prefixes.
    """
    text = (raw or "").strip()
    if not text:
        return ""

    text = _MARKDOWN_FENCE_RE.sub("", text)
    text = _MARKDOWN_FENCE_END_RE.sub("", text)
    text = text.strip()

    for _ in range(3):
        stripped = _OCR_PREFIX_RE.sub("", text).strip()
        if stripped == text:
            break
        text = stripped

    # Drop a single leading wrapper line like "Sure!" / "Oto tekst:"
    lines = text.splitlines()
    if len(lines) > 1 and len(lines[0]) < 48 and lines[0].rstrip().endswith((":", "!")):
        if not any(c.isdigit() for c in lines[0]) and lines[0].count(" ") <= 8:
            text = "\n".join(lines[1:]).strip()

    return text


_DISCOURAGED_VISION_MODELS = ("qwen3.6", "qwen3.5", "gemma", "35b")


def _warn_if_heavy_vision_model(model: str, *, context: str) -> None:
    lower = model.lower()
    if any(tag in lower for tag in _DISCOURAGED_VISION_MODELS):
        logger.warning(
            "Vision model %r (%s) is discouraged for OCR — use minicpm-v or qwen2.5vl:7b",
            model,
            context,
        )


def _escalation_model(explicit: str | None = None) -> str:
    if explicit:
        _warn_if_heavy_vision_model(explicit, context="explicit")
        return explicit
    if settings.vision_escalation_model.strip():
        model = settings.vision_escalation_model.strip()
    else:
        model = settings.vision_model_fast.strip() or "minicpm-v"
    _warn_if_heavy_vision_model(model, context="escalation")
    return model


# ── persistent HTTP session (TCP keep-alive, no per-call reconnect) ──────────
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=8, max_retries=0
        )
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _session = s
    return _session


# ── helpers ───────────────────────────────────────────────────────────────────

def _downscale(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    ratio = max_side / longest
    return img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)


def _to_b64_jpeg(img: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _ollama_call(payload: dict, *, timeout: float, stream: bool):
    """Single POST to /api/generate.  Returns parsed JSON (non-stream) or iterator of chunks."""
    try:
        r = _get_session().post(
            f"{settings.ollama_url}/api/generate",
            json=payload,
            timeout=timeout,
            stream=stream,
        )
        r.raise_for_status()
    except requests.RequestException as exc:
        raise VisionLLMError(f"Ollama request failed: {exc}") from exc

    if not stream:
        try:
            return r.json()
        except json.JSONDecodeError as exc:
            raise VisionLLMError(f"Invalid JSON from Ollama: {exc}") from exc

    # True per-token streaming: parse NDJSON ourselves from raw bytes.
    # `iter_lines()` from requests buffers heavily; iterating on the raw
    # socket and splitting on '\n' surfaces each chunk the instant Ollama
    # writes it (time-to-first-token drops from ~9 s to ~3 s).
    def _iter():
        pending = b""
        for chunk in r.raw.stream(amt=512, decode_content=True):
            if not chunk:
                continue
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
        if pending.strip():
            try:
                yield json.loads(pending.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    return _iter()


# ── public API ────────────────────────────────────────────────────────────────

def _run_vision_ocr_uncached(
    img: Image.Image,
    *,
    model: str,
    prompt: str,
    max_side: int,
    timeout: float,
    on_token: Callable[[str], None] | None = None,
) -> tuple[str, dict]:
    original_size = img.size
    img = _downscale(img.convert("RGB"), max_side)
    b64 = _to_b64_jpeg(img, quality=settings.vision_jpeg_quality)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": on_token is not None,
        "keep_alive": settings.vision_keep_alive,
        "options": _ocr_vision_options(),
    }

    t0 = time.time()
    eval_count = 0
    eval_duration = 0
    load_duration = 0
    prompt_eval_duration = 0
    chunks: list[str] = []

    if on_token is None:
        data = _ollama_call(payload, timeout=timeout, stream=False)
        chunks.append((data.get("response") or "").strip())
        eval_count = int(data.get("eval_count") or 0)
        eval_duration = int(data.get("eval_duration") or 0)
        load_duration = int(data.get("load_duration") or 0)
        prompt_eval_duration = int(data.get("prompt_eval_duration") or 0)
    else:
        for chunk in _ollama_call(payload, timeout=timeout, stream=True):
            piece = chunk.get("response", "")
            if piece:
                chunks.append(piece)
                on_token(piece)
            if chunk.get("done"):
                eval_count = int(chunk.get("eval_count") or 0)
                eval_duration = int(chunk.get("eval_duration") or 0)
                load_duration = int(chunk.get("load_duration") or 0)
                prompt_eval_duration = int(chunk.get("prompt_eval_duration") or 0)

    elapsed = time.time() - t0
    text = sanitize_ocr_output("".join(chunks))
    tokens_per_sec = (
        round(eval_count / (eval_duration / 1e9), 1) if eval_duration else None
    )

    metadata = {
        "engine": "vision_llm",
        "model": model,
        "original_size": list(original_size),
        "processed_size": list(img.size),
        "elapsed_sec": round(elapsed, 2),
        "load_sec": round(load_duration / 1e9, 2),
        "prompt_eval_sec": round(prompt_eval_duration / 1e9, 2),
        "generate_sec": round(eval_duration / 1e9, 2),
        "tokens": eval_count,
        "tokens_per_sec": tokens_per_sec,
    }
    return text, metadata


def _run_vision_ocr(
    img: Image.Image,
    *,
    model: str,
    prompt: str,
    max_side: int,
    timeout: float,
    on_token: Callable[[str], None] | None = None,
) -> tuple[str, dict]:
    """Vision OCR with disk cache (non-streaming only)."""
    if settings.ocr_cache_enabled and on_token is None:
        from app.services.ocr_cache_integration import cached_vision_ocr

        def _compute() -> tuple[str, dict]:
            return _run_vision_ocr_uncached(
                img,
                model=model,
                prompt=prompt,
                max_side=max_side,
                timeout=timeout,
                on_token=None,
            )

        text, meta, _hit = cached_vision_ocr(
            img,
            _compute,
            model=model,
            max_side=max_side,
        )
        return text, meta

    return _run_vision_ocr_uncached(
        img,
        model=model,
        prompt=prompt,
        max_side=max_side,
        timeout=timeout,
        on_token=on_token,
    )


def ocr_crop_vision(
    crop: Image.Image,
    *,
    model: str | None = None,
    handwriting: bool = False,
    rapid_text: str = "",
) -> tuple[str, float | None, dict]:
    """
    VLM OCR on a small crop only (line-level escalation).

    Priority chain:
      1. MiniCPM-V direct inference  (sub-second, when enabled + model available)
      2. Ollama via minicpm_ocr       (crop-optimised ctx/tokens, when configured)
      3. Original Ollama path         (full vision_timeout, legacy fallback)

    Parameters
    ----------
    crop : PIL.Image
        Small image crop (a few text lines at most).
    model : str | None
        Override escalation model (Ollama path only).
    handwriting : bool
        Inject handwriting-specific prompt suffix.
    rapid_text : str
        RapidOCR baseline for logging and quality guard inside minicpm_ocr.
    """
    # ── 1 + 2: MiniCPM-V dedicated path ─────────────────────────────────────
    if settings.minicpm_direct_enabled or settings.minicpm_ollama_fallback:
        try:
            from app.services.minicpm_ocr import (
                CropSizeError,
                MiniCPMError,
                is_minicpm_output_useful,
                ocr_crop as _minicpm_crop,
            )
            text, conf, meta = _minicpm_crop(
                crop,
                handwriting=handwriting,
                rapid_text=rapid_text,
            )
            meta["mode"] = "crop_escalation"
            if is_minicpm_output_useful(text, rapid_baseline=rapid_text):
                return text, conf, meta
            # Output not useful — log and fall through to legacy Ollama
            logger.debug(
                "MiniCPM-V output discarded (not useful): %r  baseline=%r",
                text[:60], rapid_text[:40],
            )
        except CropSizeError:
            # Full-page image passed by mistake — skip VLM entirely
            raise VisionLLMError(
                "Image too large for crop escalation (full-page guard triggered)"
            )
        except MiniCPMError as exc:
            logger.debug("MiniCPM-V path failed, falling back to legacy Ollama: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Unexpected MiniCPM-V error: %s", exc)

    # ── 3: legacy Ollama path (original behaviour) ───────────────────────────
    model_name = _escalation_model(model)
    timeout = settings.vision_timeout
    text, meta = _run_vision_ocr(
        crop,
        model=model_name,
        prompt=_PROMPT_OCR_STRICT,
        max_side=settings.vision_max_side,
        timeout=timeout,
    )
    meta["mode"] = "crop_escalation"
    meta["path"] = "legacy_ollama"
    return text, None, meta


def ocr_image_vision(
    image_path: Path,
    *,
    model: str | None = None,
    max_side: int | None = None,
    timeout: float | None = None,
    on_token: Callable[[str], None] | None = None,
    use_exact_prompt: bool = True,
) -> tuple[str, float | None, dict]:
    """
    OCR a full image via Ollama vision LLM (discouraged — use line-level path).

    If `on_token` is given, streams partial chunks.
    """
    model = model or _escalation_model(None)
    max_side = max_side or settings.vision_max_side
    timeout = timeout or settings.vision_timeout
    prompt = _PROMPT_OCR_STRICT if use_exact_prompt else _PROMPT_LEGACY

    with Image.open(image_path) as raw:
        text, metadata = _run_vision_ocr(
            raw,
            model=model,
            prompt=prompt,
            max_side=max_side,
            timeout=timeout,
            on_token=on_token,
        )
    return text, None, metadata


def warmup(model: str | None = None) -> None:
    """Pre-load the vision model into VRAM (avoids cold-start on first request)."""
    model = model or settings.vision_model
    try:
        _ollama_call(
            {
                "model": model,
                "prompt": "ok",
                "stream": False,
                "keep_alive": settings.vision_keep_alive,
                "options": {"num_predict": 1, "num_ctx": 256},
            },
            timeout=180.0,
            stream=False,
        )
    except VisionLLMError:
        pass
