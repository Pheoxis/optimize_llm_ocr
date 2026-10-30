"""Ollama text LLM client for structured JSON extraction."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


class TextLLMError(RuntimeError):
    """Raised when Ollama chat API fails."""


def chat_json(
    *,
    system: str,
    user: str,
    model: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """
    Call Ollama ``/api/chat`` and parse the assistant message as JSON object.

    Uses ``format: json`` when supported by the model/runtime.
    """
    model = model or settings.invoice_text_model
    timeout = timeout or settings.invoice_llm_timeout_sec
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": 0,
            "top_p": 0.1,
            "num_ctx": settings.invoice_llm_num_ctx,
        },
    }
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise TextLLMError(f"Ollama chat failed: {exc}") from exc

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise TextLLMError(f"Invalid JSON from Ollama: {exc}") from exc

    content = (data.get("message") or {}).get("content") or ""
    if not content.strip():
        raise TextLLMError("Empty response from Ollama")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TextLLMError(f"Model returned non-JSON content: {exc}") from exc

    if not isinstance(parsed, dict):
        raise TextLLMError("Model JSON root must be an object")
    return parsed
