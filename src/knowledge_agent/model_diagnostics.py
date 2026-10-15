"""Bounded backend diagnostics; provider details never enter public answers."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any


_LOGGER = logging.getLogger("knowledge_agent.models")
_SECRET = re.compile(
    r"(?i)([\"']?(?:[a-z0-9_]*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"secret|password)|authorization|cookie)[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)
_TOKEN = re.compile(r"(?i)\b(?:Bearer\s+[^\s\"'<>]+|sk-[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_]+)")
_PRIVATE_KEY = re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?(?:-----END [^-]*PRIVATE KEY-----|$)")
_URL = re.compile(r"https?://[^\s\"'<>]+", re.I)
_PAYLOAD = re.compile(r"(?i)[\"']?\b(?:messages|prompt|request_body|tool_calls|input)[\"']?\s*[:=]")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\ud800-\udfff]")


def _error_text(exc: Exception) -> str:
    # SDKs normally expose a parsed error body. Read only its error/message;
    # never read request headers, request objects or an entire HTTP response.
    for value in (getattr(exc, "error", None), getattr(exc, "body", None)):
        if isinstance(value, Mapping):
            value = value.get("error", value)
            if isinstance(value, Mapping):
                value = value.get("message")
        if isinstance(value, str) and value.strip():
            return value
    return str(exc)


def _safe_detail(exc: Exception) -> str:
    text = _error_text(exc)
    # Some parser exceptions append complete request or response JSON. Keep the
    # technical prefix only; payloads are not needed to identify an HTTP error.
    payload = _PAYLOAD.search(text)
    if payload is not None:
        text = text[:payload.start()] + "[payload omitted]"
    if "outputparser" in type(exc).__name__.lower():
        return "Provider response could not be parsed; raw response omitted."
    text = _PRIVATE_KEY.sub("[private key omitted]", text)
    text = _TOKEN.sub("[redacted]", text)
    text = _SECRET.sub(r"\1[redacted]", text)
    text = _URL.sub("[url omitted]", text)
    return _CONTROL.sub(" ", text).strip()[:640]


def record_chat_failure(
    exc: Exception, *, status_code: int | None, error_category: str,
    retryable: bool, attempt: int, model: Any,
) -> None:
    """Log one real failed attempt without changing its retry policy or result."""
    try:
        client = getattr(model, "bound", model)
        diagnostic = {
            "event": "chat_model_failure",
            "client": type(client).__name__,
            "exception_type": type(exc).__name__,
            "http_status": status_code,
            "error_category": error_category,
            "retryable": retryable,
            "attempt": attempt,
            "detail": _safe_detail(exc),
        }
        _LOGGER.warning("%s", json.dumps(diagnostic, ensure_ascii=True))
    except Exception:
        # Diagnostics must not hide the provider error or cause another request.
        _LOGGER.warning("chat_model_failure diagnostic_unavailable")


def record_embedding_timing(response: Any) -> None:
    """Expose actual load/total durations without logging text or vectors.

    This is INFO-level instrumentation, not a warm-up request or proof of GPU
    eviction. Operators can enable knowledge_agent.models logging to inspect it.
    """
    try:
        def value(name: str) -> Any:
            return response.get(name) if isinstance(response, Mapping) else getattr(response, name, None)

        durations = {}
        for key in ("total_duration", "load_duration"):
            raw = value(key)
            if type(raw) is int and raw >= 0:
                durations[key.replace("_duration", "_ms")] = round(raw / 1_000_000, 3)
        if durations:
            _LOGGER.info("embedding_timing %s", json.dumps(durations, ensure_ascii=True))
    except Exception:
        pass
