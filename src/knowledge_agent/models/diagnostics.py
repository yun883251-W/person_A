"""有界的后端诊断；提供方细节绝不进入公开回答。"""

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
    # SDK 通常会暴露已解析的错误体。只读取其中的 error/message；
    # 绝不读取请求头、请求对象或完整 HTTP 响应。
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
    # 某些解析异常会附加完整的请求或响应 JSON。只保留
    # 技术性前缀；识别 HTTP 错误并不需要载荷内容。
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
    """记录一次真实的失败尝试，但不改变其重试策略或结果。"""
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
        # 诊断不得掩盖提供方错误，也不得再发起一次请求。
        _LOGGER.warning("chat_model_failure diagnostic_unavailable")


def record_embedding_timing(response: Any) -> None:
    """记录实际的加载/总耗时，但不记录文本或向量。

    这是 INFO 级观测埋点，不是预热请求，也不能证明 GPU 已被驱逐。
    运维人员可启用 knowledge_agent.models 日志查看。
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
