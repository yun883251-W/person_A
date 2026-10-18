"""有界的私有回答诊断，并提供明确的公开/存储投影。"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from knowledge_agent.schemas import AnswerDiagnostics, AnswerResult, StepAnswerDiagnostics


MAX_FINAL_EXCERPT_CHARS = 2000
_DECLARED = frozenset({
    "supported", "partial", "insufficient_evidence", "needs_clarification", "out_of_scope",
})
_VALIDATION = frozenset({"passed", "warning", "rejected"})
_PUBLIC_FIELDS = frozenset({
    "declared_status", "validation_status", "validation_reason",
    "supplied_source_count", "accepted_source_count",
})
_PRIVATE_FIELDS = frozenset({"model_final_excerpt", "excerpt_truncated"})
_THINKING = re.compile(r"<(think|analysis|reasoning)\b[^>]*>.*?</\1\s*>", re.I | re.S)
_OPEN_THINKING = re.compile(r"<(?:think|analysis|reasoning)\b", re.I)
_SECRET = re.compile(
    r"(?i)(?:\bBearer\s+[^\s,;]+|\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]+|"
    r"\b(?:api[_-]?key|authorization|password|secret)\s*[:=]\s*[\"']?[^\s\"',;]+)"
)


def final_answer_excerpt(text: str) -> tuple[str | None, bool]:
    """只捕获有界的最终正文，排除显式的推理块。"""
    cleaned = _THINKING.sub("", text)
    opening = _OPEN_THINKING.search(cleaned)
    if opening is not None:
        cleaned = cleaned[:opening.start()]
    cleaned = _SECRET.sub("[redacted]", cleaned)
    cleaned = "".join(
        char for char in cleaned
        if char in "\n\t" or (ord(char) >= 32 and not 127 <= ord(char) <= 159
                              and not 0xD800 <= ord(char) <= 0xDFFF)
    ).strip()
    return cleaned[:MAX_FINAL_EXCERPT_CHARS] or None, len(cleaned) > MAX_FINAL_EXCERPT_CHARS


def validate_public_answer_diagnostics(value: Any) -> dict[str, Any] | None:
    """校验精确的公开允许列表；私有文本一律不接受。"""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _PUBLIC_FIELDS:
        raise ValueError("Invalid public answer diagnostics fields")
    declared = value["declared_status"]
    if declared is not None and (not isinstance(declared, str) or declared not in _DECLARED):
        raise ValueError("Invalid declared answer status")
    validation = value["validation_status"]
    if not isinstance(validation, str) or validation not in _VALIDATION:
        raise ValueError("Invalid answer validation status")
    reason = value["validation_reason"]
    if reason is not None and (not isinstance(reason, str) or not re.fullmatch(r"[a-z0-9_]{1,128}", reason)):
        raise ValueError("Invalid answer validation reason")
    for key in ("supplied_source_count", "accepted_source_count"):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"Invalid answer diagnostics {key}")
    if value["accepted_source_count"] > value["supplied_source_count"]:
        raise ValueError("Accepted sources cannot exceed supplied sources")
    return dict(value)


def public_answer_diagnostics(value: AnswerDiagnostics | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return validate_public_answer_diagnostics({
        "declared_status": value.declared_status,
        "validation_status": value.validation_status,
        "validation_reason": value.validation_reason,
        "supplied_source_count": value.supplied_source_count,
        "accepted_source_count": value.accepted_source_count,
    })


def answer_diagnostics_trace_fields(value: AnswerDiagnostics | None) -> dict[str, Any]:
    public = public_answer_diagnostics(value)
    if public is None:
        return {}
    # result_count 是最终事件上已有的已接受来源计数。
    return {key: public[key] for key in (
        "declared_status", "validation_status", "validation_reason", "supplied_source_count",
    )}


def answer_diagnostics_to_storage(value: AnswerDiagnostics | None) -> dict[str, Any] | None:
    public = public_answer_diagnostics(value)
    if public is None:
        return None
    private = {**public, "model_final_excerpt": value.model_final_excerpt,
               "excerpt_truncated": value.excerpt_truncated}
    answer_diagnostics_from_storage(private)
    return private


def answer_diagnostics_from_storage(value: Any) -> AnswerDiagnostics | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _PUBLIC_FIELDS | _PRIVATE_FIELDS:
        raise ValueError("Invalid stored answer diagnostics fields")
    public = validate_public_answer_diagnostics({key: value[key] for key in _PUBLIC_FIELDS})
    excerpt = value["model_final_excerpt"]
    if excerpt is not None and (not isinstance(excerpt, str) or len(excerpt) > MAX_FINAL_EXCERPT_CHARS):
        raise ValueError("Invalid private answer excerpt")
    if not isinstance(value["excerpt_truncated"], bool):
        raise ValueError("Invalid answer excerpt truncation flag")
    return AnswerDiagnostics(**public, model_final_excerpt=excerpt,
                             excerpt_truncated=value["excerpt_truncated"])


def step_diagnostics_to_storage(values: tuple[StepAnswerDiagnostics, ...]) -> list[dict[str, Any]]:
    payload = [{"step_id": item.step_id, "diagnostics": answer_diagnostics_to_storage(item.diagnostics)}
               for item in values]
    step_diagnostics_from_storage(payload)
    return payload


def step_diagnostics_from_storage(value: Any) -> tuple[StepAnswerDiagnostics, ...]:
    if not isinstance(value, list) or len(value) > 3:
        raise ValueError("Invalid stored step answer diagnostics")
    restored: list[StepAnswerDiagnostics] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"step_id", "diagnostics"}:
            raise ValueError("Invalid stored step diagnostics fields")
        step_id = item["step_id"]
        if not isinstance(step_id, str) or not re.fullmatch(r"P[1-3]", step_id) or step_id in seen:
            raise ValueError("Invalid stored diagnostics step ID")
        diagnostics = answer_diagnostics_from_storage(item["diagnostics"])
        if diagnostics is None:
            raise ValueError("Missing stored step diagnostics")
        restored.append(StepAnswerDiagnostics(step_id, diagnostics))
        seen.add(step_id)
    return tuple(restored)


def public_answer_result_dict(result: AnswerResult) -> dict[str, Any]:
    """CLI 投影；显式省略私有片段和未来的私有字段。

    CLI 还会通过各自的校验器进一步投影 assessment/trace/plan。
    该允许列表绝不能随 AnswerResult 字段自动扩张。
    """
    return {
        "question": result.question,
        "answer": result.answer,
        "status": result.status,
        "sources": [asdict(source) for source in result.sources],
        "index_version": result.index_version,
        "model_calls": result.model_calls,
        "tool_calls": result.tool_calls,
        "stop_reason": result.stop_reason,
        "trace": asdict(result.trace) if result.trace is not None else None,
        "routing": asdict(result.routing) if result.routing is not None else None,
        "plan": result.plan,
        "answer_diagnostics": public_answer_diagnostics(result.answer_diagnostics),
        "dialogue": asdict(result.dialogue) if result.dialogue is not None else None,
        "business_state": asdict(result.business_state) if result.business_state is not None else None,
    }
