"""回答收尾：校验本轮来源标记、生成引用并附加路由与执行摘要。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ..schemas import AnswerResult, AnswerSource, AnswerStatus, RouteSummary
from .citations import (
    SOURCE_LIKE_TOKEN_RE as _SOURCE_LIKE_TOKEN_RE,
    SOURCE_TOKEN_RE as _SOURCE_TOKEN_RE,
    WRITTEN_CITATION_RE as _WRITTEN_CITATION_RE,
    format_source as _format_source,
)
from .router import RouteDecision
from .state import ModelTurnResult, TurnState
from .trace import finish_trace_safely


def _attach_completed_trace(result: AnswerResult, state: TurnState) -> AnswerResult:
    """Finalize and attach this turn's optional execution trace."""
    if state.trace_recorder is None:
        return result
    failed = result.status in {"service_error", "limit_reached"}
    trace = finish_trace_safely(
        state.trace_recorder,
        "failed" if failed else "completed",
        error_code=result.stop_reason if failed else None,
    )
    return replace(result, trace=trace)


def _attach_routing(result: AnswerResult, decision: RouteDecision) -> AnswerResult:
    """Attach the validated route without exposing prompts or tool payloads."""
    return replace(
        result,
        routing=RouteSummary(
            route=decision.route.value,
            intent=decision.intent.kind.value,
            decision_source=decision.intent.decision_source,
            reason_code=decision.reason_code,
        ),
    )


_INSUFFICIENT_EVIDENCE_MARKERS = (
    "当前知识库资料不足",
    "资料不足以回答",
    "参考资料中暂无相关信息",
    "insufficient evidence",
    "not enough information",
)
_PARTIAL_ANSWER_MARKERS = (
    "只能回答",
    "仅能回答",
    "只支持部分",
    "仅支持部分",
    "部分回答",
    "partial answer",
)
_CLARIFICATION_MARKERS = (
    "请说明",
    "请明确",
    "请提供",
    "请补充",
    "你指的是",
    "could you clarify",
    "please clarify",
)


def _public_failure_answer(status: AnswerStatus, stop_reason: str | None) -> str:
    """Return a stable public message without leaking exception details."""
    if status == "limit_reached":
        return "已达到本轮处理上限，当前未能生成完整可验证的回答。"
    if stop_reason == "model_empty_response":
        return "模型未返回可用的回答，请稍后重试。"
    return "当前无法完成回答，请稍后重试。"


def _answer_index_version(sources: Sequence[AnswerSource]) -> str | None:
    versions = {source.index_version for source in sources}
    return next(iter(versions)) if len(versions) == 1 else None


def finalize_mcp_answer(
    result: ModelTurnResult | None,
    state: TurnState,
) -> AnswerResult:
    """Build the public result for an MCP turn without knowledge citations."""
    if state.browser_action_result is not None:
        return state.browser_action_result
    if state.status in {"service_error", "limit_reached"}:
        status = state.status
        return AnswerResult(
            question=state.original_question,
            answer=_public_failure_answer(status, state.stop_reason),
            status=status,
            model_calls=state.model_calls,
            tool_calls=state.tool_calls,
            stop_reason=state.stop_reason,
        )
    if state.successful_tool_calls == 0:
        code = state.last_tool_error_code
        proposed_answer = (
            result.answer.strip()
            if result is not None and result.kind == "final_answer" and result.answer
            else ""
        )
        asks_for_clarification = any(
            marker in proposed_answer.casefold() for marker in _CLARIFICATION_MARKERS
        )
        if asks_for_clarification or code in {
            "invalid_arguments",
            "invalid_tool_arguments",
        }:
            status: AnswerStatus = "needs_clarification"
            answer = proposed_answer or "缺少完成该操作所需的有效参数，请补充后重试。"
            stop_reason = code or "model_requested_clarification"
        else:
            status = "service_error"
            stop_reason = code or "tool_not_executed"
            answer = (
                "工具调用超时，操作可能已经产生外部效果；请先核实实际状态，避免直接重试。"
                if code in {"timeout", "tool_timeout"}
                else "工具未成功完成该操作。"
            )
        state.status = status
        state.stop_reason = stop_reason
        return AnswerResult(
            question=state.original_question,
            answer=answer,
            status=status,
            sources=(),
            index_version=None,
            model_calls=state.model_calls,
            tool_calls=state.tool_calls,
            stop_reason=stop_reason,
        )
    if result is None or result.kind != "final_answer" or not result.answer:
        state.status = "service_error"
        state.stop_reason = state.stop_reason or "model_did_not_finish"
        return AnswerResult(
            question=state.original_question,
            answer=_public_failure_answer("service_error", state.stop_reason),
            status="service_error",
            model_calls=state.model_calls,
            tool_calls=state.tool_calls,
            stop_reason=state.stop_reason,
        )

    answer = result.answer.strip()
    is_clarification = any(
        marker in answer.casefold() for marker in _CLARIFICATION_MARKERS
    )
    status: AnswerStatus = "needs_clarification" if is_clarification else "success"
    stop_reason = (
        "model_requested_clarification"
        if is_clarification
        else "partial_tool_failure"
        if state.failed_tool_calls
        else "answer_completed"
    )
    state.status = status
    state.stop_reason = stop_reason
    return AnswerResult(
        question=state.original_question,
        answer=answer,
        status=status,
        model_calls=state.model_calls,
        tool_calls=state.tool_calls,
        stop_reason=stop_reason,
    )


def finalize_answer(
    result: ModelTurnResult | None,
    state: TurnState,
) -> AnswerResult:
    """Finalize an answer and record the citation-validation outcome."""
    recorder = state.trace_recorder
    finalize_event = (
        recorder.start_event("finalize", "program", step_id=state.step_id) if recorder is not None else None
    )
    try:
        answer_result = _finalize_answer(result, state)
    except Exception:
        if recorder is not None and finalize_event is not None:
            recorder.finish_event(
                finalize_event,
                "error",
                error_code="answer_finalization_failed",
            )
        raise

    if recorder is not None and finalize_event is not None:
        if answer_result.status in {"service_error", "limit_reached"}:
            trace_status = "error"
        elif answer_result.status == "insufficient_evidence":
            trace_status = "empty"
        elif answer_result.status in {"needs_clarification", "out_of_scope"}:
            trace_status = "skipped"
        else:
            trace_status = "success"
        recorder.finish_event(
            finalize_event,
            trace_status,
            result_count=len(answer_result.sources),
            index_version=answer_result.index_version,
            error_code=(
                answer_result.stop_reason if trace_status != "success" else None
            ),
        )
    return answer_result


def _finalize_answer(
    result: ModelTurnResult | None,
    state: TurnState,
) -> AnswerResult:
    """Validate and resolve the final answer's citations.

    Only ``[S<n>]`` tokens registered in this turn are accepted. Their public
    rendering is generated from immutable index metadata, and only sources
    actually cited by the answer are returned. A response requiring evidence
    is never published as successful when its citations cannot be verified.
    """
    if state.status in {"service_error", "limit_reached"}:
        status = state.status
        return AnswerResult(
            question=state.original_question,
            answer=_public_failure_answer(status, state.stop_reason),
            status=status,
            model_calls=state.model_calls,
            tool_calls=state.tool_calls,
            stop_reason=state.stop_reason,
        )

    if result is None or result.kind != "final_answer" or not result.answer:
        status: AnswerStatus = state.status or "service_error"
        stop_reason = state.stop_reason or "model_did_not_finish"
        state.status = status
        state.stop_reason = stop_reason
        return AnswerResult(
            question=state.original_question,
            answer=_public_failure_answer(status, stop_reason),
            status=status,
            model_calls=state.model_calls,
            tool_calls=state.tool_calls,
            stop_reason=stop_reason,
        )

    answer = result.answer.strip()
    folded_answer = answer.casefold()
    is_clarification = any(
        marker in folded_answer for marker in _CLARIFICATION_MARKERS
    )
    is_insufficient = any(
        marker in folded_answer for marker in _INSUFFICIENT_EVIDENCE_MARKERS
    )
    is_partial = any(marker in folded_answer for marker in _PARTIAL_ANSWER_MARKERS)

    cited_ids: list[str] = []
    seen_ids: set[str] = set()
    for match in _SOURCE_TOKEN_RE.finditer(answer):
        source_id = match.group(1)
        if source_id not in seen_ids:
            seen_ids.add(source_id)
            cited_ids.append(source_id)

    recognized_tokens = {f"[{source_id}]" for source_id in cited_ids}
    source_like_tokens = set(_SOURCE_LIKE_TOKEN_RE.findall(answer))
    has_unknown_source = any(
        token not in recognized_tokens
        or token[1:-1] not in state.source_map
        for token in source_like_tokens
    )
    has_written_citation = _WRITTEN_CITATION_RE.search(answer) is not None

    if has_unknown_source or has_written_citation:
        state.status = "insufficient_evidence"
        state.stop_reason = (
            "unknown_citation" if has_unknown_source else "unverified_citation_metadata"
        )
        return AnswerResult(
            question=state.original_question,
            answer="回答中的来源标注无法核验，当前未能生成可靠答案。",
            status="insufficient_evidence",
            model_calls=state.model_calls,
            tool_calls=state.tool_calls,
            stop_reason=state.stop_reason,
        )

    cited_sources = tuple(state.source_map[source_id] for source_id in cited_ids)
    if not cited_sources and not (is_clarification or is_insufficient):
        state.status = "insufficient_evidence"
        state.stop_reason = "missing_citations"
        return AnswerResult(
            question=state.original_question,
            answer="回答缺少可核验的知识库引用，当前未能生成可靠答案。",
            status="insufficient_evidence",
            model_calls=state.model_calls,
            tool_calls=state.tool_calls,
            stop_reason=state.stop_reason,
        )

    resolved_answer = _SOURCE_TOKEN_RE.sub(
        lambda match: _format_source(state.source_map[match.group(1)]),
        answer,
    )
    if is_clarification:
        status = "needs_clarification"
        stop_reason = "model_requested_clarification"
    elif is_insufficient or is_partial:
        status = "insufficient_evidence"
        stop_reason = (
            "partial_answer"
            if cited_sources
            else "model_reported_insufficient_evidence"
        )
    else:
        status = "success"
        stop_reason = "answer_completed"

    state.status = status
    state.stop_reason = stop_reason
    return AnswerResult(
        question=state.original_question,
        answer=resolved_answer,
        status=status,
        sources=cited_sources,
        index_version=_answer_index_version(cited_sources),
        model_calls=state.model_calls,
        tool_calls=state.tool_calls,
        stop_reason=stop_reason,
    )
