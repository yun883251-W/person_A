"""回答收尾：校验本轮来源标记、生成引用并附加路由与执行摘要。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from knowledge_agent.schemas import (
    AnswerDiagnostics, AnswerResult, AnswerSource, AnswerStatus,
    AnswerValidationStatus, RouteSummary,
)
from knowledge_agent.agent.answering.contract import (
    ANSWER_VALIDATION_REASONS,
    has_evidence_gap,
    has_scope_refusal,
    parse_answer_contract,
    plain_refusal_kind,
)
from knowledge_agent.agent.answering.diagnostics import answer_diagnostics_trace_fields, final_answer_excerpt
from knowledge_agent.agent.answering.citations import (
    SOURCE_LIKE_TOKEN_RE as _SOURCE_LIKE_TOKEN_RE,
    SOURCE_TOKEN_RE as _SOURCE_TOKEN_RE,
    WRITTEN_CITATION_RE as _WRITTEN_CITATION_RE,
    format_source as _format_source,
)
from knowledge_agent.agent.router import RouteDecision
from knowledge_agent.agent.state import ModelTurnResult, TurnState
from knowledge_agent.observability.trace import finish_trace_safely


def _attach_completed_trace(result: AnswerResult, state: TurnState) -> AnswerResult:
    """完成并附加本轮可选的执行 trace。"""
    if state.trace_recorder is None:
        return result
    failed = (
        result.status in {"service_error", "limit_reached"}
        or result.stop_reason in ANSWER_VALIDATION_REASONS
        or result.stop_reason == "partial_tool_failure"
    )
    trace = finish_trace_safely(
        state.trace_recorder,
        "failed" if failed else "completed",
        error_code=result.stop_reason if failed else None,
    )
    return replace(result, trace=trace)


def _attach_routing(result: AnswerResult, decision: RouteDecision) -> AnswerResult:
    """附加已验证的路由，但不暴露 Prompt 或工具载荷。"""
    return replace(
        result,
        routing=RouteSummary(
            route=decision.route.value,
            intent=decision.intent.kind.value,
            decision_source=decision.intent.decision_source,
            reason_code=decision.reason_code,
        ),
    )


# MCP 澄清与知识回答/证据契约相互独立。
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
    """返回稳定的公开消息，不泄露异常细节。"""
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
    """构建 MCP 轮次的公开结果，不含知识库引用。"""
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
    """完成回答，并记录引用校验结果。"""
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
        if (answer_result.status in {"service_error", "limit_reached"}
                or answer_result.stop_reason in ANSWER_VALIDATION_REASONS):
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
            reason_code=answer_result.stop_reason,
            error_code=(
                answer_result.stop_reason if trace_status == "error" else None
            ),
            **answer_diagnostics_trace_fields(answer_result.answer_diagnostics),
        )
    return answer_result


def _finalize_answer(
    result: ModelTurnResult | None,
    state: TurnState,
) -> AnswerResult:
    """校验并解析最终回答中的引用。

    只接受本轮已注册的 ``[S<n>]`` 标记。其公开渲染由不可变的索引元数据
    生成，且只返回回答实际引用的来源。需要证据的回答若无法核验引用，
    绝不作为成功结果发布。
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

    parsed = parse_answer_contract(result.answer)
    answer = parsed.body
    excerpt, excerpt_truncated = final_answer_excerpt(result.answer)

    def diagnostics(
        validation_status: AnswerValidationStatus = "passed",
        validation_reason: str | None = None,
        accepted_source_count: int = 0,
    ) -> AnswerDiagnostics:
        return AnswerDiagnostics(
            declared_status=parsed.kind,
            validation_status=validation_status,
            validation_reason=validation_reason,
            supplied_source_count=len(state.source_map),
            accepted_source_count=accepted_source_count,
            model_final_excerpt=excerpt,
            excerpt_truncated=excerpt_truncated,
        )

    def rejected(code: str, message: str) -> AnswerResult:
        state.status = "insufficient_evidence"
        state.stop_reason = code
        return AnswerResult(
            question=state.original_question,
            answer=message,
            status="insufficient_evidence",
            model_calls=state.model_calls,
            tool_calls=state.tool_calls,
            stop_reason=code,
            answer_diagnostics=diagnostics("rejected", code),
        )

    if parsed.error_code is not None:
        return rejected(
            parsed.error_code,
            "回答状态格式不完整，当前无法可靠判断回答与资料的关系，请重试。",
        )

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
        return rejected(
            "unknown_citation" if has_unknown_source else "unverified_citation_metadata",
            "回答中的来源标注无法核验，当前未能生成可靠答案。",
        )

    cited_sources = tuple(state.source_map[source_id] for source_id in cited_ids)
    # 拒答绝不能给无关文档配上看似有效的引用。先检查来源 ID，
    # 再从纯拒答的展示文本中移除它们并返回空的来源集合。
    # 历史回答也会收到同样经过校验的公开正文，只是不带协议头。
    uncited_answer = _SOURCE_TOKEN_RE.sub("", answer).strip()
    refusal = plain_refusal_kind(uncited_answer)
    non_factual_kinds = {"insufficient_evidence", "needs_clarification", "out_of_scope"}
    if refusal is not None:
        # 已核验的纯非事实性拒答即使模型选错头部也可以保留。
        # 其正文决定最终结果；模型自称“supported”绝不能把拒答变成成功。
        status = refusal
        stop_reason = {
            "insufficient_evidence": "model_reported_insufficient_evidence",
            "needs_clarification": "model_requested_clarification",
            "out_of_scope": "model_reported_out_of_scope",
        }[refusal]
        state.status = status
        state.stop_reason = stop_reason
        return AnswerResult(
            question=state.original_question,
            answer=uncited_answer,
            status=status,
            sources=(),
            index_version=None,
            model_calls=state.model_calls,
            tool_calls=state.tool_calls,
            stop_reason=stop_reason,
            answer_diagnostics=diagnostics(
                "warning" if parsed.kind is not None and parsed.kind != refusal else "passed",
                "answer_status_body_mismatch" if parsed.kind is not None and parsed.kind != refusal else None,
            ),
        )

    if parsed.kind in non_factual_kinds:
        # 格式错误的拒答是格式警告，不代表请求在技术上失败。
        # 保留模型声明的业务结果，但只发布程序生成的文本。
        # 未知引用和手写元数据已在上面的边界处被拒绝。
        status = parsed.kind
        stop_reason = {
            "insufficient_evidence": "model_reported_insufficient_evidence",
            "needs_clarification": "model_requested_clarification",
            "out_of_scope": "model_reported_out_of_scope",
        }[status]
        message = {
            "insufficient_evidence": "模型报告本轮资料不足以支持回答。其回复格式不符合要求，已用此说明替代原文；相关资料是否充分仍需进一步核实。",
            "needs_clarification": "模型表示需要补充问题的对象、参数或上下文。其回复格式不符合要求，已用此说明替代原文，请明确你希望解决的问题。",
            "out_of_scope": "模型判断该问题超出当前助手的服务范围。其回复格式不符合要求，已用此说明替代原文；本助手支持 Python 和 LangChain 学习问题。",
        }[status]
        state.status = status
        state.stop_reason = stop_reason
        return AnswerResult(
            question=state.original_question,
            answer=message,
            status=status,
            model_calls=state.model_calls,
            tool_calls=state.tool_calls,
            stop_reason=stop_reason,
            answer_diagnostics=diagnostics("warning", "refusal_format_nonconforming"),
        )
    if has_scope_refusal(uncited_answer):
        return rejected(
            "answer_status_conflict",
            "回答同时声明超出服务范围并给出实质内容，当前无法核验这些内容。",
        )
    if not cited_sources:
        return rejected(
            "missing_citations",
            "本轮尚未形成有资料支持的回答：正文包含实质内容，但没有可核验的知识库引用。",
        )

    resolved_answer = _SOURCE_TOKEN_RE.sub(
        lambda match: _format_source(state.source_map[match.group(1)]),
        answer,
    )
    if parsed.kind == "partial" or has_evidence_gap(uncited_answer):
        status = "insufficient_evidence"
        stop_reason = "partial_answer"
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
        answer_diagnostics=diagnostics(accepted_source_count=len(cited_sources)),
    )
