"""执行一个经宿主确认的 B 站任务，并提供持久化的派发保护。

调用方已提交 READY 转换。此边界会重新校验原始请求和当前槽位来源，
在外部调用之前消费待处理任务，并且只接受浏览器服务的精确结果契约。
它不用模型分类、不凭空生成工具调用，也不重放不确定的工作。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from browser_mcp.contracts import SEARCH_BILIBILI_TOOL, search_success_payload, validate_search_arguments

from ...config import AgentConfig
from ...schemas import AnswerResult, AnswerStatus, DialogueSummary, RouteSummary
from ...tools.registry import ToolRegistry
from knowledge_agent.agent.answering.answers import _attach_completed_trace
from knowledge_agent.agent.browser.actions import (
    classify_bilibili_request, extract_bilibili_topic, is_explicit_bilibili_topic, requested_video_limit,
)
from knowledge_agent.agent.browser.execution import (
    SEARCH_NOT_DISPATCHED_ERRORS, search_error_code, search_result_answer,
    verified_search_payload,
)
from ..state import TurnState
from knowledge_agent.observability.trace import TraceRecorder
from .contracts import (
    BusinessPhase, BusinessType, DialogueContractError, DialogueDecision,
    DialogueIntent, ExecutionMode, SessionBusinessState, SlotSource,
    StateTransition, validate_business_state, validate_dialogue_decision,
)
from .transitions import (
    ExecutionOutcome, apply_execution_outcome, mark_dispatching, prepare_execution,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ready_arguments(
    question: str, session_id: str, ready: SessionBusinessState,
    decision: DialogueDecision, *, turn_id: str, registry: ToolRegistry[TurnState],
) -> dict[str, Any]:
    """独立于先前的分类器，重新检查当前轮次的授权。"""
    validate_business_state(ready)
    validate_dialogue_decision(decision)
    pending = ready.pending
    if (
        ready.session_id != session_id
        or ready.focus is None
        or ready.focus.business_type is not BusinessType.BILIBILI_SEARCH
        or ready.focus.phase is not BusinessPhase.READY
        or pending is None
        or pending.business_type is not BusinessType.BILIBILI_SEARCH
        or pending.session_id != session_id
        or pending.missing_slots
        or decision.intent not in {DialogueIntent.NEW_BUSINESS, DialogueIntent.CONTINUE_BUSINESS}
        or decision.business_type is not BusinessType.BILIBILI_SEARCH
        or decision.execution_mode is not ExecutionMode.SINGLE
        or decision.missing_slots
        or decision.allowed_tool_names != (SEARCH_BILIBILI_TOOL,)
        or decision.expected_revision + 1 != ready.revision
    ):
        raise DialogueContractError("browser_pending_authorization_invalid")

    # READY 是基于该决定提交的，因此执行只有在证明它恰好是该决定的
    # 下一个状态之后，才使用新的 revision。
    current_decision = replace(decision, expected_revision=ready.revision)
    proposal = prepare_execution(ready, current_decision)
    arguments = validate_search_arguments(proposal.arguments)
    if arguments != dict(pending.confirmed_slots) or len(proposal.slot_bindings) != len(arguments):
        raise DialogueContractError("browser_pending_authorization_invalid")
    bindings = {item.slot_name: item for item in proposal.slot_bindings}
    if set(bindings) != set(arguments):
        raise DialogueContractError("browser_pending_authorization_invalid")
    for name, item in bindings.items():
        if (item.value != arguments[name] or item.session_id != session_id
                or item.source_turn_id != turn_id or item.source is not SlotSource.CURRENT_USER_INPUT
                or item.raw_input.strip() != question.strip()):
            raise DialogueContractError("browser_pending_authorization_invalid")
    binding = bindings["query"]
    topic = arguments["query"]
    if (
        binding.slot_name != "query" or binding.value != topic
        or binding.session_id != session_id or binding.source_turn_id != turn_id
        or binding.source is not SlotSource.CURRENT_USER_INPUT
        or binding.raw_input.strip() != question.strip()
        or topic not in binding.raw_input
        or not is_explicit_bilibili_topic(topic)
    ):
        raise DialogueContractError("browser_pending_authorization_invalid")

    original_action = classify_bilibili_request(pending.original_request)
    semantic_authorized = (
        original_action is not None and original_action.kind == "semantic_candidate"
        and decision.intent is DialogueIntent.NEW_BUSINESS
        and decision.reason_code == "browser_bilibili_semantic_search"
        and decision.decision_source.value == "model"
        and decision.authorized_user_input.strip() == question.strip()
    )
    if decision.intent is DialogueIntent.NEW_BUSINESS:
        if (
            pending.request_turn_id != turn_id
            or pending.original_request.strip() != question.strip()
        ):
            raise DialogueContractError("browser_pending_authorization_invalid")
        literal_topic = extract_bilibili_topic(pending.original_request)
        if literal_topic != topic:
            # 语义分类器已经通过宿主的显式操作边界。
            # 保留这份精确的当前请求和字面主题；
            # 泛化澄清或历史模型内容都不够。
            if not semantic_authorized:
                raise DialogueContractError("browser_pending_authorization_invalid")
    elif (
        pending.request_turn_id == turn_id
        or original_action is None
        or original_action.kind != "needs_clarification"
        or original_action.reason_code != "browser_bilibili_missing_topic"
        or topic != question.strip()
    ):
        # 跨轮执行只是对一个已确立、具体的缺主题问题的字面回答，
        # 绝不是猜测出来的业务延续。
        raise DialogueContractError("browser_pending_authorization_invalid")

    # 模型提取的limit必须来自本轮已验证决定及持久化绑定，不能在执行时再猜。
    # 旧版待办只存query，仍可由原始请求恢复规则明确的数量。
    limit = requested_video_limit(pending.original_request)
    if not semantic_authorized:
        if "limit" in arguments and arguments["limit"] != limit:
            raise DialogueContractError("browser_pending_authorization_invalid")
        if limit is not None:
            arguments = validate_search_arguments({**arguments, "limit": limit})
    registration = registry.get(SEARCH_BILIBILI_TOOL)
    if registration is None:
        raise DialogueContractError("tool_unavailable")
    try:
        if registration.argument_validator is not None:
            checked = registration.argument_validator(arguments)
        else:
            checked = registration.definition.get_input_schema().model_validate(arguments).model_dump()
    except Exception as exc:
        # MCP 适配器和本地 schema 暴露不同的异常类。
        # 被拒绝的 schema 仍属于派发前失败，而不是 HTTP 崩溃。
        raise DialogueContractError("browser_invalid_tool_arguments") from exc
    if not isinstance(checked, dict) or checked != arguments:
        raise DialogueContractError("browser_invalid_tool_arguments")
    return arguments


def _result(
    state: TurnState, decision: DialogueDecision, answer: str, *,
    status: AnswerStatus, code: str,
) -> AnswerResult:
    source = "model" if decision.decision_source.value == "model" else "rules"
    if state.trace_recorder is not None:
        handle = state.trace_recorder.start_event("finalize", "program")
        state.trace_recorder.finish_event(
            handle, "success" if status == "success" else "error",
            reason_code=code, error_code=code if status != "success" else None,
        )
    state.status = status
    state.stop_reason = code
    return _attach_completed_trace(AnswerResult(
        question=state.original_question, answer=answer, status=status,
        model_calls=state.model_calls, tool_calls=state.tool_calls,
        stop_reason=code,
        routing=RouteSummary(
            route="mcp", intent="mcp_tool", decision_source=source,
            reason_code=decision.reason_code,
        ),
        dialogue=DialogueSummary(
            action=decision.intent.value,
            business_type=decision.business_type.value if decision.business_type else None,
            execution_mode=decision.execution_mode.value, reason_code=decision.reason_code,
        ),
    ), state)


def dispatch_pending_search(
    question: str, session_id: str, ready: SessionBusinessState,
    decision: DialogueDecision, *, turn_id: str, model_calls: int,
    registry: ToolRegistry[TurnState], config: AgentConfig,
    save_transition: Callable[[str, StateTransition], SessionBusinessState],
    commit_turn: Callable[..., AnswerResult],
) -> AnswerResult:
    """READY 之后只派发一次；若最终持久化失败，保留已消费状态。"""
    trace = TraceRecorder()
    state = TurnState(original_question=question, trace_recorder=trace)
    state.model_calls = model_calls
    intent_event = trace.start_event("intent", "program")
    try:
        arguments = _ready_arguments(
            question, session_id, ready, decision, turn_id=turn_id, registry=registry,
        )
        if type(model_calls) is not int or not 0 <= model_calls <= config.max_model_calls:
            raise DialogueContractError("model_call_budget_exhausted")
        if config.max_tool_calls < 1:
            raise DialogueContractError("tool_budget_exhausted")
        canonical = search_success_payload(arguments["query"])
        state.context_chars = len(question) + (2600 + arguments["limit"] * 550 if "limit" in arguments else len(json.dumps(canonical, ensure_ascii=False)))
        if state.context_chars > config.max_context_chars:
            raise DialogueContractError("context_budget_exhausted")
    except (TypeError, ValueError) as exc:
        code = exc.code if isinstance(exc, DialogueContractError) else "browser_invalid_tool_arguments"
        trace.finish_event(intent_event, "error", error_code=code, reason_code=code)
        limited = code.endswith("budget_exhausted")
        result = _result(
            state, decision,
            "本轮执行预算不足，尚未请求打开浏览器。" if limited else
            "搜索请求的参数、来源或工具状态未通过校验，尚未请求打开浏览器。请重新发送完整搜索请求。",
            status="limit_reached" if limited else "service_error", code=code,
        )
        # 被拒绝的跨会话提议不得写入任一会话。
        if ready.session_id != session_id:
            return result
        return commit_turn(
            session_id, state, result,
            (HumanMessage(content=question), AIMessage(content=result.answer)),
            dialogue_decision=decision, turn_id=turn_id, allow_failure=True,
        )

    trace.finish_event(
        intent_event, "success", intent_kind="mcp_tool", route_kind="mcp",
        decision_source="model" if decision.decision_source.value == "model" else "rules",
        reason_code=decision.reason_code, dialogue_action=decision.intent.value,
        business_type=BusinessType.BILIBILI_SEARCH.value,
        execution_mode=decision.execution_mode.value,
    )
    dispatch = mark_dispatching(ready, expected_revision=ready.revision, updated_at=_utc_now())
    try:
        dispatching = save_transition(session_id, dispatch)
    except Exception:
        # 操作尚未尝试。绝不能以持久化失败换取一次未记录的
        # 浏览器启动，也不要在这里重试写入。
        return _result(
            state, decision, "业务状态保存失败，尚未请求打开浏览器。请稍后重新发送完整搜索请求。",
            status="service_error", code="browser_dispatch_state_unsaved",
        )

    tool_event = trace.start_event("tool", "program", tool_name=SEARCH_BILIBILI_TOOL)
    state.tool_calls += 1
    outcome = ExecutionOutcome.UNKNOWN
    code = "browser_result_unconfirmed"
    payload: dict[str, Any] | None = None
    try:
        output = registry.execute(SEARCH_BILIBILI_TOOL, arguments, context=state)
        payload = verified_search_payload(output, arguments)
        if payload is not None:
            outcome = ExecutionOutcome.SUCCEEDED
            code = payload["action"]
        else:
            code = search_error_code(output)
            if code in SEARCH_NOT_DISPATCHED_ERRORS:
                outcome = ExecutionOutcome.FAILED
            elif code in {"timeout", "tool_timeout"}:
                outcome = ExecutionOutcome.TIMED_OUT
    except TimeoutError:
        outcome, code = ExecutionOutcome.TIMED_OUT, "tool_timeout"
    except Exception:
        # 传输断开或回复格式错误并不能证明浏览器启动失败。
        # 已消费的任务绝不会被放回 READY。
        outcome = ExecutionOutcome.UNKNOWN

    success = payload is not None
    if success:
        state.successful_tool_calls += 1
    else:
        state.failed_tool_calls += 1
        state.last_tool_error_code = code
    answer = search_result_answer(arguments["query"], success=success, code=code, payload=payload, limit=arguments.get("limit"))
    trace.finish_event(
        tool_event, "success" if success else "error", result_count=len(payload["items"]) if success and "items" in payload else (1 if success else 0),
        reason_code=code, error_code=None if success else code,
    )
    final_transition = apply_execution_outcome(
        dispatching, expected_revision=dispatching.revision, outcome=outcome,
        updated_at=_utc_now(),
    )
    result = _result(state, decision, answer, status="success" if success else "service_error", code=code)
    # 已消费的 DISPATCHING 状态已经持久化。若这次提交失败，
    # 调用方报告持久化失败，而不复活或重放它。
    return commit_turn(
        session_id, state, result,
        (HumanMessage(content=question), AIMessage(content=answer)),
        business_transition=final_transition, dialogue_decision=decision,
        turn_id=turn_id, allow_failure=True,
    )
