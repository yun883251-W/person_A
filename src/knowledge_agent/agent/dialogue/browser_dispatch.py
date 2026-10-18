"""Execute one host-confirmed Bilibili task with durable dispatch protection.

The caller has committed the READY transition. This boundary revalidates the
original request and current slot provenance, consumes the pending task before
the external call, and accepts only the browser service's exact result contract.
It does not classify with a model, invent tool calls, or replay uncertain work.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from browser_mcp.contracts import SEARCH_BILIBILI_TOOL, build_search_url, validate_search_arguments

from ...config import AgentConfig
from ...schemas import AnswerResult, AnswerStatus, DialogueSummary, RouteSummary
from ...tools.registry import ToolRegistry
from knowledge_agent.agent.answering.answers import _attach_completed_trace
from knowledge_agent.agent.browser.actions import (
    classify_bilibili_request, extract_bilibili_topic, is_explicit_bilibili_topic,
)
from knowledge_agent.agent.browser.execution import _verified_payload
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


_NOT_DISPATCHED_ERRORS = frozenset({
    "unknown_tool", "unavailable", "invalid_tool_arguments", "invalid_arguments",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ready_arguments(
    question: str, session_id: str, ready: SessionBusinessState,
    decision: DialogueDecision, *, turn_id: str, registry: ToolRegistry[TurnState],
) -> dict[str, str]:
    """Recheck current-turn authority independently of the earlier classifier."""
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

    # READY was committed from this decision, so execution uses the new revision
    # only after proving that it is exactly the next state of this decision.
    current_decision = replace(decision, expected_revision=ready.revision)
    proposal = prepare_execution(ready, current_decision)
    arguments = validate_search_arguments(proposal.arguments)
    if arguments != dict(pending.confirmed_slots) or len(proposal.slot_bindings) != 1:
        raise DialogueContractError("browser_pending_authorization_invalid")
    binding = proposal.slot_bindings[0]
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
    if decision.intent is DialogueIntent.NEW_BUSINESS:
        if (
            pending.request_turn_id != turn_id
            or pending.original_request.strip() != question.strip()
        ):
            raise DialogueContractError("browser_pending_authorization_invalid")
        literal_topic = extract_bilibili_topic(pending.original_request)
        if literal_topic != topic:
            # The semantic classifier already passed the host's explicit-action
            # boundary. Retain that exact current request and literal topic;
            # generic clarification or historical model content is insufficient.
            semantic_authorized = (
                original_action is not None
                and original_action.kind == "semantic_candidate"
                and decision.reason_code == "browser_bilibili_semantic_search"
                and decision.decision_source.value == "model"
                and decision.authorized_user_input.strip() == question.strip()
            )
            if not semantic_authorized:
                raise DialogueContractError("browser_pending_authorization_invalid")
    elif (
        pending.request_turn_id == turn_id
        or original_action is None
        or original_action.kind != "needs_clarification"
        or original_action.reason_code != "browser_bilibili_missing_topic"
        or topic != question.strip()
    ):
        # Cross-turn execution is only a literal answer to an established,
        # specific missing-topic question, never a guessed business continuation.
        raise DialogueContractError("browser_pending_authorization_invalid")

    registration = registry.get(SEARCH_BILIBILI_TOOL)
    if registration is None:
        raise DialogueContractError("tool_unavailable")
    try:
        if registration.argument_validator is not None:
            checked = registration.argument_validator(arguments)
        else:
            checked = registration.definition.get_input_schema().model_validate(arguments).model_dump()
    except Exception as exc:
        # MCP adapters and local schemas expose different exception classes.
        # A rejected schema is still a pre-dispatch failure, not an HTTP crash.
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
    """Dispatch once after READY; preserve consumed state if final persistence fails."""
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
        url = build_search_url(arguments["query"])
        canonical = {
            "status": "success", "action": "browser_launch_requested", "site": "bilibili",
            "query": arguments["query"], "url": url,
            "browser_launch_requested": True, "page_verified": False,
        }
        state.context_chars = len(question) + len(json.dumps(canonical, ensure_ascii=False))
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
        # A rejected cross-session proposal must not be written to either session.
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
        # The action has not been attempted. Never trade persistence failure for
        # an unrecorded browser launch, and do not retry the write here.
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
        payload = _verified_payload(output, arguments)
        if payload is not None:
            outcome = ExecutionOutcome.SUCCEEDED
            code = "browser_launch_requested"
        elif isinstance(output, Mapping) and output.get("status") == "error":
            error = output.get("error")
            error_code = error.get("code") if isinstance(error, Mapping) else None
            if error_code in _NOT_DISPATCHED_ERRORS:
                outcome, code = ExecutionOutcome.FAILED, str(error_code)
            elif error_code in {"timeout", "tool_timeout"}:
                outcome, code = ExecutionOutcome.TIMED_OUT, str(error_code)
    except TimeoutError:
        outcome, code = ExecutionOutcome.TIMED_OUT, "tool_timeout"
    except Exception:
        # A disconnected transport or malformed reply does not prove that the
        # browser launch failed. The consumed task is never put back in READY.
        outcome = ExecutionOutcome.UNKNOWN

    success = payload is not None
    if success:
        state.successful_tool_calls += 1
        answer = (
            "已向本机默认浏览器发送 B 站搜索打开请求。\n\n"
            f"关键词：{arguments['query']}\n\n[B站搜索页]({url})\n\n"
            "尚未检查页面是否加载，也未读取视频结果。"
        )
    else:
        state.failed_tool_calls += 1
        state.last_tool_error_code = code
        answer = (
            "B 站搜索未能执行，本轮没有重复调用。"
            if outcome is ExecutionOutcome.FAILED else
            "浏览器打开请求未获成功确认，本轮没有重复调用。请先查看浏览器窗口。"
        ) + f"\n\n关键词：{arguments['query']}\n\n也可以打开：[B站搜索页]({url})。"
    trace.finish_event(
        tool_event, "success" if success else "error", result_count=1 if success else 0,
        reason_code=code, error_code=None if success else code,
    )
    final_transition = apply_execution_outcome(
        dispatching, expected_revision=dispatching.revision, outcome=outcome,
        updated_at=_utc_now(),
    )
    result = _result(state, decision, answer, status="success" if success else "service_error", code=code)
    # The consumed DISPATCHING state is already durable. If this commit fails,
    # the caller reports persistence failure without resurrecting/replaying it.
    return commit_turn(
        session_id, state, result,
        (HumanMessage(content=question), AIMessage(content=answer)),
        business_transition=final_transition, dialogue_decision=decision,
        turn_id=turn_id, allow_failure=True,
    )
