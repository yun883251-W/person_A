"""在检索和工具派发之前进行纯对话意图解析。

解析器消费调用方持有的快照，返回经校验的决定以及可选的状态转换提议。
它绝不执行能力、持久化状态、初始化模型，也不改变测评状态。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from browser_mcp.contracts import SEARCH_BILIBILI_TOOL, validate_search_arguments

from knowledge_agent.agent import assessment
from knowledge_agent.agent.planning import planner
from ...config import IntentSettings, get_intent_settings
from knowledge_agent.agent.browser.actions import extract_bilibili_topic, is_explicit_bilibili_topic
from .intent_types import IntentDecision, IntentKind, IntentModelBudgetError, McpCapability
from .rules import identify_intent_rules
from .classifier import classify_rule_intent
from .contracts import (
    BusinessFocus, BusinessPhase, BusinessType, DecisionSource,
    DialogueContractError, DialogueDecision, DialogueIntent, ExecutionMode,
    SessionBusinessState, SlotBinding, SlotSource,
    StateTransition, empty_business_state,
    validate_business_state, validate_dialogue_decision,
)
from .transitions import (
    cancel_current_business, merge_pending_transition,
    propose_transition, record_unresolved_clarification, start_business_transition,
)


_BARE_CANCEL_RE = re.compile(r"^(?:请)?(?:取消|退出|结束)(?:当前|这个|该)?(?:任务|操作|待办)?[。.!！?？]*$")
_BARE_PAUSE_RE = re.compile(r"^(?:请)?(?:暂停|先停一下)(?:当前|这个|该)?(?:任务|测试|答题)?[。.!！?？]*$")
_BARE_RESUME_RE = re.compile(r"^(?:请)?(?:继续|恢复|下一题)(?:当前|这个|该)?(?:任务|测试|答题)?[。.!！?？]*$")
_VAGUE_SLOT_VALUES = frozenset({
    "", "随便", "任意", "都行", "它", "这个", "那个", "这些", "那些",
    "上面", "刚才那个", "whatever", "anything", "it", "that",
})
_SCORE_RE = re.compile(r"^(?:我选|选择|选)?\s*([1-5])(?:分)?[。.!！]?$")
_TAGGED_ANSWER_RE = re.compile(r"^MBTI答题\s+[a-f0-9]{32}\s+([0-9]{1,2})\s+([1-5])$", re.I)
_ANSWER_LABELS = {
    "完全不符合": 1, "不太符合": 2, "中立": 3, "中立/不确定": 3,
    "不确定": 3, "比较符合": 4, "完全符合": 5,
}


@dataclass(frozen=True, slots=True)
class DialogueResolution:
    """可信的路由决定，以及可选的、未提交的状态变更。"""

    decision: DialogueDecision
    state_transition: StateTransition | None = None
    model_calls: int = 0


def _validate_inputs(
    question: str,
    session_id: str | None,
    state: SessionBusinessState | None,
    assessment_state: Mapping[str, Any] | None,
    history: Sequence[object],
    capabilities: Sequence[McpCapability],
    remaining_model_calls: int,
    model_invoker: Callable[[Sequence[object]], object] | None,
    updated_at: str,
) -> tuple[str, SessionBusinessState | None, Mapping[str, Any] | None, tuple[object, ...], tuple[McpCapability, ...]]:
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be blank")
    if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
        raise ValueError("session_id must be a nonblank string or None")
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError("history must be a sequence")
    if isinstance(capabilities, (str, bytes)) or not isinstance(capabilities, Sequence):
        raise TypeError("capabilities must be a sequence")
    capability_snapshot = tuple(capabilities)
    if any(not isinstance(item, McpCapability) for item in capability_snapshot):
        raise TypeError("capabilities must contain McpCapability snapshots")
    names = tuple(item.name for item in capability_snapshot)
    if len(names) != len(set(names)):
        raise ValueError("capability names must be unique")
    if type(remaining_model_calls) is not int or remaining_model_calls < 0:
        raise ValueError("remaining_model_calls must be a nonnegative integer")
    if model_invoker is not None and not callable(model_invoker):
        raise TypeError("model_invoker must be callable or None")
    if not isinstance(updated_at, str) or not updated_at:
        raise ValueError("updated_at must be a timestamp string")

    if session_id is None:
        if state is not None:
            raise DialogueContractError("cross_session_state")
        state_snapshot = None
    else:
        state_snapshot = empty_business_state(session_id, updated_at=updated_at) if state is None else validate_business_state(state)
        if state_snapshot.session_id != session_id:
            raise DialogueContractError("cross_session_state")

    assessment_snapshot: Mapping[str, Any] | None = None
    if assessment_state is not None:
        assessment_snapshot = assessment.validate_snapshot(dict(assessment_state))
    if (
        state_snapshot is not None and state_snapshot.focus is not None
        and state_snapshot.focus.business_type is BusinessType.MBTI
        and state_snapshot.focus.test_id is not None
        and (assessment_snapshot is None or assessment_snapshot["test_id"] != state_snapshot.focus.test_id)
    ):
        raise DialogueContractError("mbti_test_mismatch")
    return cleaned, state_snapshot, assessment_snapshot, tuple(history), capability_snapshot


def _source_ids(session_id: str | None, turn_id: str | None) -> tuple[str, str]:
    if turn_id is not None and (not isinstance(turn_id, str) or not turn_id.strip()):
        raise ValueError("turn_id must be a nonblank string or None")
    return session_id or "stateless", turn_id.strip() if turn_id is not None else "current"


def _bindings(
    slots: Mapping[str, Any], *, raw_input: str, session_id: str, turn_id: str,
) -> tuple[SlotBinding, ...]:
    return tuple(
        SlotBinding(
            slot_name=name, value=value, raw_input=raw_input,
            session_id=session_id, source_turn_id=turn_id,
            source=SlotSource.CURRENT_USER_INPUT,
        )
        for name, value in slots.items()
    )


def _decision(
    *, intent: DialogueIntent, revision: int,
    business_type: BusinessType | None = None,
    execution_mode: ExecutionMode = ExecutionMode.SINGLE,
    slots: Mapping[str, Any] | None = None,
    missing_slots: Sequence[str] = (), clarification: str = "",
    source: DecisionSource = DecisionSource.RULES, reason_code: str,
    allowed_tools: Sequence[str] = (), raw_plan_request: str = "",
    raw_input: str = "", session_id: str = "stateless",
    turn_id: str = "current", retrieval_query: str = "",
    direct_response_kind: str = "", authorized_user_input: str = "",
) -> DialogueDecision:
    confirmed = dict(slots or {})
    value = DialogueDecision(
        intent=intent, business_type=business_type,
        execution_mode=execution_mode, confirmed_slots=confirmed,
        missing_slots=tuple(missing_slots),
        slot_bindings=_bindings(
            confirmed, raw_input=raw_input,
            session_id=session_id, turn_id=turn_id,
        ),
        clarification_question=clarification, decision_source=source,
        reason_code=reason_code, expected_revision=revision,
        allowed_tool_names=tuple(allowed_tools),
        raw_plan_request=raw_plan_request,
        retrieval_query=retrieval_query, direct_response_kind=direct_response_kind,
        authorized_user_input=authorized_user_input,
    )
    return validate_dialogue_decision(value)


def _unclear(
    revision: int, reason_code: str, clarification: str, *,
    source: DecisionSource = DecisionSource.FALLBACK,
) -> DialogueDecision:
    return _decision(
        intent=DialogueIntent.UNCLEAR, revision=revision,
        clarification=clarification, source=source,
        reason_code=reason_code,
    )


def _supplemental_query(question: str) -> str | None:
    normalized = question.strip().strip("。.!！?？").casefold()
    if (normalized in _VAGUE_SLOT_VALUES or len(question) > 200
            or not is_explicit_bilibili_topic(question)):
        return None
    if re.search(
        r"(?:你能做什么|怎么用|如何|为什么|是什么|什么是|解释|介绍|说明|分析|比较|总结|"
        r"取消|暂停|继续|开始|调用|执行|打开|删除|发送|上传|下载|"
        r"不要|不用|不想|如果|假如|假设|然后|同时|换个|换一个|"
        r"[，,；;。\r\n]|在\s*[bB]\s*站|\b(?:cancel|pause|resume|explain|open|delete|if)\b)",
        question,
        re.IGNORECASE,
    ):
        return None
    try:
        return validate_search_arguments({"query": question.strip()})["query"]
    except (TypeError, ValueError):
        return None


def _assessment_slots(
    action: str, question: str, snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    slots: dict[str, Any] = {"action": action}
    if snapshot is not None:
        slots["test_id"] = snapshot["test_id"]
    if action == "answer":
        tagged = _TAGGED_ANSWER_RE.fullmatch(question.strip())
        score_match = _SCORE_RE.fullmatch(question.strip())
        score = (
            int(tagged.group(2)) if tagged else
            int(score_match.group(1)) if score_match else
            _ANSWER_LABELS.get(question.strip())
        )
        if score is not None:
            slots["answer"] = score
        if tagged:
            slots["question_number"] = int(tagged.group(1)) + 1
        elif snapshot is not None:
            slots["question_number"] = len(snapshot["answers"]) + 1
    return slots


def _state_transition(
    state: SessionBusinessState | None, decision: DialogueDecision, *,
    question: str, turn_id: str, updated_at: str,
) -> StateTransition | None:
    if state is None:
        return None
    if decision.intent is DialogueIntent.CANCEL_BUSINESS:
        test_id = decision.confirmed_slots.get("test_id")
        return cancel_current_business(
            state, expected_revision=decision.expected_revision,
            updated_at=updated_at,
            mbti_test_id=test_id if isinstance(test_id, str) else None,
        )
    if decision.intent is DialogueIntent.CONTINUE_BUSINESS and state.pending is not None:
        return merge_pending_transition(state, decision, updated_at=updated_at)
    if decision.intent is not DialogueIntent.NEW_BUSINESS or decision.execution_mode is ExecutionMode.PLAN:
        return None
    if state.focus is not None and state.focus.phase is BusinessPhase.DISPATCHING:
        return None
    if decision.business_type is BusinessType.MCP_TOOL:
        return None
    if decision.business_type is BusinessType.KNOWLEDGE and state.pending is not None:
        return propose_transition(
            state, decision,
            focus=BusinessFocus(BusinessType.KNOWLEDGE, BusinessPhase.ACTIVE),
            pending=None, reason_code="business_started",
            updated_at=updated_at, next_actions=("answer_knowledge",),
        )
    return start_business_transition(
        state, decision, original_request=question,
        request_turn_id=turn_id, updated_at=updated_at,
    )


def _from_classification(
    classified: IntentDecision, *, question: str, revision: int,
    session_id: str, turn_id: str,
) -> DialogueDecision:
    """接受一次分类器结果；不保留更早一轮的覆盖值。"""
    source = DecisionSource(classified.decision_source)
    common = {"revision": revision, "source": source, "reason_code": classified.reason_code}
    if classified.kind in {IntentKind.GREETING, IntentKind.SCOPE}:
        return _decision(
            intent=DialogueIntent.CHITCHAT,
            direct_response_kind=classified.kind.value, **common,
        )
    if classified.kind is IntentKind.OUT_OF_SCOPE:
        return _decision(intent=DialogueIntent.UNSUPPORTED, **common)
    if classified.kind is IntentKind.NEEDS_CLARIFICATION:
        return _unclear(
            revision, classified.reason_code,
            classified.clarification_question or "请明确说明你本轮想完成的任务。",
            source=source,
        )
    if classified.kind is IntentKind.KNOWLEDGE:
        return _decision(
            intent=DialogueIntent.NEW_BUSINESS, business_type=BusinessType.KNOWLEDGE,
            retrieval_query=classified.retrieval_query or question,
            **common,
        )
    if classified.allowed_tool_names == (SEARCH_BILIBILI_TOOL,):
        topic = classified.search_topic or extract_bilibili_topic(question)
        if not topic or topic not in question or not is_explicit_bilibili_topic(topic):
            return _unclear(
                revision, "browser_bilibili_unconfirmed_topic",
                "请在完整的 B 站搜索请求中写出具体主题或作品名称。",
                source=source,
            )
        topic = validate_search_arguments({"query": topic})["query"]
        if classified.authorized_user_input != question:
            raise DialogueContractError("invalid_authorized_input")
        return _decision(
            intent=DialogueIntent.NEW_BUSINESS, business_type=BusinessType.BILIBILI_SEARCH,
            slots={"query": topic, **({"limit": classified.search_limit} if classified.search_limit is not None else {})}, allowed_tools=classified.allowed_tool_names,
            raw_input=question, session_id=session_id, turn_id=turn_id,
            authorized_user_input=classified.authorized_user_input, **common,
        )
    return _decision(
        intent=DialogueIntent.NEW_BUSINESS, business_type=BusinessType.MCP_TOOL,
        allowed_tools=classified.allowed_tool_names,
        authorized_user_input=classified.authorized_user_input, **common,
    )


def resolve_dialogue(
    question: str, *, session_id: str | None,
    state: SessionBusinessState | None,
    assessment_state: Mapping[str, Any] | None,
    history: Sequence[object], capabilities: Sequence[McpCapability],
    remaining_model_calls: int,
    model_invoker: Callable[[Sequence[object]], object] | None,
    updated_at: str, turn_id: str | None = None,
    settings: IntentSettings | None = None,
    max_history_turns: int = 6, max_context_chars: int = 4000,
) -> DialogueResolution:
    """无副作用地解析恰好一条原始用户消息。

    没有 ``session_id`` 时，完整请求仍正常路由；而不完整请求
    会被澄清，但不会创建共享的待处理状态。
    调用方必须先提交返回的转换，然后才能派发任何操作。
    """

    cleaned, state, assessment_snapshot, history, capabilities = _validate_inputs(
        question, session_id, state, assessment_state, history, capabilities,
        remaining_model_calls, model_invoker, updated_at,
    )
    if settings is None:
        settings = get_intent_settings()
    if not isinstance(settings, IntentSettings):
        raise TypeError("settings must be IntentSettings")
    binding_session_id, source_turn_id = _source_ids(session_id, turn_id)
    revision = state.revision if state is not None else 0
    focus = state.focus if state is not None else None

    explicit_mbti = re.search(r"mbti|人格测试|性格测试", cleaned, re.I) is not None
    assessment_context = (
        assessment_snapshot
        if explicit_mbti or (focus is not None and focus.business_type is BusinessType.MBTI)
        else None
    )
    assessment_clarification = (
        assessment.request_clarification(cleaned, assessment_context)
        if focus is None or focus.business_type is BusinessType.MBTI or explicit_mbti
        else None
    )
    if assessment_clarification is not None:
        return DialogueResolution(_unclear(
            revision, "assessment_request_boundary", assessment_clarification,
        ))

    action = assessment.request_action(cleaned, assessment_context)
    bare_cancel = _BARE_CANCEL_RE.fullmatch(cleaned) is not None
    bare_pause = _BARE_PAUSE_RE.fullmatch(cleaned) is not None
    bare_resume = _BARE_RESUME_RE.fullmatch(cleaned) is not None
    if bare_cancel and focus is not None:
        business = focus.business_type
        slots = _assessment_slots("cancel", cleaned, assessment_context) if business is BusinessType.MBTI else {}
        decision = _decision(
            intent=DialogueIntent.CANCEL_BUSINESS,
            business_type=business, revision=revision, slots=slots,
            reason_code="pending_cancelled" if state and state.pending else "business_cleared",
            raw_input=question, session_id=binding_session_id,
            turn_id=source_turn_id,
        )
        try:
            transition = _state_transition(
                state, decision, question=question,
                turn_id=source_turn_id, updated_at=updated_at,
            )
        except DialogueContractError as exc:
            decision = _unclear(
                revision, exc.code,
                "当前业务不能取消；如果要开始新任务，请直接写出完整请求。",
            )
            transition = None
        return DialogueResolution(decision, transition)
    if (bare_cancel or bare_pause or bare_resume) and focus is None and action is None:
        return DialogueResolution(_unclear(
            revision, "no_active_business",
            "当前没有可继续、暂停或取消的业务。请直接写出完整请求。",
        ))
    if (bare_pause or bare_resume) and focus is not None and focus.business_type is not BusinessType.MBTI:
        return DialogueResolution(_unclear(
            revision, "ambiguous_business_target",
            "当前业务不支持暂停或裸“继续”；请直接写出新的完整请求。",
        ))

    if action is not None:
        slots = _assessment_slots(action, cleaned, assessment_context)
        if action == "cancel":
            dialogue_intent = DialogueIntent.CANCEL_BUSINESS
        elif action == "pause":
            dialogue_intent = DialogueIntent.PAUSE_BUSINESS
        elif action == "resume":
            dialogue_intent = DialogueIntent.RESUME_BUSINESS
        elif focus is not None and focus.business_type is BusinessType.MBTI and action != "restart":
            dialogue_intent = DialogueIntent.CONTINUE_BUSINESS
        else:
            dialogue_intent = DialogueIntent.NEW_BUSINESS
        decision = _decision(
            intent=dialogue_intent, business_type=BusinessType.MBTI,
            revision=revision, slots=slots, reason_code=f"mbti_{action}",
            raw_input=question, session_id=binding_session_id,
            turn_id=source_turn_id,
        )
        transition = None
        if action == "cancel" and state is not None and focus is not None and focus.business_type is BusinessType.MBTI:
            try:
                transition = _state_transition(
                    state, decision, question=question,
                    turn_id=source_turn_id, updated_at=updated_at,
                )
            except DialogueContractError:
                transition = None
        return DialogueResolution(decision, transition)

    semantic_plan = settings.semantic_mode == "hybrid" and planner.semantic_plan_candidate(cleaned)
    if planner.should_plan(cleaned) or semantic_plan:
        decision = _decision(
            intent=DialogueIntent.NEW_BUSINESS, revision=revision,
            execution_mode=ExecutionMode.PLAN,
            reason_code="semantic_plan_candidate" if semantic_plan else "explicit_multi_task", raw_plan_request=question,
        )
        transition = None
        if state is not None and state.pending is not None:
            transition = cancel_current_business(
                state, expected_revision=revision, updated_at=updated_at,
            )
        return DialogueResolution(decision, transition)

    # 对已知缺失字段的简短回答会在模型分类之前处理。
    rules = identify_intent_rules(
        cleaned, context=history, available_tools=capabilities,
        assessment_state=None, max_history_turns=max_history_turns,
        max_context_chars=max_context_chars,
    )
    if (
        state is not None and state.pending is not None
        and state.focus is not None
        and state.focus.phase is BusinessPhase.AWAITING_INPUT
        and state.pending.business_type is BusinessType.BILIBILI_SEARCH
        and state.pending.missing_slots == ("query",)
    ):
        # 标题可能不属于知识领域，但仍能回答
        # 尚未解决的搜索主题问题。
        may_fill = rules.reason_code in {"default_knowledge", "out_of_scope_marker"}
        query = _supplemental_query(cleaned) if may_fill else None
        if query is not None:
            if SEARCH_BILIBILI_TOOL not in {item.name for item in capabilities}:
                return DialogueResolution(_decision(
                    intent=DialogueIntent.UNSUPPORTED, revision=revision,
                    source=DecisionSource.FALLBACK, reason_code="tool_unavailable",
                ))
            decision = _decision(
                intent=DialogueIntent.CONTINUE_BUSINESS,
                business_type=BusinessType.BILIBILI_SEARCH,
                revision=revision, slots={"query": query},
                reason_code="slot_filled", allowed_tools=(SEARCH_BILIBILI_TOOL,),
                raw_input=question, session_id=binding_session_id, turn_id=source_turn_id,
            )
            transition = _state_transition(
                state, decision, question=question,
                turn_id=source_turn_id, updated_at=updated_at,
            )
            return DialogueResolution(decision, transition)
        normalized = cleaned.strip("。.!！?？").casefold()
        if (
            normalized in _VAGUE_SLOT_VALUES
            or (may_fill and not is_explicit_bilibili_topic(cleaned))
        ):
            decision = _unclear(
                revision, "clarification_unresolved",
                "请补充具体搜索主题或作品名称，例如“火影忍者”；也可以取消这次搜索。",
                source=DecisionSource.RULES,
            )
            transition = record_unresolved_clarification(
                state, expected_revision=revision,
                clarification_question=decision.clarification_question, updated_at=updated_at,
            )
            return DialogueResolution(decision, transition)

    if rules.kind is IntentKind.NEEDS_CLARIFICATION and rules.reason_code == "browser_bilibili_missing_topic":
        if SEARCH_BILIBILI_TOOL not in {item.name for item in capabilities}:
            return DialogueResolution(_decision(
                intent=DialogueIntent.UNSUPPORTED, revision=revision,
                source=DecisionSource.FALLBACK, reason_code="tool_unavailable",
            ))
        clarification = (
            "你想在 B 站搜索什么主题或作品？可以直接回复名称，例如“火影忍者”。"
            if session_id is not None else
            "请补全主题后重新发送完整请求，例如“帮我在 B 站搜索火影忍者”。"
        )
        decision = _decision(
            intent=DialogueIntent.NEW_BUSINESS,
            business_type=BusinessType.BILIBILI_SEARCH,
            revision=revision, missing_slots=("query",),
            clarification=clarification, reason_code=rules.reason_code,
            raw_input=question, session_id=binding_session_id, turn_id=source_turn_id,
            authorized_user_input=cleaned,
        )
        transition = _state_transition(
            state, decision, question=question,
            turn_id=source_turn_id, updated_at=updated_at,
        )
        return DialogueResolution(decision, transition)

    model_calls = 0

    def invoke_classifier(messages: Sequence[object]) -> object:
        nonlocal model_calls
        # 为回答/执行保留一次调用。通用分类和浏览器
        # 分类对本次输入互斥。
        if model_invoker is None or model_calls >= max(0, remaining_model_calls - 1):
            raise IntentModelBudgetError()
        model_calls += 1
        return model_invoker(messages)

    classified = (
        classify_rule_intent(
            cleaned, rules, context=history, available_tools=capabilities,
            settings=settings, model_invoker=invoke_classifier,
            max_history_turns=max_history_turns, max_context_chars=max_context_chars,
        ) if model_invoker is not None else rules
    )
    decision = _from_classification(
        classified, question=cleaned, revision=revision,
        session_id=binding_session_id, turn_id=source_turn_id,
    )
    if (
        decision.intent is DialogueIntent.NEW_BUSINESS
        and decision.business_type is BusinessType.KNOWLEDGE
        and focus is not None and focus.business_type is BusinessType.KNOWLEDGE
        and (classified.reason_code in {"knowledge_followup", "knowledge_followup_semantic"}
             or classified.reason_code.startswith("followup_semantic_"))
    ):
        decision = _decision(
            intent=DialogueIntent.CONTINUE_BUSINESS,
            business_type=BusinessType.KNOWLEDGE,
            revision=revision, reason_code=classified.reason_code,
            source=DecisionSource(classified.decision_source),
            retrieval_query=classified.retrieval_query,
        )
    transition = _state_transition(
        state, decision, question=question,
        turn_id=source_turn_id, updated_at=updated_at,
    )
    if (
        decision.intent is DialogueIntent.UNCLEAR
        and state is not None and state.pending is not None
        and state.focus is not None and state.focus.phase is BusinessPhase.AWAITING_INPUT
        and decision.decision_source is not DecisionSource.FALLBACK
    ):
        transition = record_unresolved_clarification(
            state, expected_revision=revision,
            clarification_question=decision.clarification_question,
            updated_at=updated_at,
        )
    return DialogueResolution(decision, transition, model_calls)


__all__ = ["DialogueResolution", "resolve_dialogue"]
