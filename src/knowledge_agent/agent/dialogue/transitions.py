"""会话业务状态转换。

功能：根据已校验决策和实际执行结果，产生可提交的新状态。
输入：旧 SessionBusinessState、可信 DialogueDecision、执行事实和调用方时间。
输出目标：StateTransition；状态版本由存储层检查，提交成功后才能更新 Agent 内存。
本模块只做纯转换，不识别自然语言、不调用模型、MCP 或 SQLite。

状态更改仅以提案返回；调用方必须先持久化，再发布到内存。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any

from .contracts import (
    BusinessFocus,
    BusinessPhase,
    BusinessType,
    CommitTiming,
    DialogueContractError,
    DialogueDecision,
    DialogueIntent,
    ExecutionMode,
    PendingBusiness,
    SessionBusinessState,
    SlotBinding,
    StateTransition,
    empty_business_state,
    required_business_slots,
    state_from_payload,
    validate_business_state,
    validate_business_slots,
    validate_dialogue_decision,
    validate_state_transition,
)


@dataclass(frozen=True, slots=True)
class _TransitionRule:
    starts: frozenset[BusinessPhase]
    ends: frozenset[BusinessPhase]
    requires_execution: bool = False


_ALL_PHASES = frozenset(BusinessPhase)
_TRANSITION_RULES: Mapping[str, _TransitionRule] = MappingProxyType({
    "pending_created": _TransitionRule(
        _ALL_PHASES - {BusinessPhase.DISPATCHING},
        frozenset({BusinessPhase.AWAITING_INPUT}),
    ),
    "slot_filled": _TransitionRule(
        frozenset({BusinessPhase.AWAITING_INPUT}),
        frozenset({BusinessPhase.AWAITING_INPUT, BusinessPhase.READY}),
    ),
    "clarification_unresolved": _TransitionRule(
        frozenset({BusinessPhase.AWAITING_INPUT}),
        frozenset({BusinessPhase.AWAITING_INPUT}),
    ),
    "pending_cancelled": _TransitionRule(
        frozenset({BusinessPhase.AWAITING_INPUT, BusinessPhase.READY}),
        frozenset({BusinessPhase.IDLE, BusinessPhase.CANCELLED}),
    ),
    "pending_replaced": _TransitionRule(
        frozenset({BusinessPhase.AWAITING_INPUT, BusinessPhase.READY}),
        frozenset({BusinessPhase.AWAITING_INPUT, BusinessPhase.READY}),
    ),
    "business_cleared": _TransitionRule(
        _ALL_PHASES - {BusinessPhase.IDLE, BusinessPhase.DISPATCHING},
        frozenset({BusinessPhase.IDLE, BusinessPhase.CANCELLED}),
    ),
    "business_started": _TransitionRule(
        _ALL_PHASES - {BusinessPhase.DISPATCHING},
        frozenset({BusinessPhase.READY, BusinessPhase.ACTIVE, BusinessPhase.COMPLETED}),
    ),
    "business_continued": _TransitionRule(
        frozenset({BusinessPhase.ACTIVE}),
        frozenset({BusinessPhase.ACTIVE, BusinessPhase.COMPLETED}),
    ),
    "business_paused": _TransitionRule(
        frozenset({BusinessPhase.ACTIVE}),
        frozenset({BusinessPhase.PAUSED}),
    ),
    "business_resumed": _TransitionRule(
        frozenset({BusinessPhase.PAUSED}),
        frozenset({BusinessPhase.ACTIVE}),
    ),
    "dispatch_started": _TransitionRule(
        frozenset({BusinessPhase.READY}),
        frozenset({BusinessPhase.DISPATCHING}),
        requires_execution=True,
    ),
    "execution_completed": _TransitionRule(
        frozenset({BusinessPhase.ACTIVE, BusinessPhase.DISPATCHING}),
        frozenset({BusinessPhase.COMPLETED, BusinessPhase.ACTIVE, BusinessPhase.PAUSED}),
    ),
    "execution_unknown": _TransitionRule(
        frozenset({BusinessPhase.DISPATCHING}),
        frozenset({BusinessPhase.EXECUTION_UNKNOWN}),
    ),
    "execution_failed": _TransitionRule(
        frozenset({BusinessPhase.DISPATCHING}),
        frozenset({BusinessPhase.COMPLETED}),
    ),
    "execution_timed_out": _TransitionRule(
        frozenset({BusinessPhase.DISPATCHING}),
        frozenset({BusinessPhase.EXECUTION_UNKNOWN}),
    ),
})


def _phase(state: SessionBusinessState) -> BusinessPhase:
    return state.focus.phase if state.focus is not None else BusinessPhase.IDLE


def _build_transition(
    previous: SessionBusinessState,
    *,
    expected_revision: int,
    focus: BusinessFocus | None,
    pending: PendingBusiness | None,
    reason_code: str,
    updated_at: str,
    next_actions: Sequence[str] = (),
    commit_timing: CommitTiming = CommitTiming.WITH_TURN,
) -> StateTransition:
    previous = validate_business_state(previous)
    if type(expected_revision) is not int or expected_revision != previous.revision:
        raise DialogueContractError("state_conflict")
    if pending is not None and pending.session_id != previous.session_id:
        raise DialogueContractError("cross_session_state")
    rule = _TRANSITION_RULES.get(reason_code)
    if rule is None:
        raise DialogueContractError("unknown_transition_reason")
    next_phase = focus.phase if focus is not None else BusinessPhase.IDLE
    if _phase(previous) not in rule.starts or next_phase not in rule.ends:
        raise DialogueContractError("invalid_state_transition")
    if rule.requires_execution != (commit_timing is CommitTiming.BEFORE_DISPATCH):
        raise DialogueContractError("invalid_commit_timing")

    next_state = replace(
        previous,
        revision=previous.revision + 1,
        focus=focus,
        pending=pending,
        updated_at=updated_at,
    )
    transition = StateTransition(
        expected_revision=previous.revision,
        next_state=next_state,
        reason_code=reason_code,
        next_actions=tuple(next_actions),
        commit_timing=commit_timing,
    )
    return validate_state_transition(previous, transition)


def propose_transition(
    previous: SessionBusinessState,
    decision: DialogueDecision,
    *,
    focus: BusinessFocus | None,
    pending: PendingBusiness | None,
    reason_code: str,
    updated_at: str,
    next_actions: Sequence[str] = (),
    commit_timing: CommitTiming = CommitTiming.WITH_TURN,
    tool_schema: Mapping[str, object] | None = None,
) -> StateTransition:
    """Build a validated, copy-on-write state proposal without committing it."""

    previous = validate_business_state(previous)
    decision = validate_dialogue_decision(decision, tool_schema=tool_schema)
    if decision.expected_revision != previous.revision:
        raise DialogueContractError("state_conflict")
    if any(binding.session_id != previous.session_id for binding in decision.slot_bindings):
        raise DialogueContractError("cross_session_state")
    return _build_transition(
        previous,
        expected_revision=decision.expected_revision,
        focus=focus,
        pending=pending,
        reason_code=reason_code,
        next_actions=tuple(next_actions),
        commit_timing=commit_timing,
        updated_at=updated_at,
    )


def transition_requires_execution(transition: StateTransition) -> bool:
    """Return the execution policy attached to a validated transition reason."""

    rule = _TRANSITION_RULES.get(transition.reason_code)
    if rule is None:
        raise DialogueContractError("unknown_transition_reason")
    return rule.requires_execution

def _validate_decision_for_state(
    previous: SessionBusinessState,
    decision: DialogueDecision,
    *,
    tool_schema: Mapping[str, Any] | None = None,
) -> DialogueDecision:
    previous = validate_business_state(previous)
    decision = validate_dialogue_decision(decision, tool_schema=tool_schema)
    if decision.expected_revision != previous.revision:
        raise DialogueContractError("state_conflict")
    if any(binding.session_id != previous.session_id for binding in decision.slot_bindings):
        raise DialogueContractError("cross_session_state")
    return decision


def _pending_from_decision(
    state: SessionBusinessState,
    decision: DialogueDecision,
    *,
    original_request: str,
    request_turn_id: str,
    clarification_count: int,
) -> PendingBusiness:
    if decision.business_type not in {BusinessType.BILIBILI_SEARCH, BusinessType.MBTI}:
        raise DialogueContractError("business_unsupported")
    return PendingBusiness(
        session_id=state.session_id,
        business_type=decision.business_type,
        original_request=original_request,
        request_turn_id=request_turn_id,
        confirmed_slots=decision.confirmed_slots,
        slot_bindings=decision.slot_bindings,
        missing_slots=decision.missing_slots,
        clarification_question=decision.clarification_question,
        clarification_count=clarification_count,
    )


def create_pending_transition(
    previous: SessionBusinessState,
    decision: DialogueDecision,
    *,
    original_request: str,
    request_turn_id: str,
    updated_at: str,
    tool_schema: Mapping[str, Any] | None = None,
) -> StateTransition:
    """Create the sole resumable task for a supported, incomplete request."""

    decision = _validate_decision_for_state(previous, decision, tool_schema=tool_schema)
    if (
        decision.intent is not DialogueIntent.NEW_BUSINESS
        or decision.execution_mode is not ExecutionMode.SINGLE
        or not decision.missing_slots
        or not decision.clarification_question
    ):
        raise DialogueContractError("invalid_pending_business")
    pending = _pending_from_decision(
        previous,
        decision,
        original_request=original_request,
        request_turn_id=request_turn_id,
        clarification_count=1,
    )
    test_id = decision.confirmed_slots.get("test_id") if decision.business_type is BusinessType.MBTI else None
    focus = BusinessFocus(decision.business_type, BusinessPhase.AWAITING_INPUT, test_id=test_id)
    return propose_transition(
        previous,
        decision,
        focus=focus,
        pending=pending,
        reason_code="pending_replaced" if previous.pending is not None else "pending_created",
        updated_at=updated_at,
        next_actions=("ask_clarification",),
        tool_schema=tool_schema,
    )


def start_business_transition(
    previous: SessionBusinessState,
    decision: DialogueDecision,
    *,
    original_request: str,
    request_turn_id: str,
    updated_at: str,
    tool_schema: Mapping[str, Any] | None = None,
) -> StateTransition:
    """Start or replace one explicit single business without retaining an old pending task."""

    decision = _validate_decision_for_state(previous, decision, tool_schema=tool_schema)
    if decision.intent is not DialogueIntent.NEW_BUSINESS or decision.execution_mode is not ExecutionMode.SINGLE:
        raise DialogueContractError("invalid_dialogue_decision")
    if decision.missing_slots:
        return create_pending_transition(
            previous,
            decision,
            original_request=original_request,
            request_turn_id=request_turn_id,
            updated_at=updated_at,
            tool_schema=tool_schema,
        )
    business_type = decision.business_type
    if business_type is None:
        raise DialogueContractError("invalid_dialogue_decision")
    if business_type is BusinessType.KNOWLEDGE:
        focus = BusinessFocus(business_type, BusinessPhase.ACTIVE)
        pending = None
        actions = ("answer_knowledge",)
    elif business_type in {BusinessType.BILIBILI_SEARCH, BusinessType.MBTI}:
        pending = _pending_from_decision(
            previous,
            decision,
            original_request=original_request,
            request_turn_id=request_turn_id,
            clarification_count=0,
        )
        test_id = decision.confirmed_slots.get("test_id") if business_type is BusinessType.MBTI else None
        focus = BusinessFocus(business_type, BusinessPhase.READY, test_id=test_id)
        actions = ("prepare_execution",)
    else:
        # Generic MCP requests have no cross-turn PendingBusiness contract in phase one.
        raise DialogueContractError("business_unsupported")
    return propose_transition(
        previous,
        decision,
        focus=focus,
        pending=pending,
        reason_code="pending_replaced" if previous.pending is not None else "business_started",
        updated_at=updated_at,
        next_actions=actions,
        tool_schema=tool_schema,
    )


def merge_pending_transition(
    previous: SessionBusinessState,
    decision: DialogueDecision,
    *,
    updated_at: str,
    tool_schema: Mapping[str, Any] | None = None,
) -> StateTransition:
    """Merge validated supplemental slots into the current awaiting-input task."""

    decision = _validate_decision_for_state(previous, decision, tool_schema=tool_schema)
    pending = previous.pending
    if (
        _phase(previous) is not BusinessPhase.AWAITING_INPUT
        or pending is None
        or decision.intent is not DialogueIntent.CONTINUE_BUSINESS
        or decision.business_type is not pending.business_type
    ):
        raise DialogueContractError("invalid_pending_continuation")

    combined_slots = dict(pending.confirmed_slots)
    for name, value in decision.confirmed_slots.items():
        if name in combined_slots and combined_slots[name] != value:
            raise DialogueContractError("conflicting_slot_value")
        combined_slots[name] = value
    bindings_by_name: dict[str, SlotBinding] = {item.slot_name: item for item in pending.slot_bindings}
    for binding in decision.slot_bindings:
        current = bindings_by_name.get(binding.slot_name)
        if current is not None and current.value != binding.value:
            raise DialogueContractError("conflicting_slot_value")
        bindings_by_name.setdefault(binding.slot_name, binding)

    required = set(required_business_slots(pending.business_type, tool_schema=tool_schema))
    unresolved = (set(pending.missing_slots) | set(decision.missing_slots) | required) - set(combined_slots)
    missing_slots = tuple(name for name in (*pending.missing_slots, *decision.missing_slots, *sorted(required)) if name in unresolved)
    missing_slots = tuple(dict.fromkeys(missing_slots))
    validate_business_slots(pending.business_type, combined_slots, missing_slots, tool_schema=tool_schema)
    next_pending = replace(
        pending,
        confirmed_slots=combined_slots,
        slot_bindings=tuple(bindings_by_name.values()),
        missing_slots=missing_slots,
        clarification_question=decision.clarification_question if missing_slots else "",
        clarification_count=1 if missing_slots else 0,
    )
    next_phase = BusinessPhase.AWAITING_INPUT if missing_slots else BusinessPhase.READY
    focus = replace(previous.focus, phase=next_phase) if previous.focus is not None else None
    return propose_transition(
        previous,
        decision,
        focus=focus,
        pending=next_pending,
        reason_code="slot_filled",
        updated_at=updated_at,
        next_actions=("ask_clarification",) if missing_slots else ("prepare_execution",),
        tool_schema=tool_schema,
    )


def record_unresolved_clarification(
    previous: SessionBusinessState,
    *,
    expected_revision: int,
    clarification_question: str,
    updated_at: str,
) -> StateTransition:
    """Count one genuine unresolved reply without imposing a retry ceiling."""

    previous = validate_business_state(previous)
    if _phase(previous) is not BusinessPhase.AWAITING_INPUT or previous.pending is None:
        raise DialogueContractError("invalid_pending_continuation")
    if not clarification_question:
        raise DialogueContractError("invalid_pending_business")
    pending = replace(
        previous.pending,
        clarification_question=clarification_question,
        clarification_count=previous.pending.clarification_count + 1,
    )
    return _build_transition(
        previous,
        expected_revision=expected_revision,
        focus=previous.focus,
        pending=pending,
        reason_code="clarification_unresolved",
        updated_at=updated_at,
        next_actions=("ask_clarification_with_example",),
    )


def preserve_business_state(
    previous: SessionBusinessState,
    decision: DialogueDecision,
    *,
    tool_schema: Mapping[str, Any] | None = None,
) -> SessionBusinessState:
    """Validate a passive decision and return the exact unchanged committed state."""

    decision = _validate_decision_for_state(previous, decision, tool_schema=tool_schema)
    if decision.intent not in {DialogueIntent.CHITCHAT, DialogueIntent.UNCLEAR, DialogueIntent.UNSUPPORTED}:
        raise DialogueContractError("invalid_dialogue_decision")
    return previous


def apply_mbti_outcome(
    previous: SessionBusinessState,
    *,
    expected_revision: int,
    test_id: str,
    phase: BusinessPhase,
    updated_at: str,
) -> StateTransition:
    """Project an already-validated assessment outcome without duplicating assessment logic."""

    previous = validate_business_state(previous)
    if phase not in {BusinessPhase.ACTIVE, BusinessPhase.PAUSED, BusinessPhase.COMPLETED, BusinessPhase.CANCELLED}:
        raise DialogueContractError("invalid_business_phase")
    old_focus = previous.focus
    if old_focus is not None and old_focus.business_type is BusinessType.MBTI:
        if old_focus.test_id is not None and old_focus.test_id != test_id:
            raise DialogueContractError("mbti_test_mismatch")
        if old_focus.phase is BusinessPhase.COMPLETED and phase is not BusinessPhase.COMPLETED:
            raise DialogueContractError("invalid_state_transition")
    old_phase = _phase(previous)
    if (
        old_phase is phase
        and old_phase is not BusinessPhase.ACTIVE
        and old_focus is not None
        and old_focus.test_id == test_id
    ):
        raise DialogueContractError("state_unchanged")
    if phase is BusinessPhase.PAUSED:
        reason = "business_paused"
    elif old_phase is BusinessPhase.PAUSED and phase is BusinessPhase.ACTIVE:
        reason = "business_resumed"
    elif phase is BusinessPhase.COMPLETED:
        reason = "execution_completed"
    elif phase is BusinessPhase.CANCELLED:
        reason = "business_cleared"
    elif old_phase is BusinessPhase.ACTIVE:
        reason = "business_continued"
    else:
        reason = "business_started"
    return _build_transition(
        previous,
        expected_revision=expected_revision,
        focus=BusinessFocus(BusinessType.MBTI, phase, test_id=test_id),
        pending=None,
        reason_code=reason,
        updated_at=updated_at,
    )


def cancel_current_business(
    previous: SessionBusinessState,
    *,
    expected_revision: int,
    updated_at: str,
    mbti_test_id: str | None = None,
) -> StateTransition:
    """Cancel the unambiguous current task; never attempts to interrupt dispatch."""

    previous = validate_business_state(previous)
    focus = previous.focus
    if focus is None or focus.phase in {BusinessPhase.COMPLETED, BusinessPhase.CANCELLED, BusinessPhase.EXECUTION_UNKNOWN}:
        raise DialogueContractError("no_active_business")
    if focus.phase is BusinessPhase.DISPATCHING:
        raise DialogueContractError("execution_unknown")
    if focus.business_type is BusinessType.MBTI and focus.phase not in {
        BusinessPhase.AWAITING_INPUT,
        BusinessPhase.READY,
    }:
        if focus.test_id is None or mbti_test_id != focus.test_id:
            raise DialogueContractError("mbti_test_mismatch")
    reason = "pending_cancelled" if focus.phase in {BusinessPhase.AWAITING_INPUT, BusinessPhase.READY} else "business_cleared"
    return _build_transition(
        previous,
        expected_revision=expected_revision,
        focus=replace(focus, phase=BusinessPhase.CANCELLED),
        pending=None,
        reason_code=reason,
        updated_at=updated_at,
        next_actions=("report_cancelled",),
    )


@dataclass(frozen=True, slots=True)
class ExecutionProposal:
    expected_revision: int
    business_type: BusinessType
    arguments: Mapping[str, Any]
    slot_bindings: tuple[SlotBinding, ...]
    allowed_tool_names: tuple[str, ...]


def prepare_execution(
    previous: SessionBusinessState,
    decision: DialogueDecision,
    *,
    tool_schema: Mapping[str, Any] | None = None,
) -> ExecutionProposal:
    """Revalidate a ready task and return data for the existing executor, not a tool call."""

    decision = _validate_decision_for_state(previous, decision, tool_schema=tool_schema)
    pending = previous.pending
    if _phase(previous) is not BusinessPhase.READY or pending is None or pending.missing_slots:
        raise DialogueContractError("business_not_ready")
    if decision.business_type is not pending.business_type or dict(decision.confirmed_slots) != dict(pending.confirmed_slots):
        raise DialogueContractError("pending_decision_mismatch")
    if tuple(decision.slot_bindings) != tuple(pending.slot_bindings):
        raise DialogueContractError("pending_decision_mismatch")
    arguments = validate_business_slots(
        pending.business_type,
        pending.confirmed_slots,
        (),
        tool_schema=tool_schema,
    )
    if pending.business_type is BusinessType.BILIBILI_SEARCH and not decision.allowed_tool_names:
        raise DialogueContractError("tool_unavailable")
    return ExecutionProposal(
        expected_revision=previous.revision,
        business_type=pending.business_type,
        arguments=MappingProxyType(arguments),
        slot_bindings=pending.slot_bindings,
        allowed_tool_names=decision.allowed_tool_names,
    )


def mark_dispatching(
    previous: SessionBusinessState,
    *,
    expected_revision: int,
    updated_at: str,
    tool_schema: Mapping[str, Any] | None = None,
) -> StateTransition:
    """Consume a ready pending task in the short transaction before external dispatch."""

    previous = validate_business_state(previous)
    pending = previous.pending
    if _phase(previous) is not BusinessPhase.READY or pending is None or pending.missing_slots:
        raise DialogueContractError("business_not_ready")
    if pending.business_type not in {BusinessType.BILIBILI_SEARCH, BusinessType.MCP_TOOL}:
        raise DialogueContractError("business_not_dispatchable")
    validate_business_slots(pending.business_type, pending.confirmed_slots, (), tool_schema=tool_schema)
    if set(pending.confirmed_slots) != {binding.slot_name for binding in pending.slot_bindings}:
        raise DialogueContractError("slot_binding_mismatch")
    focus = replace(previous.focus, phase=BusinessPhase.DISPATCHING) if previous.focus is not None else None
    return _build_transition(
        previous,
        expected_revision=expected_revision,
        focus=focus,
        pending=None,
        reason_code="dispatch_started",
        updated_at=updated_at,
        next_actions=("dispatch_external_operation",),
        commit_timing=CommitTiming.BEFORE_DISPATCH,
    )


class ExecutionOutcome(str, Enum):
    NOT_DISPATCHED = "not_dispatched"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    UNKNOWN = "unknown"


def apply_execution_outcome(
    previous: SessionBusinessState,
    *,
    expected_revision: int,
    outcome: ExecutionOutcome,
    updated_at: str,
) -> StateTransition | None:
    """Close a dispatched operation from an executor-owned fact without retrying it."""

    previous = validate_business_state(previous)
    if not isinstance(outcome, ExecutionOutcome):
        raise DialogueContractError("invalid_execution_outcome")
    if outcome is ExecutionOutcome.NOT_DISPATCHED:
        if type(expected_revision) is not int or expected_revision != previous.revision:
            raise DialogueContractError("state_conflict")
        if _phase(previous) is not BusinessPhase.READY:
            raise DialogueContractError("invalid_execution_outcome")
        return None
    if _phase(previous) is not BusinessPhase.DISPATCHING or previous.focus is None:
        raise DialogueContractError("invalid_execution_outcome")
    if outcome is ExecutionOutcome.SUCCEEDED:
        phase, reason, action = BusinessPhase.COMPLETED, "execution_completed", "report_execution_completed"
    elif outcome is ExecutionOutcome.FAILED:
        phase, reason, action = BusinessPhase.COMPLETED, "execution_failed", "report_execution_failed"
    elif outcome is ExecutionOutcome.TIMED_OUT:
        phase, reason, action = BusinessPhase.EXECUTION_UNKNOWN, "execution_timed_out", "report_execution_timed_out"
    else:
        phase, reason, action = BusinessPhase.EXECUTION_UNKNOWN, "execution_unknown", "report_execution_unknown"
    return _build_transition(
        previous,
        expected_revision=expected_revision,
        focus=replace(previous.focus, phase=phase),
        pending=None,
        reason_code=reason,
        updated_at=updated_at,
        next_actions=(action,),
        commit_timing=CommitTiming.AFTER_EXECUTION,
    )


def restore_business_state(
    record: SessionBusinessState | Mapping[str, Any] | None,
    *,
    session_id: str,
    updated_at: str,
) -> SessionBusinessState:
    """Load only the dedicated state record; absence means a validated empty legacy state."""

    if record is None:
        return empty_business_state(session_id, updated_at=updated_at)
    state = state_from_payload(record) if isinstance(record, Mapping) else validate_business_state(record)
    if state.session_id != session_id:
        raise DialogueContractError("cross_session_state")
    return state


def restoration_next_actions(state: SessionBusinessState) -> tuple[str, ...]:
    """Describe safe resume behavior without turning recovery into execution."""

    state = validate_business_state(state)
    phase = _phase(state)
    if phase is BusinessPhase.AWAITING_INPUT:
        return ("show_clarification",)
    if phase in {BusinessPhase.DISPATCHING, BusinessPhase.EXECUTION_UNKNOWN}:
        return ("report_execution_unknown",)
    return ()


def cleared_business_state(previous: SessionBusinessState, *, updated_at: str) -> SessionBusinessState:
    """Return the empty in-memory value to publish after persistent clear/delete succeeds."""

    previous = validate_business_state(previous)
    return empty_business_state(previous.session_id, updated_at=updated_at)


__all__ = [
    "ExecutionOutcome",
    "ExecutionProposal",
    "apply_execution_outcome",
    "apply_mbti_outcome",
    "cancel_current_business",
    "cleared_business_state",
    "create_pending_transition",
    "mark_dispatching",
    "merge_pending_transition",
    "prepare_execution",
    "preserve_business_state",
    "propose_transition",
    "record_unresolved_clarification",
    "restoration_next_actions",
    "restore_business_state",
    "start_business_transition",
    "transition_requires_execution",
]
