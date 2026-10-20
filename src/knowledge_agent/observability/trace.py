"""单轮问答执行记录模块。

模块功能：记录一轮问答实际经过的阶段、Skill 使用情况、工具结果摘要和耗时，
供回答完成后的“执行详情”面板展示，也为后续效果评估提供执行信息。
输入：Agent 在真实执行边界提供的阶段事件、计数、索引版本和结束状态。
输出：可序列化的执行摘要；由 AnswerResult 携带，经 API 返回并随会话保存。

职责分工：
1. 本模块定义执行记录的数据结构、收集器和序列化规则，不发起模型或工具调用。
2. agent/runner.py、tools/registry.py 在实际执行位置记录事件，保留现有业务流程与预算。
3. schemas.py 将摘要挂到 AnswerResult；session/store.py 保存并恢复该摘要。
4. api.py 转换公开响应；前端显示本轮与历史问答的执行详情。

回答结束后返回完整记录，复用当前 SQLite 的 turns.result_json。
可选订阅器实时发布公开阶段进度；不记录模型内部推理、Token 或检索原文。
已实现收集器、问答记录及存储接入，API 与页面展示的接入位置见下方完成说明。

开发项 01—16 已实现；TODO 17—20 保留为待统一执行的验证，不代表已经验收通过。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from time import monotonic
from typing import Any, Callable, Literal
from uuid import uuid4

from knowledge_agent.schemas import AnswerValidationStatus, DeclaredAnswerStatus


_LOGGER = logging.getLogger(__name__)
TRACE_RECORDING_FAILED = "trace_recording_failed"
_UNFINISHED_EVENT = "unfinished_event"
_TURN_FAILED = "turn_failed"
_TURN_INTERRUPTED = "turn_interrupted"


# 第一阶段：数据协议与收集器。

TraceStage = Literal[
    "memory",
    "profile",
    "planning",
    "plan_step",
    "classify",
    "intent",
    "route",
    "initial_retrieval",
    "skill_select",
    "skill_retrieval",
    "skill_prepare",
    "model",
    "tool",
    "finalize",
]
TraceInitiator = Literal["program", "model"]
TraceStatus = Literal["success", "empty", "error", "skipped"]
TraceOutcome = Literal["completed", "failed", "interrupted"]
TraceDecisionSource = Literal["rules", "model", "fallback", "manual", "disabled"]
TraceSkillMode = Literal["off", "auto", "manual"]
TraceIntentKind = Literal[
    "chat", "knowledge", "greeting", "scope", "needs_clarification", "out_of_scope", "mcp_tool",
]
TraceRouteKind = Literal["knowledge", "mcp", "direct", "chat"]

_PROGRESS_OBSERVER: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "trace_progress_observer", default=None,
)


@contextmanager
def observe_trace_progress(observer: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    """只订阅当前工作线程操作，并始终恢复原有上下文。"""
    token = _PROGRESS_OBSERVER.set(observer)
    try:
        yield
    finally:
        _PROGRESS_OBSERVER.reset(token)


def _publish_progress(payload: dict[str, Any]) -> None:
    observer = _PROGRESS_OBSERVER.get()
    if observer is None:
        return
    try:
        observer(payload)
    except Exception:
        # 断开的 UI 或失败的可选订阅者不得触发重试，
        # 也不得丢弃本来可用的回答。不要记录任何载荷内容。
        _LOGGER.warning("trace_progress_unavailable")


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """轮次中的一个可观测阶段，按开始时间排序。"""

    sequence: int
    stage: TraceStage
    initiator: TraceInitiator
    status: TraceStatus
    started_offset_ms: int | None
    duration_ms: int | None
    tool_name: str | None = None
    tool_call_id: str | None = None
    result_count: int | None = None
    history_turns: int | None = None
    dropped_turns: int | None = None
    summary_chars: int | None = None
    memory_chars: int | None = None
    index_version: str | None = None
    error_code: str | None = None
    decision_source: TraceDecisionSource | None = None
    intent_kind: TraceIntentKind | None = None
    route_kind: TraceRouteKind | None = None
    reason_code: str | None = None
    skill_mode: TraceSkillMode | None = None
    selected_skill_id: str | None = None
    strategy_id: str | None = None
    step_id: str | None = None
    declared_status: DeclaredAnswerStatus | None = None
    validation_status: AnswerValidationStatus | None = None
    validation_reason: str | None = None
    supplied_source_count: int | None = None
    dialogue_action: str | None = None
    business_type: str | None = None
    execution_mode: str | None = None

@dataclass(frozen=True, slots=True)
class TurnTrace:
    """单个轮次可观测执行过程的不可变摘要。"""

    schema_version: int
    trace_id: str
    started_at: datetime
    duration_ms: int | None
    events: tuple[TraceEvent, ...]
    outcome: TraceOutcome
    requested_skill_id: str | None = None
    prepared_skill_id: str | None = None
    applied_skill_id: str | None = None
    error_code: str | None = None
    events_truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))

@dataclass(slots=True)
class _PendingEvent:
    sequence: int
    stage: TraceStage
    initiator: TraceInitiator
    started: float | None
    tool_name: str | None
    tool_call_id: str | None
    skill_mode: TraceSkillMode | None = None
    selected_skill_id: str | None = None
    strategy_id: str | None = None
    step_id: str | None = None


class TraceRecorder:
    """收集恰好一个轮次的有界、可观测执行历史。"""

    DEFAULT_MAX_EVENTS = 128
    MAX_DISPLAY_STRING_LENGTH = 256

    def __init__(
        self,
        *,
        requested_skill_id: str | None = None,
        max_events: int = DEFAULT_MAX_EVENTS,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float | None] | None = None,
    ) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 0:
            raise ValueError("max_events must be a non-negative integer")
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._monotonic_clock = monotonic_clock or monotonic
        self._started_at = self._utc_now()
        self._started = self._read_monotonic()
        self._trace_id = uuid4().hex
        self._max_events = max_events
        self._next_handle = 0
        self._pending: dict[int, _PendingEvent] = {}
        self._finished_handles: set[int] = set()
        self._events: list[TraceEvent] = []
        self._events_truncated = False
        self._requested_skill_id = self._bounded(requested_skill_id)
        self._prepared_skill_id: str | None = None
        self._applied_skill_id: str | None = None
        self._snapshot: TurnTrace | None = None

    @classmethod
    def _bounded(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(value)[: cls.MAX_DISPLAY_STRING_LENGTH]

    def _utc_now(self) -> datetime:
        value = self._wall_clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("wall_clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    def _read_monotonic(self) -> float | None:
        value = self._monotonic_clock()
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if isfinite(value) else None

    @staticmethod
    def _milliseconds(start: float | None, end: float | None) -> int | None:
        if start is None or end is None:
            return None
        return max(0, int((end - start) * 1000))

    def _ensure_active(self) -> None:
        if self._snapshot is not None:
            raise RuntimeError("trace has already been finalized")

    def start_event(
        self,
        stage: TraceStage,
        initiator: TraceInitiator,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        skill_mode: TraceSkillMode | None = None,
        selected_skill_id: str | None = None,
        strategy_id: str | None = None,
        step_id: str | None = None,
    ) -> int:
        """开始一个事件并返回记录器内部的不透明句柄。"""

        self._ensure_active()
        handle = self._next_handle
        self._next_handle += 1
        now = self._read_monotonic()
        if len(self._events) + len(self._pending) >= self._max_events:
            self._events_truncated = True
            return handle
        self._pending[handle] = _PendingEvent(
            sequence=handle,
            stage=stage,
            initiator=initiator,
            started=now,
            tool_name=self._bounded(tool_name),
            tool_call_id=self._bounded(tool_call_id),
            skill_mode=skill_mode,
            selected_skill_id=self._bounded(selected_skill_id),
            strategy_id=self._bounded(strategy_id),
            step_id=self._bounded(step_id),
        )
        # 刻意枚举安全的公开字段，而不是转发调用方的
        # kwargs；提示词、原始工具参数和证据绝不进入此 API。
        _publish_progress({
            "sequence": handle, "stage": stage, "initiator": initiator,
            "status": "running",
            "started_offset_ms": self._milliseconds(self._started, now),
            "tool_name": self._bounded(tool_name),
            "skill_mode": skill_mode,
            "selected_skill_id": self._bounded(selected_skill_id),
            "strategy_id": self._bounded(strategy_id),
            "step_id": self._bounded(step_id),
        })
        return handle

    def finish_event(
        self,
        handle: int,
        status: TraceStatus,
        *,
        result_count: int | None = None,
        history_turns: int | None = None,
        dropped_turns: int | None = None,
        summary_chars: int | None = None,
        memory_chars: int | None = None,
        index_version: str | None = None,
        error_code: str | None = None,
        decision_source: TraceDecisionSource | None = None,
        intent_kind: TraceIntentKind | None = None,
        route_kind: TraceRouteKind | None = None,
        reason_code: str | None = None,
        skill_mode: TraceSkillMode | None = None,
        selected_skill_id: str | None = None,
        strategy_id: str | None = None,
        step_id: str | None = None,
        declared_status: DeclaredAnswerStatus | None = None,
        validation_status: AnswerValidationStatus | None = None,
        validation_reason: str | None = None,
        supplied_source_count: int | None = None,
        dialogue_action: str | None = None,
        business_type: str | None = None,
        execution_mode: str | None = None,
    ) -> TraceEvent | None:
        """结束一个事件；因上限被省略的事件返回 ``None``。"""

        self._ensure_active()
        if handle in self._finished_handles:
            raise RuntimeError("event has already been finished")
        if not isinstance(handle, int) or isinstance(handle, bool) or not 0 <= handle < self._next_handle:
            raise ValueError("unknown event handle")
        self._finished_handles.add(handle)
        pending = self._pending.pop(handle, None)
        if pending is None:
            return None
        now = self._read_monotonic()
        event = TraceEvent(
            sequence=pending.sequence,
            stage=pending.stage,
            initiator=pending.initiator,
            status=status,
            started_offset_ms=self._milliseconds(self._started, pending.started),
            duration_ms=self._milliseconds(pending.started, now),
            tool_name=pending.tool_name,
            tool_call_id=pending.tool_call_id,
            result_count=result_count,
            history_turns=history_turns,
            dropped_turns=dropped_turns,
            summary_chars=summary_chars,
            memory_chars=memory_chars,
            index_version=self._bounded(index_version),
            error_code=self._bounded(error_code),
            decision_source=decision_source,
            intent_kind=intent_kind,
            route_kind=route_kind,
            reason_code=self._bounded(reason_code),
            skill_mode=skill_mode if skill_mode is not None else pending.skill_mode,
            selected_skill_id=self._bounded(selected_skill_id) if selected_skill_id is not None else pending.selected_skill_id,
            strategy_id=self._bounded(strategy_id) if strategy_id is not None else pending.strategy_id,
            step_id=self._bounded(step_id) if step_id is not None else pending.step_id,
            declared_status=declared_status,
            validation_status=validation_status,
            validation_reason=self._bounded(validation_reason),
            supplied_source_count=supplied_source_count,
            dialogue_action=self._bounded(dialogue_action),
            business_type=self._bounded(business_type),
            execution_mode=self._bounded(execution_mode),
        )
        self._events.append(event)
        self._publish_finished(event)
        return event

    @staticmethod
    def _publish_finished(event: TraceEvent) -> None:
        try:
            payload = _event_to_dict(event, "progress")
        except (TypeError, ValueError):
            _LOGGER.warning("trace_progress_unavailable")
            return
        # 工具调用 ID 和索引指纹对最终 trace 有用，
        # 但实时阶段指示器不需要它们。
        payload.pop("tool_call_id", None)
        payload.pop("index_version", None)
        _publish_progress(payload)

    def mark_skill_prepared(self, skill_id: str | None) -> None:
        self._ensure_active()
        self._prepared_skill_id = self._bounded(skill_id)

    def mark_skill_applied(self, skill_id: str | None) -> None:
        self._ensure_active()
        self._applied_skill_id = self._bounded(skill_id)

    def finish_turn(
        self,
        outcome: TraceOutcome,
        *,
        error_code: str | None = None,
    ) -> TurnTrace:
        """结束轮次并返回稳定的不可变快照。"""

        if self._snapshot is not None:
            return self._snapshot
        ended = self._read_monotonic()
        if self._pending:
            pending_status: TraceStatus = "skipped" if outcome == "completed" else "error"
            pending_error_code = error_code or {
                "completed": _UNFINISHED_EVENT,
                "failed": _TURN_FAILED,
                "interrupted": _TURN_INTERRUPTED,
            }.get(outcome, _UNFINISHED_EVENT)
            for handle, pending in self._pending.items():
                event = TraceEvent(
                    sequence=pending.sequence,
                    stage=pending.stage,
                    initiator=pending.initiator,
                    status=pending_status,
                    started_offset_ms=self._milliseconds(self._started, pending.started),
                    duration_ms=self._milliseconds(pending.started, ended),
                    tool_name=pending.tool_name,
                    tool_call_id=pending.tool_call_id,
                    error_code=self._bounded(pending_error_code),
                    skill_mode=pending.skill_mode,
                    selected_skill_id=pending.selected_skill_id,
                    strategy_id=pending.strategy_id,
                    step_id=pending.step_id,
                )
                self._events.append(event)
                self._publish_finished(event)
                self._finished_handles.add(handle)
            self._pending.clear()
        self._snapshot = TurnTrace(
            schema_version=1,
            trace_id=self._trace_id,
            started_at=self._started_at,
            duration_ms=self._milliseconds(self._started, ended),
            events=tuple(sorted(self._events, key=lambda event: event.sequence)),
            outcome=outcome,
            requested_skill_id=self._requested_skill_id,
            prepared_skill_id=self._prepared_skill_id,
            applied_skill_id=self._applied_skill_id,
            error_code=self._bounded(error_code),
            events_truncated=self._events_truncated,
        )
        return self._snapshot

    def snapshot(self) -> TurnTrace:
        """返回已定稿的快照，不重建也不修改它。"""

        if self._snapshot is None:
            raise RuntimeError("trace has not been finalized")
        return self._snapshot

_TRACE_FIELDS = {
    "schema_version",
    "trace_id",
    "started_at",
    "duration_ms",
    "events",
    "outcome",
    "requested_skill_id",
    "prepared_skill_id",
    "applied_skill_id",
    "error_code",
    "events_truncated",
}
_EVENT_FIELDS = {
    "sequence",
    "stage",
    "initiator",
    "status",
    "started_offset_ms",
    "duration_ms",
    "tool_name",
    "tool_call_id",
    "result_count",
    "index_version",
    "error_code",
}
_OPTIONAL_EVENT_FIELDS = {"history_turns", "dropped_turns", "summary_chars", "memory_chars", "decision_source", "intent_kind", "route_kind", "reason_code", "skill_mode", "selected_skill_id", "strategy_id", "step_id", "declared_status", "validation_status", "validation_reason", "supplied_source_count", "dialogue_action", "business_type", "execution_mode"}
_TRACE_STAGES = frozenset(("memory", "profile", "planning", "plan_step", "classify", "intent", "route", "initial_retrieval", "skill_select", "skill_retrieval", "skill_prepare", "model", "tool", "finalize"))
_TRACE_INITIATORS = frozenset(("program", "model"))
_TRACE_STATUSES = frozenset(("success", "empty", "error", "skipped"))
_TRACE_OUTCOMES = frozenset(("completed", "failed", "interrupted"))
_TRACE_DECISION_SOURCES = frozenset(("rules", "model", "fallback", "manual", "disabled"))
_TRACE_SKILL_MODES = frozenset(("off", "auto", "manual"))
_TRACE_INTENT_KINDS = frozenset((
    "chat", "knowledge", "greeting", "scope", "needs_clarification", "out_of_scope", "mcp_tool",
))
_TRACE_ROUTE_KINDS = frozenset(("knowledge", "mcp", "direct", "chat"))
_DIALOGUE_ACTIONS = frozenset(("new_business", "continue_business", "cancel_business", "pause_business", "resume_business", "chitchat", "unclear", "unsupported"))
_BUSINESS_TYPES = frozenset(("knowledge", "bilibili_search", "mbti", "mcp_tool"))
_EXECUTION_MODES = frozenset(("single", "plan"))
_DECLARED_ANSWER_STATUSES = frozenset(("supported", "partial", "insufficient_evidence", "needs_clarification", "out_of_scope"))
_ANSWER_VALIDATION_STATUSES = frozenset(("passed", "warning", "rejected"))


def _require_exact_fields(
    value: dict[str, Any], fields: set[str], label: str, *, optional: set[str] | None = None,
) -> None:
    missing = fields - value.keys()
    extra = value.keys() - fields - (optional or set())
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)!r}")
        if extra:
            details.append(f"unexpected {sorted(extra)!r}")
        raise ValueError(f"Invalid {label}: {', '.join(details)}")


def _public_string(value: Any, label: str, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value):
        raise ValueError(f"{label} must be {'a non-empty string' if required else 'a string or null'}")
    if len(value) > TraceRecorder.MAX_DISPLAY_STRING_LENGTH:
        raise ValueError(f"{label} exceeds the public string length limit")
    return value


def _non_negative_int(value: Any, label: str, *, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        suffix = " or null" if nullable else ""
        raise ValueError(f"{label} must be a non-negative integer{suffix}")
    return value


def _enum(value: Any, allowed: frozenset[str], label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _event_to_dict(event: TraceEvent, label: str) -> dict[str, Any]:
    return {
        "history_turns": _non_negative_int(event.history_turns, f"{label}.history_turns", nullable=True),
        "dropped_turns": _non_negative_int(event.dropped_turns, f"{label}.dropped_turns", nullable=True),
        "summary_chars": _non_negative_int(event.summary_chars, f"{label}.summary_chars", nullable=True),
        "memory_chars": _non_negative_int(event.memory_chars, f"{label}.memory_chars", nullable=True),
        "sequence": _non_negative_int(event.sequence, f"{label}.sequence"),
        "stage": _enum(event.stage, _TRACE_STAGES, f"{label}.stage"),
        "initiator": _enum(event.initiator, _TRACE_INITIATORS, f"{label}.initiator"),
        "status": _enum(event.status, _TRACE_STATUSES, f"{label}.status"),
        "started_offset_ms": _non_negative_int(event.started_offset_ms, f"{label}.started_offset_ms", nullable=True),
        "duration_ms": _non_negative_int(event.duration_ms, f"{label}.duration_ms", nullable=True),
        "tool_name": _public_string(event.tool_name, f"{label}.tool_name"),
        "tool_call_id": _public_string(event.tool_call_id, f"{label}.tool_call_id"),
        "result_count": _non_negative_int(event.result_count, f"{label}.result_count", nullable=True),
        "index_version": _public_string(event.index_version, f"{label}.index_version"),
        "error_code": _public_string(event.error_code, f"{label}.error_code"),
        "decision_source": (
            _enum(event.decision_source, _TRACE_DECISION_SOURCES, f"{label}.decision_source")
            if event.decision_source is not None else None
        ),
        "intent_kind": (
            _enum(event.intent_kind, _TRACE_INTENT_KINDS, f"{label}.intent_kind")
            if event.intent_kind is not None else None
        ),
        "route_kind": (
            _enum(event.route_kind, _TRACE_ROUTE_KINDS, f"{label}.route_kind")
            if event.route_kind is not None else None
        ),
        "reason_code": _public_string(event.reason_code, f"{label}.reason_code"),
        "skill_mode": (
            _enum(event.skill_mode, _TRACE_SKILL_MODES, f"{label}.skill_mode")
            if event.skill_mode is not None else None
        ),
        "selected_skill_id": _public_string(event.selected_skill_id, f"{label}.selected_skill_id"),
        "strategy_id": _public_string(event.strategy_id, f"{label}.strategy_id"),
        "step_id": _public_string(event.step_id, f"{label}.step_id"),
        "declared_status": (
            _enum(event.declared_status, _DECLARED_ANSWER_STATUSES, f"{label}.declared_status")
            if event.declared_status is not None else None
        ),
        "validation_status": (
            _enum(event.validation_status, _ANSWER_VALIDATION_STATUSES, f"{label}.validation_status")
            if event.validation_status is not None else None
        ),
        "validation_reason": _public_string(event.validation_reason, f"{label}.validation_reason"),
        "supplied_source_count": _non_negative_int(event.supplied_source_count, f"{label}.supplied_source_count", nullable=True),
        "dialogue_action": (
            _enum(event.dialogue_action, _DIALOGUE_ACTIONS, f"{label}.dialogue_action")
            if event.dialogue_action is not None else None
        ),
        "business_type": (
            _enum(event.business_type, _BUSINESS_TYPES, f"{label}.business_type")
            if event.business_type is not None else None
        ),
        "execution_mode": (
            _enum(event.execution_mode, _EXECUTION_MODES, f"{label}.execution_mode")
            if event.execution_mode is not None else None
        ),
    }


def trace_to_dict(trace: TurnTrace) -> dict[str, Any]:
    """序列化轮次 trace 中已列入白名单的公开字段。"""

    if not isinstance(trace, TurnTrace):
        raise TypeError("trace must be a TurnTrace")
    if isinstance(trace.schema_version, bool) or trace.schema_version != 1:
        raise ValueError(f"Unsupported trace schema_version: {trace.schema_version!r}")
    if not isinstance(trace.started_at, datetime) or trace.started_at.tzinfo is None or trace.started_at.utcoffset() is None:
        raise ValueError("trace.started_at must be a timezone-aware datetime")
    events = [_event_to_dict(event, f"trace.events[{index}]") for index, event in enumerate(trace.events)]
    sequences = [event["sequence"] for event in events]
    if any(current <= previous for previous, current in zip(sequences, sequences[1:])):
        raise ValueError("trace.events must be in strictly increasing sequence order")
    if not isinstance(trace.events_truncated, bool):
        raise ValueError("trace.events_truncated must be a boolean")
    return {
        "schema_version": 1,
        "trace_id": _public_string(trace.trace_id, "trace.trace_id", required=True),
        "started_at": trace.started_at.astimezone(timezone.utc).isoformat(),
        "duration_ms": _non_negative_int(trace.duration_ms, "trace.duration_ms", nullable=True),
        "events": events,
        "outcome": _enum(trace.outcome, _TRACE_OUTCOMES, "trace.outcome"),
        "requested_skill_id": _public_string(trace.requested_skill_id, "trace.requested_skill_id"),
        "prepared_skill_id": _public_string(trace.prepared_skill_id, "trace.prepared_skill_id"),
        "applied_skill_id": _public_string(trace.applied_skill_id, "trace.applied_skill_id"),
        "error_code": _public_string(trace.error_code, "trace.error_code"),
        "events_truncated": trace.events_truncated,
    }


def trace_from_dict(value: Any) -> TurnTrace:
    """校验并反序列化公开的轮次 trace 映射。"""

    if not isinstance(value, dict):
        raise ValueError("trace must be an object")
    _require_exact_fields(value, _TRACE_FIELDS, "trace")
    if isinstance(value["schema_version"], bool) or value["schema_version"] != 1:
        raise ValueError(f"Unsupported trace schema_version: {value['schema_version']!r}")
    started_at_value = _public_string(value["started_at"], "trace.started_at", required=True)
    try:
        started_at = datetime.fromisoformat(started_at_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("trace.started_at must be an ISO 8601 datetime") from exc
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise ValueError("trace.started_at must include a UTC offset")
    event_values = value["events"]
    if not isinstance(event_values, list):
        raise ValueError("trace.events must be a list")
    events: list[TraceEvent] = []
    previous_sequence = -1
    for index, item in enumerate(event_values):
        label = f"trace.events[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_fields(item, _EVENT_FIELDS, label, optional=_OPTIONAL_EVENT_FIELDS)
        sequence = _non_negative_int(item["sequence"], f"{label}.sequence")
        if sequence <= previous_sequence:
            raise ValueError("trace.events must be in strictly increasing sequence order")
        previous_sequence = sequence
        events.append(TraceEvent(
            history_turns=_non_negative_int(item.get("history_turns"), f"{label}.history_turns", nullable=True),
            dropped_turns=_non_negative_int(item.get("dropped_turns"), f"{label}.dropped_turns", nullable=True),
            summary_chars=_non_negative_int(item.get("summary_chars"), f"{label}.summary_chars", nullable=True),
            memory_chars=_non_negative_int(item.get("memory_chars"), f"{label}.memory_chars", nullable=True),
            sequence=sequence,
            stage=_enum(item["stage"], _TRACE_STAGES, f"{label}.stage"),
            initiator=_enum(item["initiator"], _TRACE_INITIATORS, f"{label}.initiator"),
            status=_enum(item["status"], _TRACE_STATUSES, f"{label}.status"),
            started_offset_ms=_non_negative_int(item["started_offset_ms"], f"{label}.started_offset_ms", nullable=True),
            duration_ms=_non_negative_int(item["duration_ms"], f"{label}.duration_ms", nullable=True),
            tool_name=_public_string(item["tool_name"], f"{label}.tool_name"),
            tool_call_id=_public_string(item["tool_call_id"], f"{label}.tool_call_id"),
            result_count=_non_negative_int(item["result_count"], f"{label}.result_count", nullable=True),
            index_version=_public_string(item["index_version"], f"{label}.index_version"),
            error_code=_public_string(item["error_code"], f"{label}.error_code"),
            decision_source=(
                _enum(item["decision_source"], _TRACE_DECISION_SOURCES, f"{label}.decision_source")
                if item.get("decision_source") is not None else None
            ),
            intent_kind=(
                _enum(item["intent_kind"], _TRACE_INTENT_KINDS, f"{label}.intent_kind")
                if item.get("intent_kind") is not None else None
            ),
            route_kind=(
                _enum(item["route_kind"], _TRACE_ROUTE_KINDS, f"{label}.route_kind")
                if item.get("route_kind") is not None else None
            ),
            reason_code=_public_string(item.get("reason_code"), f"{label}.reason_code"),
            skill_mode=(
                _enum(item["skill_mode"], _TRACE_SKILL_MODES, f"{label}.skill_mode")
                if item.get("skill_mode") is not None else None
            ),
            selected_skill_id=_public_string(item.get("selected_skill_id"), f"{label}.selected_skill_id"),
            strategy_id=_public_string(item.get("strategy_id"), f"{label}.strategy_id"),
            step_id=_public_string(item.get("step_id"), f"{label}.step_id"),
            declared_status=(
                _enum(item["declared_status"], _DECLARED_ANSWER_STATUSES, f"{label}.declared_status")
                if item.get("declared_status") is not None else None
            ),
            validation_status=(
                _enum(item["validation_status"], _ANSWER_VALIDATION_STATUSES, f"{label}.validation_status")
                if item.get("validation_status") is not None else None
            ),
            validation_reason=_public_string(item.get("validation_reason"), f"{label}.validation_reason"),
            supplied_source_count=_non_negative_int(item.get("supplied_source_count"), f"{label}.supplied_source_count", nullable=True),
            dialogue_action=(
                _enum(item["dialogue_action"], _DIALOGUE_ACTIONS, f"{label}.dialogue_action")
                if item.get("dialogue_action") is not None else None
            ),
            business_type=(
                _enum(item["business_type"], _BUSINESS_TYPES, f"{label}.business_type")
                if item.get("business_type") is not None else None
            ),
            execution_mode=(
                _enum(item["execution_mode"], _EXECUTION_MODES, f"{label}.execution_mode")
                if item.get("execution_mode") is not None else None
            ),
        ))
    events_truncated = value["events_truncated"]
    if not isinstance(events_truncated, bool):
        raise ValueError("trace.events_truncated must be a boolean")
    return TurnTrace(
        schema_version=1,
        trace_id=_public_string(value["trace_id"], "trace.trace_id", required=True),
        started_at=started_at.astimezone(timezone.utc),
        duration_ms=_non_negative_int(value["duration_ms"], "trace.duration_ms", nullable=True),
        events=tuple(events),
        outcome=_enum(value["outcome"], _TRACE_OUTCOMES, "trace.outcome"),
        requested_skill_id=_public_string(value["requested_skill_id"], "trace.requested_skill_id"),
        prepared_skill_id=_public_string(value["prepared_skill_id"], "trace.prepared_skill_id"),
        applied_skill_id=_public_string(value["applied_skill_id"], "trace.applied_skill_id"),
        error_code=_public_string(value["error_code"], "trace.error_code"),
        events_truncated=events_truncated,
    )


def finish_trace_safely(
    recorder: TraceRecorder,
    outcome: TraceOutcome,
    *,
    error_code: str | None = None,
) -> TurnTrace | None:
    """结束记录，同时不让可预期的 trace 故障掩盖业务结果。

    该边界不执行任何业务工作。特别地，它绝不重试模型/工具调用，
    也不持久化轮次。``KeyboardInterrupt`` 等进程控制异常刻意不捕获。
    """

    try:
        return recorder.finish_turn(outcome, error_code=error_code)
    except (TypeError, ValueError, RuntimeError):
        _LOGGER.warning(TRACE_RECORDING_FAILED, exc_info=True)
        return None


# 第二阶段：接入实际问答流程。





# 第三阶段：存储、API 与页面接入。

# 已实现 14：api.py 返回公开摘要（待统一验证）。
# TraceEventResponse / TurnTraceResponse 定义公开结构，_trace_response() 复用
# trace_to_dict() 的字段限制与校验；摘要格式不可用时降级为 None，保留有效回答。
# AnswerResponse、TurnResponse 及 ErrorResponse.result 携带可选 trace。
# _answer_response()、_turn_response() 统一转换，本轮、历史与保存失败响应复用同一规则。
# 未进入 Agent 的请求没有虚构摘要；沿用现有 HTTP、业务状态与持久化语义。

# 已实现 15：web/assets/app.js 展示本轮与历史的执行详情（待统一验证）。
# message() 复用本轮与历史的回答渲染，添加默认收起的执行详情。
# 展示 Skill 选择/准备/进入模型请求状态、真实阶段、数量、耗时与错误类别，
# 并保留 AnswerResult 中已有的调用计数、结束原因和索引版本。
# 内容使用安全 DOM 文本渲染；缺少摘要、未知状态、事件截断和未保存轮有对应提示。
# 旧历史不从当前下拉框推断 Skill，缺失耗时不当作 0 ms；不新增问答请求。

# 已实现 16：页面样式与 README.md 使用说明（待统一验证）。
# web/assets/app.css 提供折叠面板、阶段文字状态、长标识换行与移动端布局。
# 采用原生 details/summary 支持键盘展开，不需要额外的静态 HTML 入口。
# README.md 说明查看入口、计数与耗时口径、Skill 状态、旧历史及未保存失败轮限制。
# 本功能在回答结束后一次性展示实际记录；沿用现有忙碌与断连恢复流程。


# 验证项：以下 TODO 尚未统一执行。

# TODO 17：集中验证收集器与序列化。
# 使用假时钟覆盖顺序、耗时、重复结束、定稿快照、事件截断和多轮隔离。
# 覆盖 None/旧摘要、未知版本、非法枚举、非有限数值及不允许公开的字段。
# 可新增 tests/test_trace.py；验证实际行为和边界，避免只测试字段赋值。

# TODO 18：集中验证 Agent 执行记录。
# 用假模型和假检索覆盖普通问答、三个 Skill、提前结束、补查、空结果和服务错误。
# 覆盖 Skill 仅选择/仅准备/实际进入调用三种状态，以及内部重试的计数口径。
# 覆盖首次检索只记录一次、被拒绝工具不执行、引用校验失败和调用预算耗尽。
# 确认启用记录不会增加模型/工具调用，也不会改变原有回答、引用和保存规则。

# TODO 19：集中验证存储、API 和前端兼容。
# 通过临时 SQLite 验证旧会话、新摘要和损坏摘要处理，不修改真实会话数据库。
# 验证正常响应、错误响应、历史分页都保持同一摘要，未保存失败轮不进入历史。
# 验证页面折叠展开、移动端、特殊字符、未知状态和 trace 缺失的降级展示。
# 保留现有断连、刷新历史和忙碌状态的验证范围，不把没有记录误报为没有执行。

# TODO 20：最后进行真实小范围联调。
# 用户安排统一联调时，再运行普通问答、概念讲解、概念比较、代码解读各一轮。
# 核对页面记录、接口返回与实际执行是否一致，刷新页面后读取已保存摘要。
# 对照使用同一知识库与模型配置的独立会话；本模块不输出 Skill 效果好坏的评分，
# 答案质量对比继续由后续 evaluation.py 承担。
