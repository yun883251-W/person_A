"""会话任务的数据契约、校验、持久化格式与公开投影。

功能：分离模型提供的意图候选、程序校验后的决策、持久化业务状态和公开摘要。
输入：当前用户消息、程序读取的会话状态、实际可用能力及现有 MBTI 状态。
输出目标：IntentProposal、DialogueDecision、SessionBusinessState 和状态更新提案。
约束：不执行识别、检索、工具、状态写入；模型输出不得自行取得执行权限。

本模块只依赖标准库和业务参数契约，不执行识别、状态写入或外部操作。
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any


class DialogueIntent(str, Enum):
    """用户的会话级意图，与业务路由无关。"""

    NEW_BUSINESS = "new_business"
    CONTINUE_BUSINESS = "continue_business"
    CANCEL_BUSINESS = "cancel_business"
    PAUSE_BUSINESS = "pause_business"
    RESUME_BUSINESS = "resume_business"
    CHITCHAT = "chitchat"
    UNCLEAR = "unclear"
    UNSUPPORTED = "unsupported"


class BusinessType(str, Enum):
    """请求的业务领域；未知领域用 None 表示。"""

    KNOWLEDGE = "knowledge"
    BILIBILI_SEARCH = "bilibili_search"
    MBTI = "mbti"
    MCP_TOOL = "mcp_tool"


class ExecutionMode(str, Enum):
    """请求的执行方式，与业务领域无关。"""

    SINGLE = "single"
    PLAN = "plan"

SCHEMA_VERSION = 1
MAX_MODEL_JSON_LENGTH = 16_384
MAX_STRING_LENGTH = 4_096
MAX_SLOT_COUNT = 32
MAX_OBJECT_DEPTH = 4
_TEST_ID_RE = re.compile(r"[a-f0-9]{32}")
_SLOT_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")


class DialogueContractError(ValueError):
    """稳定的校验失败，其消息绝不嵌入私有输入。"""

    def __init__(self, code: str, message: str = "Invalid dialogue contract") -> None:
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str = "Invalid dialogue contract") -> None:
    raise DialogueContractError(code, message)


def _enum_value(enum_type: type[Enum], value: object, code: str) -> Any:
    if type(value) is not str:
        _fail(code)
    try:
        return enum_type(value)
    except ValueError:
        _fail(code)


def _validate_json_value(value: object, *, depth: int = 0) -> None:
    if depth > MAX_OBJECT_DEPTH:
        _fail("intent_model_object_too_deep")
    if value is None or type(value) in {bool, int, float}:
        return
    if type(value) is str:
        if len(value) > MAX_STRING_LENGTH:
            _fail("intent_model_string_too_long")
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_SLOT_COUNT:
            _fail("intent_model_too_many_fields")
        for key, item in value.items():
            if type(key) is not str or not _SLOT_NAME_RE.fullmatch(key):
                _fail("intent_model_invalid_field_name")
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_SLOT_COUNT:
            _fail("intent_model_too_many_items")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    _fail("intent_model_invalid_value_type")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _frozen_slots(value: Mapping[str, Any], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or len(value) > MAX_SLOT_COUNT:
        _fail(code)
    _validate_json_value(value)
    return _freeze(value)


def _slot_names(value: Sequence[str], code: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(code)
    result = tuple(value)
    if len(result) > MAX_SLOT_COUNT or len(set(result)) != len(result):
        _fail(code)
    if any(type(name) is not str or not _SLOT_NAME_RE.fullmatch(name) for name in result):
        _fail(code)
    return result


@dataclass(frozen=True, slots=True)
class IntentProposal:
    """为兼容性保留的旧提议 DTO，绝不具有执行权限。

    统一运行时使用 intent_types.IntentDecision 加宿主生命周期规则；
    它不会运行第二个模型来输出这个对话 schema。
    """

    intent: DialogueIntent
    business_type: BusinessType | None
    execution_mode: ExecutionMode
    slots: Mapping[str, Any] = field(default_factory=dict)
    missing_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.intent, DialogueIntent):
            _fail("intent_model_invalid_intent")
        if self.business_type is not None and not isinstance(self.business_type, BusinessType):
            _fail("intent_model_invalid_business_type")
        if not isinstance(self.execution_mode, ExecutionMode):
            _fail("intent_model_invalid_execution_mode")
        object.__setattr__(self, "slots", _frozen_slots(self.slots, "intent_model_invalid_slots"))
        object.__setattr__(self, "missing_slots", _slot_names(self.missing_slots, "intent_model_invalid_missing_slots"))
        if set(self.slots) & set(self.missing_slots):
            _fail("slot_also_marked_missing")
        _validate_intent_combination(self.intent, self.business_type, self.execution_mode, self.slots)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("intent_model_duplicate_key")
        result[key] = value
    return result


def parse_intent_proposal(value: str | Mapping[str, Any]) -> IntentProposal:
    """精确解析五个模型自有字段，并拒绝重复的 JSON 键。"""

    if type(value) is str:
        if not value or len(value) > MAX_MODEL_JSON_LENGTH:
            _fail("intent_model_response_too_long" if value else "intent_model_empty_response")
        try:
            payload = json.loads(value, object_pairs_hook=_unique_object)
        except DialogueContractError:
            raise
        except (json.JSONDecodeError, RecursionError):
            _fail("intent_model_invalid_json")
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        _fail("intent_model_invalid_structure")
    required = {"intent", "business_type", "execution_mode", "slots", "missing_slots"}
    if not isinstance(payload, dict) or set(payload) != required:
        _fail("intent_model_invalid_fields")
    business_raw = payload["business_type"]
    business = None if business_raw is None else _enum_value(
        BusinessType, business_raw, "intent_model_invalid_business_type"
    )
    return IntentProposal(
        intent=_enum_value(DialogueIntent, payload["intent"], "intent_model_invalid_intent"),
        business_type=business,
        execution_mode=_enum_value(
            ExecutionMode, payload["execution_mode"], "intent_model_invalid_execution_mode"
        ),
        slots=payload["slots"],
        missing_slots=_slot_names(payload["missing_slots"], "intent_model_invalid_missing_slots"),
    )


def _validate_intent_combination(
    intent: DialogueIntent,
    business_type: BusinessType | None,
    execution_mode: ExecutionMode,
    slots: Mapping[str, Any],
) -> None:
    passive = {DialogueIntent.CHITCHAT, DialogueIntent.UNCLEAR, DialogueIntent.UNSUPPORTED}
    if intent in passive and (business_type is not None or slots):
        _fail("invalid_intent_business_combination")
    if business_type is None and slots:
        _fail("slots_without_business")
    if execution_mode is ExecutionMode.PLAN and intent is not DialogueIntent.NEW_BUSINESS:
        _fail("invalid_plan_intent")
    if execution_mode is ExecutionMode.PLAN and slots:
        _fail("invalid_plan_slots")
    if intent in {DialogueIntent.PAUSE_BUSINESS, DialogueIntent.RESUME_BUSINESS} and business_type not in {
        None, BusinessType.MBTI
    }:
        _fail("invalid_business_control")


def _validate_schema_value(value: Any, schema: Mapping[str, Any], path: str) -> None:
    schema_type = schema.get("type")
    valid = {
        "string": type(value) is str,
        "integer": type(value) is int,
        "number": type(value) in {int, float},
        "boolean": type(value) is bool,
        "array": isinstance(value, (list, tuple)),
        "object": isinstance(value, Mapping),
        "null": value is None,
    }
    if isinstance(schema_type, list):
        if not any(valid.get(item, False) for item in schema_type):
            _fail("invalid_business_arguments", f"Invalid value for {path}")
    elif schema_type is not None and not valid.get(schema_type, False):
        _fail("invalid_business_arguments", f"Invalid value for {path}")
    if "enum" in schema and value not in schema["enum"]:
        _fail("invalid_business_arguments", f"Invalid value for {path}")
    if type(value) is str:
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", MAX_STRING_LENGTH):
            _fail("invalid_business_arguments", f"Invalid value for {path}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            _fail("invalid_business_arguments", f"Invalid value for {path}")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", ())
        if not isinstance(properties, Mapping) or not isinstance(required, (list, tuple)):
            _fail("invalid_tool_schema")
        if schema.get("additionalProperties") is False and not set(value) <= set(properties):
            _fail("invalid_business_arguments")
        if not set(required) <= set(value):
            _fail("missing_business_arguments")
        for key, item in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, Mapping):
                _validate_schema_value(item, child_schema, f"{path}.{key}")
    if isinstance(value, (list, tuple)) and isinstance(schema.get("items"), Mapping):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", MAX_SLOT_COUNT):
            _fail("invalid_business_arguments", f"Invalid value for {path}")
        for index, item in enumerate(value):
            _validate_schema_value(item, schema["items"], f"{path}[{index}]")


def required_business_slots(
    business_type: BusinessType, *, tool_schema: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    if business_type is BusinessType.BILIBILI_SEARCH:
        return ("query",)
    if business_type is BusinessType.MBTI:
        return ("action",)
    if business_type is BusinessType.MCP_TOOL:
        if not isinstance(tool_schema, Mapping) or tool_schema.get("type") != "object":
            _fail("unregistered_tool_schema")
        required = tool_schema.get("required", ())
        return _slot_names(required, "invalid_tool_schema")
    return ()


def validate_business_slots(
    business_type: BusinessType,
    slots: Mapping[str, Any],
    missing_slots: Sequence[str] = (),
    *,
    tool_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """校验已确认参数，但不把声明的缺失值当作已确认。"""

    if not isinstance(business_type, BusinessType):
        _fail("invalid_business_type")
    frozen = _frozen_slots(slots, "invalid_business_arguments")
    missing = _slot_names(missing_slots, "invalid_missing_slots")
    if set(frozen) & set(missing):
        _fail("slot_also_marked_missing")
    required = required_business_slots(business_type, tool_schema=tool_schema)
    if set(required) - set(frozen) - set(missing):
        _fail("missing_business_arguments")
    if business_type is BusinessType.KNOWLEDGE:
        if frozen or missing:
            _fail("knowledge_slots_not_allowed")
        return {}
    if business_type is BusinessType.BILIBILI_SEARCH:
        if set(frozen) - {"query"} or set(missing) - {"query"}:
            _fail("invalid_business_arguments")
        if "query" not in frozen:
            return {}
        from browser_mcp.contracts import InvalidSearchArguments, validate_search_arguments

        try:
            return validate_search_arguments(frozen)
        except InvalidSearchArguments:
            _fail("invalid_business_arguments")
    if business_type is BusinessType.MBTI:
        allowed = {"action", "answer", "question_number", "test_id"}
        if set(frozen) - allowed or set(missing) - allowed:
            _fail("invalid_business_arguments")
        if "action" not in frozen:
            return {}
        from ..assessment import ACTIONS

        action = frozen["action"]
        if type(action) is not str or action not in ACTIONS:
            _fail("invalid_business_arguments")
        if "answer" in frozen and (type(frozen["answer"]) is not int or not 1 <= frozen["answer"] <= 5):
            _fail("invalid_business_arguments")
        if "question_number" in frozen and (
            type(frozen["question_number"]) is not int or not 1 <= frozen["question_number"] <= 48
        ):
            _fail("invalid_business_arguments")
        if "test_id" in frozen and (
            type(frozen["test_id"]) is not str or _TEST_ID_RE.fullmatch(frozen["test_id"]) is None
        ):
            _fail("invalid_business_arguments")
        if action == "answer" and "answer" not in frozen and "answer" not in missing:
            _fail("missing_business_arguments")
        return _thaw(frozen)
    if not isinstance(tool_schema, Mapping) or tool_schema.get("type") != "object":
        _fail("unregistered_tool_schema")
    properties = tool_schema.get("properties", {})
    if not isinstance(properties, Mapping) or set(frozen) - set(properties) or set(missing) - set(properties):
        _fail("invalid_business_arguments")
    partial_schema = dict(tool_schema)
    partial_schema["required"] = [name for name in required if name not in missing]
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError, ValidationError

        Draft202012Validator.check_schema(partial_schema)
        Draft202012Validator(partial_schema).validate(_thaw(frozen))
    except ImportError:
        _validate_schema_value(frozen, partial_schema, "slots")
    except SchemaError:
        _fail("invalid_tool_schema")
    except ValidationError:
        _fail("invalid_business_arguments")
    return _thaw(frozen)


class SlotSource(str, Enum):
    CURRENT_USER_INPUT = "current_user_input"
    PENDING_USER_REQUEST = "pending_user_request"
    SAME_REQUEST_CLAUSE = "same_request_clause"


@dataclass(frozen=True, slots=True)
class SlotBinding:
    """宿主为某个已确认参数创建的来源信息。"""

    slot_name: str
    value: Any
    raw_input: str
    session_id: str
    source_turn_id: str
    source: SlotSource
    source_clause: str | None = None

    def __post_init__(self) -> None:
        if type(self.slot_name) is not str or _SLOT_NAME_RE.fullmatch(self.slot_name) is None:
            _fail("invalid_slot_binding")
        _validate_json_value(self.value)
        object.__setattr__(self, "value", _freeze(self.value))
        for value in (self.raw_input, self.session_id, self.source_turn_id):
            if type(value) is not str or not value or len(value) > MAX_STRING_LENGTH:
                _fail("invalid_slot_binding")
        if not isinstance(self.source, SlotSource):
            _fail("invalid_slot_source")
        if self.source is SlotSource.SAME_REQUEST_CLAUSE:
            if type(self.source_clause) is not str or not self.source_clause:
                _fail("missing_source_clause")
        elif self.source_clause is not None:
            _fail("unexpected_source_clause")


class DecisionSource(str, Enum):
    RULES = "rules"
    MODEL = "model"
    FALLBACK = "fallback"
    PROGRAM = "program"


@dataclass(frozen=True, slots=True)
class DialogueDecision:
    """经程序校验的决定。只有该类型可以进入执行路由。"""

    intent: DialogueIntent
    business_type: BusinessType | None
    execution_mode: ExecutionMode = ExecutionMode.SINGLE
    confirmed_slots: Mapping[str, Any] = field(default_factory=dict)
    missing_slots: tuple[str, ...] = ()
    slot_bindings: tuple[SlotBinding, ...] = ()
    clarification_question: str = ""
    decision_source: DecisionSource = DecisionSource.RULES
    reason_code: str = ""
    expected_revision: int = 0
    allowed_tool_names: tuple[str, ...] = ()
    raw_plan_request: str = ""
    # 只保留已接受的路由数据。绝不保留更早的分类器决定：
    # 语义澄清之后，它不得覆盖当前决定。
    retrieval_query: str = ""
    direct_response_kind: str = ""
    authorized_user_input: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.intent, DialogueIntent) or (
            self.business_type is not None and not isinstance(self.business_type, BusinessType)
        ) or not isinstance(self.execution_mode, ExecutionMode):
            _fail("invalid_dialogue_decision")
        slots = _frozen_slots(self.confirmed_slots, "invalid_business_arguments")
        missing = _slot_names(self.missing_slots, "invalid_missing_slots")
        if set(slots) & set(missing):
            _fail("slot_also_marked_missing")
        if (
            self.business_type is not None
            and self.business_type is not BusinessType.MCP_TOOL
            and self.intent in {DialogueIntent.NEW_BUSINESS, DialogueIntent.CONTINUE_BUSINESS}
            and self.execution_mode is ExecutionMode.SINGLE
        ):
            slots = _freeze(validate_business_slots(self.business_type, slots, missing))
        bindings = tuple(self.slot_bindings)
        if any(not isinstance(binding, SlotBinding) for binding in bindings):
            _fail("invalid_slot_binding")
        if len({binding.slot_name for binding in bindings}) != len(bindings):
            _fail("duplicate_slot_binding")
        if set(slots) != {binding.slot_name for binding in bindings}:
            _fail("slot_binding_mismatch")
        for binding in bindings:
            if _thaw(binding.value) != _thaw(slots[binding.slot_name]):
                _fail("slot_binding_mismatch")
        if type(self.expected_revision) is not int or self.expected_revision < 0:
            _fail("invalid_expected_revision")
        if not isinstance(self.decision_source, DecisionSource):
            _fail("invalid_decision_source")
        for text in (self.clarification_question, self.reason_code, self.raw_plan_request):
            if type(text) is not str or len(text) > MAX_STRING_LENGTH:
                _fail("invalid_dialogue_decision")
        if type(self.retrieval_query) is not str or len(self.retrieval_query) > 200_000:
            _fail("invalid_retrieval_query")
        if type(self.direct_response_kind) is not str or self.direct_response_kind not in {"", "greeting", "scope"}:
            _fail("invalid_direct_response_kind")
        if self.direct_response_kind and self.intent is not DialogueIntent.CHITCHAT:
            _fail("invalid_direct_response_kind")
        if type(self.authorized_user_input) is not str or len(self.authorized_user_input) > 200_000:
            _fail("invalid_authorized_input")
        tools = tuple(self.allowed_tool_names)
        if len(tools) != len(set(tools)) or any(type(name) is not str or not name for name in tools):
            _fail("invalid_allowed_tools")
        _validate_intent_combination(self.intent, self.business_type, self.execution_mode, slots)
        if self.intent is DialogueIntent.UNSUPPORTED and tools:
            _fail("unsupported_cannot_execute")
        if self.business_type is BusinessType.KNOWLEDGE and tools:
            _fail("knowledge_cannot_authorize_tools")
        if tools and self.business_type not in {BusinessType.BILIBILI_SEARCH, BusinessType.MCP_TOOL}:
            _fail("invalid_allowed_tools")
        if self.execution_mode is ExecutionMode.PLAN and not self.raw_plan_request:
            _fail("missing_raw_plan_request")
        object.__setattr__(self, "confirmed_slots", slots)
        object.__setattr__(self, "missing_slots", missing)
        object.__setattr__(self, "slot_bindings", bindings)
        object.__setattr__(self, "allowed_tool_names", tools)


def validate_dialogue_decision(
    decision: DialogueDecision, *, tool_schema: Mapping[str, Any] | None = None
) -> DialogueDecision:
    """在执行边界重新校验可信决定。"""

    if not isinstance(decision, DialogueDecision):
        _fail("invalid_dialogue_decision")
    if decision.business_type is None:
        if decision.confirmed_slots or decision.missing_slots or decision.slot_bindings:
            _fail("slots_without_business")
        return decision
    if decision.intent in {DialogueIntent.NEW_BUSINESS, DialogueIntent.CONTINUE_BUSINESS}:
        # 第一阶段中，通用且已获授权的 MCP 请求保持既有执行器
        # 契约。它们刻意不做成可恢复的，因此这里没有
        # 需要校验的跨轮参数载荷。
        if (
            decision.business_type is BusinessType.MCP_TOOL
            and not decision.confirmed_slots and not decision.missing_slots
        ):
            return decision
        normalized = validate_business_slots(
            decision.business_type,
            decision.confirmed_slots,
            decision.missing_slots,
            tool_schema=tool_schema,
        )
        if normalized != _thaw(decision.confirmed_slots):
            _fail("unnormalized_business_arguments")
    return decision


@dataclass(frozen=True, slots=True)
class PendingBusiness:
    """一个会话中唯一可恢复、尚未派发的请求。"""

    session_id: str
    business_type: BusinessType
    original_request: str
    request_turn_id: str
    confirmed_slots: Mapping[str, Any] = field(default_factory=dict)
    slot_bindings: tuple[SlotBinding, ...] = ()
    missing_slots: tuple[str, ...] = ()
    clarification_question: str = ""
    clarification_count: int = 1

    def __post_init__(self) -> None:
        if self.business_type not in {BusinessType.BILIBILI_SEARCH, BusinessType.MBTI}:
            _fail("invalid_pending_business")
        for value in (self.session_id, self.original_request, self.request_turn_id):
            if type(value) is not str or not value or len(value) > MAX_STRING_LENGTH:
                _fail("invalid_pending_business")
        slots = _frozen_slots(self.confirmed_slots, "invalid_business_arguments")
        missing = _slot_names(self.missing_slots, "invalid_missing_slots")
        slots = _freeze(validate_business_slots(self.business_type, slots, missing))
        bindings = tuple(self.slot_bindings)
        if len(bindings) != len({binding.slot_name for binding in bindings}) or set(slots) != {
            binding.slot_name for binding in bindings
        } or any(
            binding.session_id != self.session_id for binding in bindings
        ):
            _fail("slot_binding_mismatch")
        for binding in bindings:
            if _thaw(binding.value) != _thaw(slots[binding.slot_name]):
                _fail("slot_binding_mismatch")
        if set(slots) & set(missing):
            _fail("slot_also_marked_missing")
        if type(self.clarification_count) is not int or self.clarification_count < 0:
            _fail("invalid_clarification_count")
        if type(self.clarification_question) is not str or len(self.clarification_question) > MAX_STRING_LENGTH:
            _fail("invalid_pending_business")
        object.__setattr__(self, "confirmed_slots", slots)
        object.__setattr__(self, "missing_slots", missing)
        object.__setattr__(self, "slot_bindings", bindings)


class BusinessPhase(str, Enum):
    IDLE = "idle"
    AWAITING_INPUT = "awaiting_input"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    DISPATCHING = "dispatching"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXECUTION_UNKNOWN = "execution_unknown"


@dataclass(frozen=True, slots=True)
class BusinessFocus:
    business_type: BusinessType
    phase: BusinessPhase
    test_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.business_type, BusinessType) or not isinstance(self.phase, BusinessPhase):
            _fail("invalid_business_focus")
        if self.business_type is BusinessType.MBTI:
            if self.test_id is not None and (
                type(self.test_id) is not str or _TEST_ID_RE.fullmatch(self.test_id) is None
            ):
                _fail("invalid_mbti_test_id")
        elif self.test_id is not None:
            _fail("unexpected_mbti_test_id")


@dataclass(frozen=True, slots=True)
class SessionBusinessState:
    session_id: str
    schema_version: int = SCHEMA_VERSION
    revision: int = 0
    focus: BusinessFocus | None = None
    pending: PendingBusiness | None = None
    updated_at: str = ""


class CommitTiming(str, Enum):
    WITH_TURN = "with_turn"
    BEFORE_DISPATCH = "before_dispatch"
    AFTER_EXECUTION = "after_execution"


@dataclass(frozen=True, slots=True)
class StateTransition:
    expected_revision: int
    next_state: SessionBusinessState
    reason_code: str
    next_actions: tuple[str, ...] = ()
    commit_timing: CommitTiming = CommitTiming.WITH_TURN

    def __post_init__(self) -> None:
        if type(self.expected_revision) is not int or self.expected_revision < 0:
            _fail("invalid_expected_revision")
        if not isinstance(self.next_state, SessionBusinessState):
            _fail("invalid_next_state")
        if type(self.reason_code) is not str or self.reason_code not in REASON_CODE_CATEGORIES:
            _fail("unknown_reason_code")
        if not isinstance(self.commit_timing, CommitTiming):
            _fail("invalid_commit_timing")
        actions = tuple(self.next_actions)
        if len(actions) > 16 or any(type(action) is not str or not action for action in actions):
            _fail("invalid_next_actions")
        object.__setattr__(self, "next_actions", actions)


_PHASES_BY_BUSINESS = {
    BusinessType.KNOWLEDGE: {BusinessPhase.ACTIVE, BusinessPhase.COMPLETED, BusinessPhase.CANCELLED},
    BusinessType.BILIBILI_SEARCH: {
        BusinessPhase.AWAITING_INPUT, BusinessPhase.READY, BusinessPhase.DISPATCHING,
        BusinessPhase.COMPLETED, BusinessPhase.CANCELLED, BusinessPhase.EXECUTION_UNKNOWN,
    },
    BusinessType.MBTI: {
        BusinessPhase.AWAITING_INPUT, BusinessPhase.READY, BusinessPhase.ACTIVE,
        BusinessPhase.PAUSED, BusinessPhase.COMPLETED, BusinessPhase.CANCELLED,
    },
    BusinessType.MCP_TOOL: {
        BusinessPhase.AWAITING_INPUT, BusinessPhase.READY, BusinessPhase.DISPATCHING,
        BusinessPhase.COMPLETED, BusinessPhase.CANCELLED, BusinessPhase.EXECUTION_UNKNOWN,
    },
}

_ALLOWED_PHASE_TRANSITIONS = {
    BusinessPhase.IDLE: {BusinessPhase.AWAITING_INPUT, BusinessPhase.READY, BusinessPhase.ACTIVE, BusinessPhase.COMPLETED},
    BusinessPhase.AWAITING_INPUT: {BusinessPhase.AWAITING_INPUT, BusinessPhase.READY, BusinessPhase.CANCELLED},
    BusinessPhase.READY: {BusinessPhase.DISPATCHING, BusinessPhase.ACTIVE, BusinessPhase.CANCELLED},
    BusinessPhase.ACTIVE: {BusinessPhase.ACTIVE, BusinessPhase.PAUSED, BusinessPhase.COMPLETED, BusinessPhase.CANCELLED},
    BusinessPhase.PAUSED: {BusinessPhase.ACTIVE, BusinessPhase.CANCELLED},
    BusinessPhase.DISPATCHING: {BusinessPhase.COMPLETED, BusinessPhase.EXECUTION_UNKNOWN, BusinessPhase.CANCELLED},
    BusinessPhase.COMPLETED: set(),
    BusinessPhase.CANCELLED: set(),
    BusinessPhase.EXECUTION_UNKNOWN: set(),
}


def _validate_timestamp(value: object) -> None:
    if type(value) is not str or not value or len(value) > 64:
        _fail("invalid_state_timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail("invalid_state_timestamp")
    if parsed.tzinfo is None:
        _fail("invalid_state_timestamp")


def validate_business_state(state: SessionBusinessState) -> SessionBusinessState:
    """校验所有持久化不变量，但不修复或重置状态。"""

    if not isinstance(state, SessionBusinessState):
        _fail("invalid_business_state")
    if type(state.session_id) is not str or not state.session_id or len(state.session_id) > 256:
        _fail("invalid_session_id")
    if type(state.schema_version) is not int or state.schema_version != SCHEMA_VERSION:
        _fail("unsupported_business_state_version")
    if type(state.revision) is not int or state.revision < 0:
        _fail("invalid_state_revision")
    _validate_timestamp(state.updated_at)
    if state.focus is None:
        if state.pending is not None:
            _fail("pending_without_focus")
        return state
    if not isinstance(state.focus, BusinessFocus) or state.focus.phase is BusinessPhase.IDLE:
        _fail("invalid_business_focus")
    if state.focus.phase not in _PHASES_BY_BUSINESS[state.focus.business_type]:
        _fail("invalid_business_phase")
    pending = state.pending
    if pending is not None:
        if pending.session_id != state.session_id:
            _fail("cross_session_pending")
        if pending.business_type is not state.focus.business_type:
            _fail("pending_focus_mismatch")
        if state.focus.phase not in {BusinessPhase.AWAITING_INPUT, BusinessPhase.READY}:
            _fail("pending_in_non_resumable_phase")
        if state.focus.phase is BusinessPhase.AWAITING_INPUT and not pending.missing_slots:
            _fail("awaiting_input_without_missing_slots")
        if state.focus.phase is BusinessPhase.READY and pending.missing_slots:
            _fail("ready_with_missing_slots")
    elif state.focus.phase in {BusinessPhase.AWAITING_INPUT, BusinessPhase.READY}:
        _fail("resumable_phase_without_pending")
    if state.focus.business_type is BusinessType.MBTI and state.focus.phase in {
        BusinessPhase.ACTIVE, BusinessPhase.PAUSED, BusinessPhase.COMPLETED, BusinessPhase.CANCELLED
    } and state.focus.test_id is None:
        _fail("missing_mbti_test_id")
    return state


def empty_business_state(session_id: str, *, updated_at: str) -> SessionBusinessState:
    state = SessionBusinessState(session_id=session_id, updated_at=updated_at)
    return validate_business_state(state)


def validate_state_transition(
    previous: SessionBusinessState, transition: StateTransition
) -> StateTransition:
    previous = validate_business_state(previous)
    next_state = validate_business_state(transition.next_state)
    if transition.expected_revision != previous.revision or next_state.revision != previous.revision + 1:
        _fail("state_conflict")
    if next_state.session_id != previous.session_id:
        _fail("cross_session_state")
    old_phase = previous.focus.phase if previous.focus else BusinessPhase.IDLE
    new_phase = next_state.focus.phase if next_state.focus else BusinessPhase.IDLE
    replacing = transition.reason_code == "pending_replaced" and next_state.focus is not None
    starting_new = transition.reason_code in {"pending_created", "business_started"} and next_state.focus is not None
    clearing = next_state.focus is None and transition.reason_code in {"pending_cancelled", "business_cleared"}
    if not replacing and not starting_new and not clearing and new_phase not in _ALLOWED_PHASE_TRANSITIONS[old_phase]:
        _fail("invalid_state_transition")
    if transition.commit_timing is CommitTiming.BEFORE_DISPATCH and new_phase is not BusinessPhase.DISPATCHING:
        _fail("invalid_dispatch_transition")
    return transition


def _binding_to_payload(binding: SlotBinding) -> dict[str, Any]:
    return {
        "slot_name": binding.slot_name,
        "value": _thaw(binding.value),
        "raw_input": binding.raw_input,
        "session_id": binding.session_id,
        "source_turn_id": binding.source_turn_id,
        "source": binding.source.value,
        "source_clause": binding.source_clause,
    }


def _pending_to_payload(pending: PendingBusiness) -> dict[str, Any]:
    return {
        "session_id": pending.session_id,
        "business_type": pending.business_type.value,
        "original_request": pending.original_request,
        "request_turn_id": pending.request_turn_id,
        "confirmed_slots": _thaw(pending.confirmed_slots),
        "slot_bindings": [_binding_to_payload(binding) for binding in pending.slot_bindings],
        "missing_slots": list(pending.missing_slots),
        "clarification_question": pending.clarification_question,
        "clarification_count": pending.clarification_count,
    }


def state_to_payload(state: SessionBusinessState) -> dict[str, Any]:
    """返回完整、严格的持久化载荷（不适合直接给客户端）。"""

    state = validate_business_state(state)
    focus = None
    if state.focus is not None:
        focus = {
            "business_type": state.focus.business_type.value,
            "phase": state.focus.phase.value,
            "test_id": state.focus.test_id,
        }
    return {
        "schema_version": state.schema_version,
        "session_id": state.session_id,
        "revision": state.revision,
        "focus": focus,
        "pending": None if state.pending is None else _pending_to_payload(state.pending),
        "updated_at": state.updated_at,
    }


def _exact_mapping(value: object, fields: set[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(code)
    return value


def _binding_from_payload(value: object) -> SlotBinding:
    item = _exact_mapping(
        value,
        {"slot_name", "value", "raw_input", "session_id", "source_turn_id", "source", "source_clause"},
        "invalid_slot_binding",
    )
    return SlotBinding(
        slot_name=item["slot_name"], value=item["value"], raw_input=item["raw_input"],
        session_id=item["session_id"], source_turn_id=item["source_turn_id"],
        source=_enum_value(SlotSource, item["source"], "invalid_slot_source"),
        source_clause=item["source_clause"],
    )


def _pending_from_payload(value: object) -> PendingBusiness:
    item = _exact_mapping(
        value,
        {
            "session_id", "business_type", "original_request", "request_turn_id", "confirmed_slots",
            "slot_bindings", "missing_slots", "clarification_question", "clarification_count",
        },
        "invalid_pending_business",
    )
    raw_bindings = item["slot_bindings"]
    if not isinstance(raw_bindings, list):
        _fail("invalid_slot_binding")
    return PendingBusiness(
        session_id=item["session_id"],
        business_type=_enum_value(BusinessType, item["business_type"], "invalid_business_type"),
        original_request=item["original_request"], request_turn_id=item["request_turn_id"],
        confirmed_slots=item["confirmed_slots"],
        slot_bindings=tuple(_binding_from_payload(binding) for binding in raw_bindings),
        missing_slots=_slot_names(item["missing_slots"], "invalid_missing_slots"),
        clarification_question=item["clarification_question"],
        clarification_count=item["clarification_count"],
    )


def state_from_payload(value: Mapping[str, Any]) -> SessionBusinessState:
    """只解码当前 schema；未知或损坏状态按失败关闭处理。"""

    item = _exact_mapping(
        value, {"schema_version", "session_id", "revision", "focus", "pending", "updated_at"},
        "invalid_business_state",
    )
    focus = None
    if item["focus"] is not None:
        raw_focus = _exact_mapping(
            item["focus"], {"business_type", "phase", "test_id"}, "invalid_business_focus"
        )
        focus = BusinessFocus(
            business_type=_enum_value(BusinessType, raw_focus["business_type"], "invalid_business_type"),
            phase=_enum_value(BusinessPhase, raw_focus["phase"], "invalid_business_phase"),
            test_id=raw_focus["test_id"],
        )
    pending = None if item["pending"] is None else _pending_from_payload(item["pending"])
    state = SessionBusinessState(
        schema_version=item["schema_version"], session_id=item["session_id"],
        revision=item["revision"], focus=focus, pending=pending, updated_at=item["updated_at"],
    )
    return validate_business_state(state)


def public_business_state(state: SessionBusinessState) -> dict[str, Any]:
    """只为 HTTP/CLI 消费者投影可安全展示、不授予权限的状态。"""

    state = validate_business_state(state)
    if state.focus is None:
        return {
            "business_type": None, "phase": BusinessPhase.IDLE.value,
            "waiting_for_input": False, "missing_slots": [], "clarification_question": "",
            "parameters": {}, "allowed_actions": [],
        }
    pending = state.pending
    parameters: dict[str, Any] = {}
    if pending is not None and pending.business_type is BusinessType.BILIBILI_SEARCH:
        query = pending.confirmed_slots.get("query")
        if type(query) is str:
            parameters["query"] = query
    allowed_actions: list[str] = []
    if state.focus.phase in {BusinessPhase.AWAITING_INPUT, BusinessPhase.READY}:
        allowed_actions.append("cancel")
    if state.focus.business_type is BusinessType.MBTI:
        if state.focus.phase is BusinessPhase.ACTIVE:
            allowed_actions.extend(("pause", "cancel"))
        elif state.focus.phase is BusinessPhase.PAUSED:
            allowed_actions.extend(("resume", "cancel"))
    return {
        "business_type": state.focus.business_type.value,
        "phase": state.focus.phase.value,
        "waiting_for_input": state.focus.phase is BusinessPhase.AWAITING_INPUT,
        "missing_slots": [] if pending is None else list(pending.missing_slots),
        "clarification_question": "" if pending is None else pending.clarification_question,
        "parameters": parameters,
        "allowed_actions": list(dict.fromkeys(allowed_actions)),
    }


class ReasonCategory(str, Enum):
    SUCCESS = "success"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"
    VALIDATION_ERROR = "validation_error"
    SERVICE_ERROR = "service_error"
    STATE_CONFLICT = "state_conflict"
    EXECUTION_UNKNOWN = "execution_unknown"


REASON_CODE_CATEGORIES: Mapping[str, ReasonCategory] = MappingProxyType({
    "pending_created": ReasonCategory.NEEDS_CLARIFICATION,
    "slot_filled": ReasonCategory.SUCCESS,
    "clarification_unresolved": ReasonCategory.NEEDS_CLARIFICATION,
    "pending_cancelled": ReasonCategory.SUCCESS,
    "pending_replaced": ReasonCategory.SUCCESS,
    "business_cleared": ReasonCategory.SUCCESS,
    "no_active_business": ReasonCategory.NEEDS_CLARIFICATION,
    "ambiguous_business_target": ReasonCategory.NEEDS_CLARIFICATION,
    "state_conflict": ReasonCategory.STATE_CONFLICT,
    "execution_unknown": ReasonCategory.EXECUTION_UNKNOWN,
    "intent_model_timeout": ReasonCategory.SERVICE_ERROR,
    "model_call_budget_exhausted": ReasonCategory.SERVICE_ERROR,
    "intent_model_invalid_json": ReasonCategory.VALIDATION_ERROR,
    "intent_model_invalid_fields": ReasonCategory.VALIDATION_ERROR,
    "invalid_business_arguments": ReasonCategory.VALIDATION_ERROR,
    "state_store_failed": ReasonCategory.SERVICE_ERROR,
    "intent_unclear": ReasonCategory.NEEDS_CLARIFICATION,
    "business_unsupported": ReasonCategory.UNSUPPORTED,
    "tool_unavailable": ReasonCategory.UNSUPPORTED,
    "business_started": ReasonCategory.SUCCESS,
    "business_continued": ReasonCategory.SUCCESS,
    "business_paused": ReasonCategory.SUCCESS,
    "business_resumed": ReasonCategory.SUCCESS,
    "dispatch_started": ReasonCategory.SUCCESS,
    "execution_completed": ReasonCategory.SUCCESS,
    "execution_failed": ReasonCategory.SERVICE_ERROR,
    "execution_timed_out": ReasonCategory.EXECUTION_UNKNOWN,
})


def reason_category(reason_code: str) -> ReasonCategory:
    try:
        return REASON_CODE_CATEGORIES[reason_code]
    except KeyError:
        _fail("unknown_reason_code")


__all__ = [
    "SCHEMA_VERSION", "DialogueContractError", "DialogueIntent", "BusinessType", "ExecutionMode",
    "IntentProposal", "parse_intent_proposal", "required_business_slots", "validate_business_slots",
    "SlotSource", "SlotBinding", "DecisionSource", "DialogueDecision", "validate_dialogue_decision",
    "PendingBusiness",
    "BusinessPhase", "BusinessFocus", "SessionBusinessState", "CommitTiming", "StateTransition",
    "empty_business_state", "validate_business_state", "validate_state_transition",
    "state_to_payload", "state_from_payload", "public_business_state", "ReasonCategory",
    "REASON_CODE_CATEGORIES", "reason_category",
]
