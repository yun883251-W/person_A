"""确定性意图规则：处理历史、工具匹配、浏览器边界与 MBTI 操作。

此层不读取模型模式、不调用模型、不执行工具或写入会话。
dialogue.resolver 可以先使用规则结果处理待补参数，再交给 classifier 决定是否补充识别。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from .intent_types import IntentDecision, IntentKind, McpCapability


_GREETING_INPUTS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "你好",
        "您好",
        "早上好",
        "下午好",
        "晚上好",
    }
)
_SCOPE_INPUTS = frozenset(
    {
        "你能做什么",
        "你会什么",
        "你能回答什么问题",
        "你的范围是什么",
        "what can you do",
        "what is your scope",
    }
)
_VAGUE_INPUTS = frozenset(
    {
        "这是什么",
        "这个怎么用",
        "它怎么用",
        "怎么用",
        "请解释一下",
        "what is this",
        "how do i use it",
        "explain this",
    }
)
_OUT_OF_SCOPE_MARKERS = (
    "天气",
    "股票",
    "星座",
    "菜谱",
    "体育比分",
    "医疗诊断",
    "weather",
    "stock price",
    "horoscope",
    "sports score",
    "medical diagnosis",
)
_IN_SCOPE_MARKERS = (
    "python",
    "langchain",
    "langgraph",
    "rag",
    "agent",
    "代码",
    "编程",
    "api",
)
_QUESTION_TRAILING_PUNCTUATION = " \t\r\n?!？！。.,，~～"


def _normalized_intent_text(question: str) -> str:
    return question.strip().rstrip(_QUESTION_TRAILING_PUNCTUATION).strip().casefold()


def _identify_current_intent(question: str) -> IntentDecision:
    """Classify a question using the deterministic legacy intent rules."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be blank")

    normalized = _normalized_intent_text(cleaned)
    if normalized in _GREETING_INPUTS:
        return IntentDecision(kind=IntentKind.GREETING, reason_code="greeting_input")
    if normalized in _SCOPE_INPUTS:
        return IntentDecision(kind=IntentKind.SCOPE, reason_code="scope_input")
    if normalized in _VAGUE_INPUTS:
        return IntentDecision(
            kind=IntentKind.NEEDS_CLARIFICATION,
            clarification_question="请说明你指的具体概念、代码或功能。",
            reason_code="vague_input",
        )
    if (
        any(marker in normalized for marker in _OUT_OF_SCOPE_MARKERS)
        and not any(marker in normalized for marker in _IN_SCOPE_MARKERS)
    ):
        return IntentDecision(
            kind=IntentKind.OUT_OF_SCOPE,
            reason_code="out_of_scope_marker",
        )
    return IntentDecision(
        kind=IntentKind.KNOWLEDGE,
        retrieval_query=cleaned,
        reason_code="default_knowledge",
    )


_FOLLOWUP_REFERENCE_RE = re.compile(
    r"(?:这(?:个|些|段|种)?|那(?:个|些|段|种)?|它|上述|上面|前面|刚才|继续|"
    r"其(?:中|他)?|\b(?:this|that|it|these|those|above|previous|continue)\b)",
    re.IGNORECASE,
)
_EXPLICIT_TOPIC_SWITCH_RE = re.compile(
    r"(?:换个话题|换一个问题|另外(?:一个)?问题|题外话|"
    r"\b(?:new topic|another question|unrelated question)\b)",
    re.IGNORECASE,
)
_UNSUPPORTED_STATEFUL_FOLLOWUP_RE = re.compile(
    r"^(?:我?选\s*[A-Da-d]|下一题|下一个|继续答题)$"
)
_DEFAULT_MAX_HISTORY_TURNS = 6
_DEFAULT_MAX_CONTEXT_CHARS = 4000
_MCP_TOOL_NAME_RE = re.compile(
    r"^mcp__(?P<server>[A-Za-z0-9_-]+)__(?P<tool>[A-Za-z0-9_-]+)$"
)
_EXPLICIT_TOOL_REQUEST_RE = re.compile(
    r"(?:调用|执行|运行|请用|使用工具|用\s*(?:mcp__)?[A-Za-z0-9_-]+\s*(?:工具)?(?:来|帮)|"
    r"\b(?:call|run|execute|invoke)\b)",
    re.IGNORECASE,
)
_EXPLICIT_TOOL_TARGET_RE = re.compile(
    r"(?:使用工具|(?:调用|执行|运行|请用)\s*(?:MCP\s*)?(?:工具\s*)?"
    r"[A-Za-z][A-Za-z0-9_-]*|"
    r"\b(?:call|run|execute|invoke)\s+(?:the\s+)?(?:MCP\s+)?(?:tool\s+)?"
    r"[A-Za-z][A-Za-z0-9_-]*)",
    re.IGNORECASE,
)
_TOOL_INSTRUCTION_RE = re.compile(
    r"(?:怎么用|如何使用|怎样使用|用法|是什么|介绍|教程|"
    r"\b(?:how\s+(?:do\s+i\s+|to\s+)?use|what\s+is|explain|tutorial)\b)",
    re.IGNORECASE,
)


def _tool_value(tool: object, name: str) -> object | None:
    if isinstance(tool, Mapping):
        return tool.get(name)
    return getattr(tool, name, None)


def _required_inputs(input_schema: object) -> tuple[tuple[str, str], ...]:
    """Extract only required field names and their public descriptions."""
    if not isinstance(input_schema, Mapping):
        return ()
    required = input_schema.get("required", ())
    properties = input_schema.get("properties", {})
    if (
        isinstance(required, (str, bytes))
        or not isinstance(required, Sequence)
        or not isinstance(properties, Mapping)
    ):
        return ()

    result: list[tuple[str, str]] = []
    for field_name in required:
        if not isinstance(field_name, str) or not field_name:
            continue
        field_schema = properties.get(field_name)
        description = ""
        if isinstance(field_schema, Mapping):
            value = field_schema.get("description")
            if isinstance(value, str):
                description = value.strip()
        result.append((field_name, description))
    return tuple(result)


def snapshot_mcp_capabilities(
    available_tools: Sequence[object],
) -> tuple[McpCapability, ...]:
    """Copy the safe public subset of an authorized, ready MCP tool snapshot."""
    if isinstance(available_tools, (str, bytes)) or not isinstance(
        available_tools, Sequence
    ):
        raise TypeError("available_tools must be a sequence")

    capabilities: list[McpCapability] = []
    seen_names: set[str] = set()
    duplicate_names: set[str] = set()
    for tool in available_tools:
        name = _tool_value(tool, "name")
        description = _tool_value(tool, "description")
        input_schema = _tool_value(tool, "input_schema")
        if (
            not isinstance(name, str)
            or len(name) > 64
            or _MCP_TOOL_NAME_RE.fullmatch(name) is None
        ):
            continue
        normalized_name = name.casefold()
        if normalized_name in seen_names:
            duplicate_names.add(normalized_name)
            continue
        seen_names.add(normalized_name)
        capabilities.append(
            McpCapability(
                name=name,
                description=description.strip() if isinstance(description, str) else "",
                required_inputs=_required_inputs(input_schema),
            )
        )
    if duplicate_names:
        capabilities = [
            item for item in capabilities if item.name.casefold() not in duplicate_names
        ]
    return tuple(capabilities)


def _requested_mcp_tools(
    question: str,
    capabilities: Sequence[McpCapability],
) -> tuple[str, ...]:
    """Resolve an explicit request only when its tool name is unambiguous."""
    if (
        _EXPLICIT_TOOL_REQUEST_RE.search(question) is None
        or _TOOL_INSTRUCTION_RE.search(question) is not None
    ):
        return ()

    normalized = question.casefold()
    full_matches = tuple(
        capability.name
        for capability in capabilities
        if re.search(
            rf"(?<![\w-]){re.escape(capability.name.casefold())}(?![\w-])",
            normalized,
        )
    )
    if full_matches:
        return full_matches

    aliases: dict[str, list[str]] = {}
    for capability in capabilities:
        match = _MCP_TOOL_NAME_RE.fullmatch(capability.name)
        if match is not None:
            aliases.setdefault(match.group("tool").casefold(), []).append(capability.name)
    matched_names: list[str] = []
    for alias, names in aliases.items():
        if len(names) == 1 and re.search(
            rf"(?<![\w-]){re.escape(alias)}(?![\w-])", normalized
        ):
            matched_names.extend(names)
    return tuple(dict.fromkeys(matched_names))


def _history_value(turn: object, name: str) -> object | None:
    if isinstance(turn, Mapping):
        return turn.get(name)
    return getattr(turn, name, None)


def _knowledge_history_questions(
    context: Sequence[object],
    *,
    max_history_turns: int,
    max_context_chars: int,
) -> tuple[str, ...]:
    """Return bounded questions from complete knowledge-route turns only."""
    selected: list[str] = []
    remaining = max_context_chars
    for turn in reversed(context[-max_history_turns:]):
        if _history_value(turn, "complete") is False:
            continue
        route = _history_value(turn, "route")
        kind = _history_value(turn, "intent_kind")
        if kind is None:
            kind = _history_value(turn, "kind")
        if isinstance(route, Enum):
            route = route.value
        if isinstance(kind, Enum):
            kind = kind.value
        if route not in (None, "knowledge") or kind not in (None, "knowledge"):
            continue

        question = _history_value(turn, "question")
        if (
            route is None
            and kind is None
            and isinstance(question, str)
            and question.strip()
            and _identify_current_intent(question).kind is not IntentKind.KNOWLEDGE
        ):
            continue
        retrieval_query = _history_value(turn, "retrieval_query")
        if retrieval_query is None:
            retrieval_query = _history_value(turn, "retrieval_question")
        candidate = retrieval_query or question
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        candidate = candidate.strip()
        if remaining <= 0:
            break
        if len(candidate) > remaining:
            candidate = candidate[:remaining].rstrip()
        if candidate:
            selected.append(candidate)
            remaining -= len(candidate)
    selected.reverse()
    return tuple(selected)


def _identify_intent_with_rules(
    question: str,
    *,
    context: Sequence[object] = (),
    available_tools: Sequence[object] = (),
    max_history_turns: int = _DEFAULT_MAX_HISTORY_TURNS,
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
) -> IntentDecision:
    """Classify a request and resolve bounded, referential knowledge follow-ups."""
    current = _identify_current_intent(question)
    capabilities = snapshot_mcp_capabilities(available_tools)
    if isinstance(max_history_turns, bool) or not isinstance(max_history_turns, int):
        raise TypeError("max_history_turns must be an integer")
    if max_history_turns <= 0:
        raise ValueError("max_history_turns must be positive")
    if isinstance(max_context_chars, bool) or not isinstance(max_context_chars, int):
        raise TypeError("max_context_chars must be an integer")
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive")
    if isinstance(context, (str, bytes)) or not isinstance(context, Sequence):
        raise TypeError("context must be a sequence of committed turns")

    cleaned = question.strip()
    normalized = _normalized_intent_text(cleaned)
    from ..browser_actions import SEARCH_BILIBILI_TOOL, classify_bilibili_request

    browser = classify_bilibili_request(cleaned)
    if browser is not None:
        if browser.kind == "search":
            if SEARCH_BILIBILI_TOOL not in {item.name for item in capabilities}:
                return IntentDecision(
                    kind=IntentKind.NEEDS_CLARIFICATION,
                    clarification_question=(
                        "B 站浏览器搜索工具当前未启用或不可用。"
                        "请启用 browser MCP 服务后重试，或明确改为知识问题。"
                    ),
                    reason_code="browser_bilibili_unavailable",
                )
            return IntentDecision(
                kind=IntentKind.MCP_TOOL,
                allowed_tool_names=(SEARCH_BILIBILI_TOOL,),
                reason_code=browser.reason_code,
                authorized_user_input=cleaned,
                search_topic=browser.topic,
            )
        if browser.kind == "needs_clarification":
            return IntentDecision(
                kind=IntentKind.NEEDS_CLARIFICATION,
                clarification_question=browser.clarification_question,
                reason_code=browser.reason_code,
            )
        if browser.kind == "semantic_candidate":
            return IntentDecision(
                kind=IntentKind.NEEDS_CLARIFICATION,
                clarification_question="请明确要在 B 站搜索的主题，例如：请在 B 站搜索 Python 装饰器。",
                reason_code=browser.reason_code,
            )
        return IntentDecision(
            kind=IntentKind.KNOWLEDGE,
            retrieval_query=cleaned,
            reason_code=browser.reason_code,
        )
    requested_tools = _requested_mcp_tools(cleaned, capabilities)
    if requested_tools:
        return IntentDecision(
            kind=IntentKind.MCP_TOOL,
            allowed_tool_names=requested_tools,
            reason_code="explicit_mcp_tool_request",
        )
    if (
        _EXPLICIT_TOOL_TARGET_RE.search(cleaned) is not None
        and _TOOL_INSTRUCTION_RE.search(cleaned) is None
    ):
        return IntentDecision(
            kind=IntentKind.NEEDS_CLARIFICATION,
            clarification_question=(
                "当前没有可用的 MCP 工具，请改为知识问题或稍后重试。"
                if not capabilities
                else "请明确指定一个当前可用且名称唯一的 MCP 工具。"
            ),
            reason_code=(
                "mcp_tools_unavailable" if not capabilities else "mcp_tool_not_matched"
            ),
        )
    if _UNSUPPORTED_STATEFUL_FOLLOWUP_RE.fullmatch(normalized):
        return IntentDecision(
            kind=IntentKind.NEEDS_CLARIFICATION,
            clarification_question="当前没有可继续的答题状态，请直接说明你想问的内容。",
            reason_code="unsupported_stateful_followup",
        )

    # A direct intent or an explicit topic change belongs entirely to this turn.
    if current.kind not in {IntentKind.KNOWLEDGE, IntentKind.NEEDS_CLARIFICATION}:
        return current
    if _EXPLICIT_TOPIC_SWITCH_RE.search(cleaned):
        return IntentDecision(
            kind=IntentKind.KNOWLEDGE,
            retrieval_query=cleaned,
            reason_code="explicit_topic_switch",
        )

    is_referential = (
        current.kind is IntentKind.NEEDS_CLARIFICATION
        or _FOLLOWUP_REFERENCE_RE.search(cleaned) is not None
    )
    if not is_referential:
        return current

    history_questions = _knowledge_history_questions(
        context,
        max_history_turns=max_history_turns,
        max_context_chars=max_context_chars,
    )
    if not history_questions:
        return IntentDecision(
            kind=IntentKind.NEEDS_CLARIFICATION,
            clarification_question="请说明你指的具体概念、代码或功能。",
            reason_code="followup_without_context",
        )

    history_text = "；".join(history_questions)
    retrieval_query = f"先前问题：{history_text}\n当前追问：{cleaned}"
    return IntentDecision(
        kind=IntentKind.KNOWLEDGE,
        retrieval_query=retrieval_query,
        reason_code="knowledge_followup",
    )


def identify_intent_rules(
    question: str,
    *,
    context: Sequence[object] = (),
    available_tools: Sequence[object] = (),
    assessment_state: Mapping[str, Any] | None = None,
    skill_id: str | None = None,
    max_history_turns: int = _DEFAULT_MAX_HISTORY_TURNS,
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
) -> IntentDecision:
    """Resolve deterministic intent, without spending a model call."""
    _validate_manual_skill_selection(skill_id)
    rules_decision = _identify_intent_with_rules(
        question,
        context=context,
        available_tools=available_tools,
        max_history_turns=max_history_turns,
        max_context_chars=max_context_chars,
    )
    if rules_decision.reason_code.startswith("browser_bilibili_"):
        # Browser requests take precedence over the broader MBTI teaching guard.
        # Semantic candidates remain unresolved here for pending-slot handling.
        return rules_decision

    from ..assessment import (
        TOOLS,
        is_tool_instruction_question,
        request_action,
        request_clarification,
    )

    clarification = request_clarification(question, assessment_state)
    if clarification is not None:
        return IntentDecision(
            kind=IntentKind.NEEDS_CLARIFICATION,
            clarification_question=clarification,
            reason_code="assessment_request_boundary",
        )
    action = request_action(question, assessment_state)
    if action is not None:
        return IntentDecision(
            kind=IntentKind.MCP_TOOL,
            allowed_tool_names=TOOLS,
            reason_code=f"mbti_{action}",
        )
    if is_tool_instruction_question(question):
        # Keep tutorials as knowledge and retain resolved follow-up context.
        return IntentDecision(
            kind=IntentKind.KNOWLEDGE,
            retrieval_query=rules_decision.retrieval_query or question.strip(),
            reason_code="tool_instruction_question",
        )
    return rules_decision


def _validate_manual_skill_selection(skill_id: str | None) -> None:
    """Validate an explicitly selected Skill without applying or auto-selecting one."""
    if skill_id is None:
        return

    from ...skill_loader import load_skill

    load_skill(skill_id)
