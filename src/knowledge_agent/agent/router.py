"""意图路由与 Agent 分支衔接。

模块功能：根据已经校验的 IntentDecision 和当前能力，决定本轮执行分支。
输入：意图、显式 Skill 选择、可用工具快照及单轮上下文。
输出：RouteDecision，供 KnowledgeAgent.ask() 执行知识问答、MCP 工具或直接回复。

路由节点不注册为工具，不自行调用识别模型，也不负责启动 MCP 子进程。
保留 knowledge、mcp、direct 三种分支；MBTI 会话操作属于受控的 MCP 分支。
知识分支保留检索与 [S1] 引用协议；MCP 分支使用独立提示、工具权限和结果校验。

已接入 agent/runner.py；文件末尾保留待统一执行的验证项。
"""


from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from collections.abc import Sequence

from browser_mcp.contracts import SEARCH_BILIBILI_TOOL

from .dialogue.intent_types import IntentDecision, IntentKind
from .dialogue.rules import snapshot_mcp_capabilities
from .dialogue.contracts import (
    BusinessType, DialogueDecision, DialogueIntent, validate_dialogue_decision,
)


class RouteKind(str, Enum):
    CHAT = "chat"
    KNOWLEDGE = "knowledge"
    MCP = "mcp"
    DIRECT = "direct"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: RouteKind
    intent: IntentDecision
    allowed_tool_names: tuple[str, ...] = ()
    reason_code: str = ""


_ROUTE_BY_INTENT = {
    IntentKind.CHAT: RouteKind.CHAT,
    IntentKind.KNOWLEDGE: RouteKind.KNOWLEDGE,
    IntentKind.MCP_TOOL: RouteKind.MCP,
    IntentKind.GREETING: RouteKind.DIRECT,
    IntentKind.SCOPE: RouteKind.DIRECT,
    IntentKind.NEEDS_CLARIFICATION: RouteKind.DIRECT,
    IntentKind.OUT_OF_SCOPE: RouteKind.DIRECT,
}


def intent_from_dialogue(decision: DialogueDecision, *, current_input: str) -> IntentDecision:
    """把最终对话决定投影为稳定的执行器契约。

    此边界不会再次分类，也不参考过时的决定。
    现有历史/API 路由枚举与存储轮次保持兼容。
    跨轮浏览器派发会校验其独立的来源绑定。
    """
    validate_dialogue_decision(decision)
    source = decision.decision_source.value
    if source == "program":
        source = "rules"
    common = {"decision_source": source, "reason_code": decision.reason_code}
    if decision.intent is DialogueIntent.UNCLEAR or decision.missing_slots:
        return IntentDecision(
            IntentKind.NEEDS_CLARIFICATION,
            clarification_question=decision.clarification_question,
            **common,
        )
    if decision.intent is DialogueIntent.UNSUPPORTED:
        return IntentDecision(IntentKind.OUT_OF_SCOPE, **common)
    if decision.intent is DialogueIntent.CHITCHAT:
        kind = IntentKind.SCOPE if decision.direct_response_kind == "scope" else IntentKind.GREETING
        return IntentDecision(kind, **common)
    if decision.business_type is BusinessType.KNOWLEDGE:
        return IntentDecision(
            IntentKind.KNOWLEDGE,
            retrieval_query=decision.retrieval_query or current_input.strip(), **common,
        )
    if decision.business_type in {BusinessType.BILIBILI_SEARCH, BusinessType.MCP_TOOL, BusinessType.MBTI}:
        from .assessment import TOOLS

        return IntentDecision(
            IntentKind.MCP_TOOL,
            allowed_tool_names=TOOLS if decision.business_type is BusinessType.MBTI else decision.allowed_tool_names,
            authorized_user_input=decision.authorized_user_input,
            search_topic=decision.confirmed_slots.get("query", "") if decision.business_type is BusinessType.BILIBILI_SEARCH else "",
            search_limit=decision.confirmed_slots.get("limit") if decision.business_type is BusinessType.BILIBILI_SEARCH else None,
            **common,
        )
    return IntentDecision(
        IntentKind.NEEDS_CLARIFICATION,
        clarification_question=decision.clarification_question or "请说明要完成的任务。",
        **common,
    )
def resolve_route(
    intent: IntentDecision,
    available_tools: Sequence[object] = (),
    skill_id: str | None = None,
    *,
    assessment_action: str | None = None,
) -> RouteDecision:
    """路由一个已校验的意图，而不再次分类。"""
    if not isinstance(intent, IntentDecision):
        raise TypeError("intent must be an IntentDecision")
    if skill_id is not None and not isinstance(skill_id, str):
        raise TypeError("skill_id must be a string or None")

    if skill_id is not None and intent.kind is IntentKind.MCP_TOOL:
        intent = IntentDecision(
            kind=IntentKind.NEEDS_CLARIFICATION,
            decision_source=intent.decision_source,
            clarification_question=(
                "当前选择的 Skill 只能用于知识问答。请取消 Skill 后重新请求工具操作，"
                "或保留 Skill 并将问题改为知识问答。"
            ),
            reason_code="manual_skill_mcp_conflict",
        )

    if (
        intent.kind is IntentKind.MCP_TOOL
        and SEARCH_BILIBILI_TOOL in intent.allowed_tool_names
        and intent.allowed_tool_names != (SEARCH_BILIBILI_TOOL,)
    ):
        intent = IntentDecision(
            kind=IntentKind.NEEDS_CLARIFICATION,
            decision_source=intent.decision_source,
            clarification_question="请将 B 站搜索单独发送，并明确本次搜索主题。",
            reason_code="browser_bilibili_single_action_only",
        )

    route = _ROUTE_BY_INTENT[intent.kind]
    allowed_tool_names: tuple[str, ...] = ()
    reason_code = intent.reason_code or f"{route.value}_intent"

    if route is RouteKind.MCP:
        available_names = {
            capability.name for capability in snapshot_mcp_capabilities(available_tools)
        }
        requested_names = intent.allowed_tool_names
        from .assessment import ACTIONS, TOOLS

        if (
            assessment_action in ACTIONS
            and intent.reason_code == f"mbti_{assessment_action}"
            and requested_names == TOOLS
        ):
            # 暂停、退出等只改会话状态；实际远程调用仍必须通过可用工具白名单。
            allowed_tool_names = tuple(name for name in TOOLS if name in available_names)
        elif requested_names and all(name in available_names for name in requested_names):
            allowed_tool_names = requested_names
        else:
            route = RouteKind.DIRECT
            reason_code = "mcp_capability_unavailable"
            intent = IntentDecision(
                kind=IntentKind.NEEDS_CLARIFICATION,
                decision_source=intent.decision_source,
                clarification_question=(
                    "当前请求的工具能力不可用。请确认相关服务已启用后重试，"
                    "或改为知识问答。"
                ),
                reason_code=reason_code,
            )

    return RouteDecision(
        route,
        intent,
        allowed_tool_names,
        reason_code,
    )
