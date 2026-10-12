"""意图路由与 Agent 分支衔接。

模块功能：根据已经校验的 IntentDecision 和当前能力，决定本轮执行分支。
输入：意图、显式 Skill 选择、可用工具快照及单轮上下文。
输出：RouteDecision，供 KnowledgeAgent.ask() 执行知识问答、MCP 工具或直接回复。

路由节点不注册为工具，不自行调用识别模型，也不负责启动 MCP 子进程。
保留 knowledge、mcp、direct 三种分支；MBTI 会话操作属于受控的 MCP 分支。
知识分支保留检索与 [S1] 引用协议；MCP 分支使用独立提示、工具权限和结果校验。

已接入 agent/runner.py；验证 TODO 10—12 留待统一执行。
"""


from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from collections.abc import Sequence

from .intent import IntentDecision, IntentKind, snapshot_mcp_capabilities


class RouteKind(str, Enum):
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
    IntentKind.KNOWLEDGE: RouteKind.KNOWLEDGE,
    IntentKind.MCP_TOOL: RouteKind.MCP,
    IntentKind.GREETING: RouteKind.DIRECT,
    IntentKind.SCOPE: RouteKind.DIRECT,
    IntentKind.NEEDS_CLARIFICATION: RouteKind.DIRECT,
    IntentKind.OUT_OF_SCOPE: RouteKind.DIRECT,
}


def resolve_route(
    intent: IntentDecision,
    available_tools: Sequence[object] = (),
    skill_id: str | None = None,
    *,
    assessment_action: str | None = None,
) -> RouteDecision:
    """Route an already validated intent without classifying it again."""
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


# 第一阶段：路由数据与分支选择。


# 第二阶段：在 agent/runner.py 衔接分支，保持现有生命周期。

# 第三阶段：开发完成后统一验证，本阶段不执行。

# TODO 10：验证知识分支兼容性及分支隔离。
# 覆盖原有建库后问答、手动 Skill、多轮知识追问、引用、资料不足和旧会话恢复。
# 覆盖 MCP 路由在索引缺失时仍可调用工具；知识路线不因 MCP 未安装或未启用而失败。
# 确认普通知识问题不会携带外部工具定义，MCP 结果不会污染本轮知识来源。

# TODO 11：验证工具循环、预算与异常的结果约束。
# 覆盖未调用工具就宣称完成、未知工具、参数缺失、工具错误、重复 ID、超时和超长结果。
# 覆盖意图模型已消耗预算后进入工具循环、同轮多个工具调用和成功失败混合返回。
# 检查 AI/Tool 消息完整配对、计数准确、失败不误存成成功、trace 与实际路径一致。

# TODO 12：最后集中验证 CLI/API/页面全链路。
# 用演示 MCP 工具走完提问、识别、路由、调用、回答、保存、恢复历史、关闭服务。
# 覆盖断连后读取历史、关闭时在途请求、工具分支切回知识分支和多会话隔离。
# 所有验证完成后再标记已接入/可用，不把 TODO 文件存在等同于功能已经实现。
