"""意图数据契约：供规则、受限分类器与会话路由共享，不执行识别或调用模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class IntentKind(str, Enum):
    """支持的意图类别。"""

    CHAT = "chat"
    KNOWLEDGE = "knowledge"
    GREETING = "greeting"
    SCOPE = "scope"
    NEEDS_CLARIFICATION = "needs_clarification"
    OUT_OF_SCOPE = "out_of_scope"
    MCP_TOOL = "mcp_tool"


@dataclass(frozen=True, slots=True)
class IntentDecision:
    """已校验、不可变的意图识别结果。"""

    kind: IntentKind
    retrieval_query: str = ""
    allowed_tool_names: tuple[str, ...] = ()
    decision_source: Literal["rules", "model", "fallback"] = "rules"
    clarification_question: str = ""
    reason_code: str = ""
    # 仅运行时的来源信息：浏览器意图校验后由程序从精确的当前输入复制，
    # 绝不由分类器 JSON 提供。
    authorized_user_input: str = ""
    # 已校验的语义搜索主题；这不授予浏览器授权。
    search_topic: str = ""
    # None表示打开搜索页；整数表示读取视频列表，已经宿主校验。
    search_limit: int | None = None


@dataclass(frozen=True, slots=True)
class McpCapability:
    """可用 MCP 工具的公开、不可执行视图。"""

    name: str
    description: str
    required_inputs: tuple[tuple[str, str], ...] = ()


class IntentModelBudgetError(RuntimeError):
    """表示所属轮次已无剩余模型调用。"""
