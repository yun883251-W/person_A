"""意图数据契约：供规则、受限分类器与会话路由共享，不执行识别或调用模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class IntentKind(str, Enum):
    """Supported intent categories."""

    KNOWLEDGE = "knowledge"
    GREETING = "greeting"
    SCOPE = "scope"
    NEEDS_CLARIFICATION = "needs_clarification"
    OUT_OF_SCOPE = "out_of_scope"
    MCP_TOOL = "mcp_tool"


@dataclass(frozen=True, slots=True)
class IntentDecision:
    """Validated, immutable result of intent identification."""

    kind: IntentKind
    retrieval_query: str = ""
    allowed_tool_names: tuple[str, ...] = ()
    decision_source: Literal["rules", "model", "fallback"] = "rules"
    clarification_question: str = ""
    reason_code: str = ""
    # Runtime-only provenance: copied by program from the exact current input
    # after browser intent validation, never supplied by classifier JSON.
    authorized_user_input: str = ""
    # Validated semantic search topic; this does not grant browser authorization.
    search_topic: str = ""


@dataclass(frozen=True, slots=True)
class McpCapability:
    """Public, non-executable view of an available MCP tool."""

    name: str
    description: str
    required_inputs: tuple[tuple[str, str], ...] = ()


class IntentModelBudgetError(RuntimeError):
    """Signal that the owning turn has no model calls remaining."""
