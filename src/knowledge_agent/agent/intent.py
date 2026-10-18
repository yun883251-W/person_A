"""旧意图接口的兼容导出；识别实现统一放在 agent.dialogue。

intent_types 定义契约，rules 只执行确定性规则，classifier 统一控制可选模型识别。
旧调用方可继续导入 identify_intent；包含业务焦点与待补参数的入口使用
dialogue.resolver，在规则识别和模型补充之间处理会话状态。
"""

from .dialogue.classifier import classify_rule_intent, identify_intent
from .dialogue.intent_types import (
    IntentDecision,
    IntentKind,
    IntentModelBudgetError,
    McpCapability,
)
from .dialogue.rules import identify_intent_rules, snapshot_mcp_capabilities

__all__ = [
    "IntentDecision",
    "IntentKind",
    "IntentModelBudgetError",
    "McpCapability",
    "classify_rule_intent",
    "identify_intent",
    "identify_intent_rules",
    "snapshot_mcp_capabilities",
]
