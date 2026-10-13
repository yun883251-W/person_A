"""Agent 工具注册与知识库工具的公共接口。

模块职责：提供工具注册、统一执行及知识库工具的公开入口。
registry.py 管理工具定义与调用；knowledge.py 封装知识库搜索和小节读取。

agent/preparation.py 组装本轮可见工具，agent/model_loop.py 调用注册器执行工具。
Skill 和 MCP 均通过注册器限制本轮可用能力，不在包导入时初始化模型或数据库。
"""


from .knowledge import KnowledgeToolDependencies, create_knowledge_tool_registry
from .registry import (
    ArgumentValidator,
    InMemoryToolRegistry,
    ToolCallResult,
    ToolRegistration,
    ToolRegistry,
    combine_tool_registries,
    create_tool_registry,
)

__all__ = (
    "ArgumentValidator",
    "InMemoryToolRegistry",
    "KnowledgeToolDependencies",
    "ToolCallResult",
    "ToolRegistration",
    "ToolRegistry",
    "combine_tool_registries",
    "create_knowledge_tool_registry",
    "create_tool_registry",
)
