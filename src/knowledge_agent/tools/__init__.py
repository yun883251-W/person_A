"""Agent 工具模块（待开发）。

模块职责：提供工具注册、统一执行及知识库工具的公开入口。
registry.py 管理工具定义与调用；knowledge.py 封装知识库搜索和小节读取。

当前只有文件骨架与 TODO，agent/runner.py 中的工具逻辑仍然是实际运行入口。
开发顺序：registry.py → knowledge.py → skill_loader.py → Agent/API/页面接入。
本阶段不迁移旧代码、不初始化模型或数据库，验证留到开发完成后统一进行。
"""


from .knowledge import KnowledgeToolDependencies, create_knowledge_tool_registry
from .registry import (
    InMemoryToolRegistry,
    ToolCallResult,
    ToolRegistration,
    ToolRegistry,
    create_tool_registry,
)

__all__ = (
    "InMemoryToolRegistry",
    "KnowledgeToolDependencies",
    "ToolCallResult",
    "ToolRegistration",
    "ToolRegistry",
    "create_knowledge_tool_registry",
    "create_tool_registry",
)
