"""创建 CLI/API 共享应用，并持有可选的 MCP 运行时。"""

from __future__ import annotations

from typing import Any


class BorrowedMcpRuntime:
    """应用借用服务级 MCP；保留工具刷新能力，但不取得关闭权。"""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def list_tools(self):
        return self.runtime.list_tools()

    def call_tool(self, name, arguments):
        return self.runtime.call_tool(name, arguments)

    def close(self) -> None:
        pass


def create_configured_application(*, session_store: Any = None, mcp_runtime: Any = None):
    from ..application import create_agent_application
    from ..config import get_mcp_settings
    from .client import McpRuntime

    runtime = (
        McpRuntime(get_mcp_settings().config_path)
        if mcp_runtime is None else BorrowedMcpRuntime(mcp_runtime)
    )
    try:
        if mcp_runtime is None:
            runtime.start()
        # 已禁用/空的运行时不得导入可选的 JSON Schema 包。
        from ..tools.registry import InMemoryToolRegistry
        registry = InMemoryToolRegistry()
        if runtime.list_tools():
            from .adapter import create_mcp_tool_registry
            registry = create_mcp_tool_registry(runtime)
        return create_agent_application(
            session_store=session_store, mcp_registry=registry,
            owned_mcp_runtime=runtime,
        )
    except BaseException:
        runtime.close()
        raise
