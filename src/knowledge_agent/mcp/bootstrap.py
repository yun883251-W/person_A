"""Create the shared CLI/API application and own its optional MCP runtime."""

from __future__ import annotations

from typing import Any


def create_configured_application(*, session_store: Any = None):
    from ..application import create_agent_application
    from ..config import get_mcp_settings
    from .client import McpRuntime

    runtime = McpRuntime(get_mcp_settings().config_path)
    try:
        runtime.start()
        # A disabled/empty runtime must not import optional JSON Schema packages.
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
