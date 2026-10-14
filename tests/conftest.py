"""跨目录共享的测试装置。

`published_mcp_tools` 启动一次真实的 MCP 运行时，返回该项目实际发布给模型的
工具注册项。Schema 契约用例与在线提供方探测用例共用它，因此两者验证的是同一
个发布面，而不是各自构造的替身。

启动服务器不会打开浏览器，也不会调用任何工具：浏览器工具只在被调用时才动作。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 在线探测必须显式开启。默认不联网、不读取凭据、不消耗提供方额度。
LIVE_PROBE_ENV_VAR = "PERSON_A_LIVE_SCHEMA_PROBE"


def live_probe_enabled() -> bool:
    """Return True only when the operator explicitly opted into live probing."""
    return os.environ.get(LIVE_PROBE_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


@pytest.fixture(scope="session")
def require_live_probe() -> None:
    """Gate live tests on an explicit opt-in and a usable chat configuration.

    未开启时跳过。已开启却读不到配置时**报错而非跳过**：操作者明确要求在线探测，
    这时静默通过会把「没跑」伪装成「通过」。
    """
    if not live_probe_enabled():
        pytest.skip(f"未设置 {LIVE_PROBE_ENV_VAR}=1，跳过在线探测")

    from knowledge_agent.config import get_chat_settings

    try:
        get_chat_settings()
    except ValueError as exc:
        pytest.fail(f"已开启在线探测，但对话模型配置不可用: {exc}")
    # 本地 Ollama 可以没有 API Key，因此这里不强制；凭据问题会由对照用例明确报出。


@pytest.fixture(scope="session")
def published_mcp_tools():
    """Return one real tool registration per tool the shipped config publishes."""
    pytest.importorskip("mcp", reason="需要 mcp 可选依赖")
    pytest.importorskip("jsonschema", reason="需要 mcp 可选依赖")

    from knowledge_agent.config import get_mcp_settings
    from knowledge_agent.mcp.adapter import create_mcp_tool_registry
    from knowledge_agent.mcp.client import McpRuntime

    settings = get_mcp_settings()
    if settings.config_path is None:
        pytest.skip("未配置 MCP_CONFIG_PATH，没有可验证的发布面")

    runtime = McpRuntime(settings.config_path)
    try:
        runtime.start()
        if not runtime.list_tools():
            # 配置启用了 MCP 却一个工具都没发布，属于静默失败，必须报错而非跳过。
            statuses = [
                f"{item.server_name}: {item.state.value}"
                + (f" ({item.error_code})" if item.error_code else "")
                for item in runtime.list_server_statuses()
            ]
            pytest.fail(f"启用的 MCP 服务器没有发布任何工具；状态: {statuses}")
        registry = create_mcp_tool_registry(runtime)
        yield registry.list_registrations()
    finally:
        runtime.close()
