"""把真实发布的工具 Schema 发给真实提供方，确认请求被接受。

这是唯一能挡住「本地全绿、上线 100% 不可用」的一层：Schema 是否被接受，只有
提供方说了算。B 站搜索曾因此在完整的开发与提交流程后完全失效，而所有离线用例
（包括同目录的保守子集检查）都是通过的。

默认不运行：必须显式设置 PERSON_A_LIVE_SCHEMA_PROBE=1。不设置时全部跳过，
因此 CI 和日常 `pytest` 都不联网、不读凭据、不消耗额度。

运行方式（项目根目录）：

    PERSON_A_LIVE_SCHEMA_PROBE=1 pytest tests/integration/mcp/test_live_provider_schema.py -v

本文件只**绑定**工具并发送一次请求，从不执行工具。浏览器工具因此不会打开任何
页面：只有工具被执行时才会动作，而这里连模型提出的 tool_calls 都直接丢弃。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

pytestmark = pytest.mark.live

# 一个不诱发工具调用的探针。目的只是让提供方校验 tools 数组，不是真的要用工具：
# 只要请求到达校验阶段，被拒的 Schema 就会以 400 结束，与模型是否选择调用无关。
_PROBE_PROMPT = "请只回复两个字：收到"


@pytest.fixture(scope="module")
def probe_model(require_live_probe: None) -> Any:
    """Return the production chat model under the operator's own configuration."""
    from knowledge_agent.models import initialize_chat_model

    return initialize_chat_model()


def _group_by_server(tools: Sequence[Any]) -> dict[str, list[Any]]:
    """Group tools the way the agent binds them: one server's tools per turn."""
    groups: dict[str, list[Any]] = {}
    for registration in tools:
        _, server, _ = registration.name.split("__", 2)
        groups.setdefault(server, []).append(registration.definition)
    return groups


def _probe(model: Any, tools: Sequence[Any], label: str) -> None:
    """Send one tool-binding request; fail with the provider's verdict."""
    from knowledge_agent.models import (
        ChatModelInvocationError,
        ChatModelRequestError,
        bind_chat_tools,
        invoke_chat_model,
    )

    bound = bind_chat_tools(model, tools)
    try:
        invoke_chat_model(bound, [HumanMessage(content=_PROBE_PROMPT)], max_retries=0)
    except ChatModelRequestError as exc:
        raise AssertionError(
            f"提供方拒绝了 {label} 的工具 Schema。"
            "这正是 B 站搜索失效的故障类别：本地校验全部通过，请求在提供方被拒。"
            f"原始错误: {exc}"
        ) from exc
    except ChatModelInvocationError as exc:
        raise AssertionError(
            f"{label} 的探测未能完成，结论无效（非 Schema 问题，请检查网络与提供方状态）。"
            f"原始错误: {exc}"
        ) from exc


def test_probe_control_request_without_tools_succeeds(probe_model: Any) -> None:
    """先确认不带工具的请求能通，再判断带工具的失败是否由 Schema 引起。"""
    from knowledge_agent.models import ChatModelInvocationError, invoke_chat_model

    try:
        invoke_chat_model(
            probe_model, [HumanMessage(content=_PROBE_PROMPT)], max_retries=0
        )
    except ChatModelInvocationError as exc:
        raise AssertionError(
            "不带工具的对照请求就失败了，说明配置或服务本身不可用，"
            f"本次探测无法判断 Schema：{exc}"
        ) from exc


def test_provider_accepts_each_published_tool_schema(
    probe_model: Any, published_mcp_tools: Sequence[Any]
) -> None:
    """逐个工具探测，失败时能直接指出是哪一个工具的 Schema 有问题。"""
    for registration in published_mcp_tools:
        _probe(probe_model, (registration.definition,), registration.name)


def test_provider_accepts_each_server_tool_group(
    probe_model: Any, published_mcp_tools: Sequence[Any]
) -> None:
    """按服务器分组探测，复现生产中的真实绑定方式。

    MBTI 的三个工具在同一轮里一起绑定；浏览器工具受单工具轮约束，自成一组。
    分组后仍可能被拒（例如名称或 `$defs` 冲突），这是逐个探测覆盖不到的情况。
    """
    for server, definitions in _group_by_server(published_mcp_tools).items():
        _probe(probe_model, definitions, f"MCP 服务器 {server} 的 {len(definitions)} 个工具")
