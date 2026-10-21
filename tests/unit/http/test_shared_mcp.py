"""共享 MCP 生命周期回归；不启动子进程或调用模型。"""
from types import SimpleNamespace

import pytest

from knowledge_agent.http.runtime import ServiceRuntime
from knowledge_agent.mcp.bootstrap import create_configured_application


@pytest.mark.parametrize("startup_fails", [False, True])
def test_sixteen_lanes_share_one_owner(monkeypatch, startup_fails):
    events, runtimes, borrowed = [], [], []

    class Store:
        def initialize(self):
            pass

        def close(self):
            pass

    class Mcp:
        def __init__(self, path):
            self.path = path
            runtimes.append(self)

        def start(self):
            events.append(("start", self.path))
            if startup_fails and self.path is not None:
                raise RuntimeError("startup timeout")

        def close(self):
            events.append(("mcp_close", self.path))

        def list_server_statuses(self):
            return []

    def factory(*, session_store, mcp_runtime):
        borrowed.append(mcp_runtime)
        return SimpleNamespace(close=lambda: events.append(("app_close", None)))

    monkeypatch.setenv("API_SESSION_CONCURRENCY", "16")
    monkeypatch.setattr("knowledge_agent.config.get_mcp_settings", lambda: SimpleNamespace(config_path="fake"))
    monkeypatch.setattr("knowledge_agent.mcp.client.McpRuntime", Mcp)
    monkeypatch.setattr("knowledge_agent.mcp.bootstrap.create_configured_application", factory)
    service = ServiceRuntime(store_factory=Store, concurrent_sessions=True)
    try:
        service.initialize()
        futures = [lane.submit(lambda lane=lane: lane.application) for lane in service._lanes]
        for future in futures:
            future.result(timeout=5)
        assert len(borrowed) == 16
        assert all(item is service._shared_mcp for item in borrowed)
        assert events.count(("start", "fake")) == 1
        assert service.concurrency_status()["mcp_startup_error"] == (
            "mcp_startup_failed" if startup_fails else None
        )
    finally:
        service.close()
    assert sum(event[0] == "app_close" for event in events) == 16
    assert events[-1] == ("mcp_close", None if startup_fails else "fake")
    assert events.count(events[-1]) == 1


def test_borrower_refreshes_tools_without_closing_owner(monkeypatch):
    closed = []
    tools = []
    owner = SimpleNamespace(
        list_tools=lambda: tools,
        call_tool=lambda name, arguments: {"name": name},
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr("knowledge_agent.application.create_agent_application", lambda **kwargs: kwargs)
    app = create_configured_application(mcp_runtime=owner)
    borrowed = app["owned_mcp_runtime"]
    tools.append("new tool")
    assert borrowed.list_tools() == ["new tool"]
    assert borrowed.call_tool("test", {}) == {"name": "test"}
    borrowed.close()
    assert closed == []
