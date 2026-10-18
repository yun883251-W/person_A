from __future__ import annotations

from types import SimpleNamespace

from browser_mcp.contracts import (
    SEARCH_BILIBILI_TOOL, build_search_url, validate_search_arguments,
)

from knowledge_agent.agent.dialogue.contracts import BusinessPhase
from knowledge_agent.agent.runner import KnowledgeAgent
from knowledge_agent.session_store import SQLiteSessionStore


class _Definition:
    name = SEARCH_BILIBILI_TOOL
    description = "Search Bilibili"
    args_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }


class _Registry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def list_definitions(self):
        return (_Definition(),)

    def get(self, name):
        if name != SEARCH_BILIBILI_TOOL:
            return None
        return SimpleNamespace(
            definition=_Definition(), argument_validator=validate_search_arguments,
        )

    def execute(self, name, arguments, *, context):
        self.calls.append((name, dict(arguments)))
        query = arguments["query"]
        return {
            "status": "success", "tool_name": name, "server_name": "browser",
            "structured_content": {
                "status": "success", "action": "browser_launch_requested",
                "site": "bilibili", "query": query, "url": build_search_url(query),
                "browser_launch_requested": True, "page_verified": False,
            },
        }


class _UnusedRetriever:
    pass


def test_pending_bilibili_request_is_consumed_once_before_dispatch(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "chat.db")
    store.initialize()
    session_id = store.create_session("dialogue-session").session_id
    registry = _Registry()
    agent = KnowledgeAgent(
        model=object(), retriever=_UnusedRetriever(),
        session_store=store, mcp_registry=registry,
    )

    clarification = agent.ask("帮我在B站找点视频", session_id=session_id)
    assert clarification.status == "needs_clarification"
    assert clarification.business_state is not None
    assert clarification.business_state.waiting_for_input is True

    completed = agent.ask("火影忍者", session_id=session_id)
    assert completed.status == "success"
    assert registry.calls == [(SEARCH_BILIBILI_TOOL, {"query": "火影忍者"})]
    assert completed.stop_reason == "browser_launch_requested"
    assert store.load_business_state(session_id).focus.phase is BusinessPhase.COMPLETED

    restored = KnowledgeAgent(
        model=object(), retriever=_UnusedRetriever(),
        session_store=store, mcp_registry=registry,
    )
    result = restored.ask("继续", session_id=session_id)
    assert result.status == "needs_clarification"
    assert len(registry.calls) == 1


def test_cancel_pending_business_is_persisted_with_the_turn(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "chat.db")
    store.initialize()
    session_id = store.create_session("cancel-session").session_id
    agent = KnowledgeAgent(
        model=object(), retriever=_UnusedRetriever(),
        session_store=store, mcp_registry=_Registry(),
    )

    agent.ask("在 B 站搜索", session_id=session_id)
    result = agent.ask("取消", session_id=session_id)

    assert result.status == "success"
    assert result.business_state is not None
    assert result.business_state.phase == "cancelled"
    assert store.load_business_state(session_id).focus.phase is BusinessPhase.CANCELLED
    assert len(store.load_turns(session_id)) == 2
