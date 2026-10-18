from __future__ import annotations

from langchain_core.messages import AIMessage

from knowledge_agent import config
from knowledge_agent.agent import model_loop, preparation, runner
from knowledge_agent.agent.assessment import TOOLS as MBTI_TOOLS
from knowledge_agent.agent.intent import IntentDecision, IntentKind
from knowledge_agent.skills.selector import SkillSelectionSettings
from knowledge_agent.config import AgentConfig, IntentSettings
from knowledge_agent.schemas import DocumentChunk, RetrievalResult


class StubRetriever:
    index_version = "test-index"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[RetrievalResult]:
        self.queries.append(query)
        return [
            RetrievalResult(
                chunk_id="test-chunk",
                chunk=DocumentChunk(
                    text="Python lists preserve insertion order.",
                    source_path="python.md",
                    heading_path=("Lists",),
                    start_line=1,
                    end_line=1,
                    content_hash="test-hash",
                    chunk_id="test-chunk",
                ),
                channels=("keyword",),
                fused_score=1.0,
                index_version=self.index_version,
            )
        ]


def _install_answer_model(monkeypatch):
    bound_model = object()

    def invoke(actual_model, _messages, **_kwargs):
        assert actual_model is bound_model
        return AIMessage(content="Python lists preserve insertion order. [S1]")

    monkeypatch.setattr(preparation, "bind_chat_tools", lambda _model, _tools: bound_model)
    monkeypatch.setattr(model_loop, "invoke_chat_model", invoke)


def test_single_and_planned_knowledge_entries_run_skill_selection(monkeypatch) -> None:
    monkeypatch.setattr(config, "get_intent_settings", lambda: IntentSettings(mode="rules"))
    _install_answer_model(monkeypatch)
    calls: list[str] = []
    original = preparation.select_skill_for_step

    def record_selection(question, **kwargs):
        calls.append(question)
        return original(question, **kwargs)

    monkeypatch.setattr(preparation, "select_skill_for_step", record_selection)
    retriever = StubRetriever()
    agent = runner.KnowledgeAgent(
        AgentConfig(max_model_calls=4),
        model=object(),
        retriever=retriever,
        skill_selection_settings=SkillSelectionSettings(mode="rules"),
    )

    single = agent.ask("What is a Python list?", skill_mode="auto")
    assert single.status == "success"
    assert calls == ["What is a Python list?"]

    calls.clear()
    planned = agent.ask(
        "1. Explain Python lists. 2. Explain Python tuples.", skill_mode="auto"
    )
    assert planned.status == "success"
    assert calls == ["Explain Python lists.", "Explain Python tuples."]
    skill_events = [event for event in planned.trace.events if event.stage == "skill_select"]
    assert len(skill_events) == 2
    assert all(event.step_id is not None for event in skill_events)


def test_greeting_and_mbti_routes_do_not_run_skill_selection(monkeypatch) -> None:
    monkeypatch.setattr(config, "get_intent_settings", lambda: IntentSettings(mode="rules"))
    monkeypatch.setattr(
        preparation,
        "select_skill_for_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-knowledge route reached skill selection")
        ),
    )
    agent = runner.KnowledgeAgent(model=object(), retriever=StubRetriever())

    greeting = agent.ask("hello", skill_mode="auto")
    assert greeting.status == "success"
    assert all(event.stage != "skill_select" for event in greeting.trace.events)

    agent._mcp_capabilities = lambda: tuple(
        {"name": name, "description": "", "input_schema": {}} for name in MBTI_TOOLS
    )
    mbti = agent.ask("开始 MBTI 测试", skill_mode="auto")
    assert mbti.stop_reason == "mbti_session_required"
    assert all(event.stage != "skill_select" for event in mbti.trace.events)


def test_mcp_route_bypasses_knowledge_and_skill_preparation(monkeypatch) -> None:
    tool_name = "mcp__test__action"
    decision = IntentDecision(
        kind=IntentKind.MCP_TOOL,
        allowed_tool_names=(tool_name,),
        reason_code="test_mcp_action",
        authorized_user_input="run test action",
    )
    monkeypatch.setattr(runner, "identify_intent", lambda *_args, **_kwargs: decision)
    monkeypatch.setattr(
        preparation,
        "select_skill_for_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("MCP route reached skill selection")
        ),
    )

    class ReachedMcpPreparation(RuntimeError):
        pass

    monkeypatch.setattr(
        runner,
        "prepare_mcp_model_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ReachedMcpPreparation),
    )
    agent = runner.KnowledgeAgent(model=object(), retriever=StubRetriever())
    agent._mcp_capabilities = lambda: (
        {"name": tool_name, "description": "", "input_schema": {}},
    )

    try:
        agent.ask("run test action", skill_mode="auto")
    except ReachedMcpPreparation:
        pass
    else:
        raise AssertionError("MCP route did not reach MCP preparation")
