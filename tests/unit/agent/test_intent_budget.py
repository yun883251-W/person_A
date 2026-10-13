from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from knowledge_agent import config
from knowledge_agent.agent import model_loop, preparation, runner
from knowledge_agent.agent.trace import TraceRecorder
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


def _knowledge_decision() -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {
                "kind": "knowledge",
                "allowed_tool_names": [],
                "clarification_question": "",
            }
        )
    )


def _intent_event(result):
    assert result.trace is not None
    assert all(event.stage != "classify" for event in result.trace.events)
    events = [event for event in result.trace.events if event.stage == "intent"]
    assert len(events) == 1
    return events[0]


def test_rules_greeting_uses_no_model_or_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(config, "get_intent_settings", lambda: IntentSettings(mode="rules"))
    monkeypatch.setattr(
        runner, "invoke_chat_model", lambda *args, **kwargs: pytest.fail("unexpected model call")
    )
    monkeypatch.setattr(
        preparation, "invoke_chat_model", lambda *args, **kwargs: pytest.fail("unexpected model call")
    )
    monkeypatch.setattr(
        model_loop, "invoke_chat_model", lambda *args, **kwargs: pytest.fail("unexpected model call")
    )
    retriever = StubRetriever()
    result = runner.KnowledgeAgent(model=object(), retriever=retriever).ask("hello")

    assert result.status == "success"
    assert result.model_calls == 0
    assert retriever.queries == []
    event = _intent_event(result)
    assert (event.decision_source, event.intent_kind, event.status) == (
        "rules", "greeting", "success"
    )
    assert event.reason_code == "greeting_input"
    assert event.error_code is None


def test_intent_exhausting_budget_stops_before_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(config, "get_intent_settings", lambda: IntentSettings(mode="hybrid"))
    model = object()
    calls = []

    def invoke(actual_model, messages, **kwargs):
        calls.append(actual_model)
        return _knowledge_decision()

    monkeypatch.setattr(runner, "invoke_chat_model", invoke)
    monkeypatch.setattr(preparation, "invoke_chat_model", invoke)
    monkeypatch.setattr(model_loop, "invoke_chat_model", invoke)
    retriever = StubRetriever()
    agent = runner.KnowledgeAgent(
        AgentConfig(max_model_calls=1), model=model, retriever=retriever
    )
    result = agent.ask("Explain Python lists")

    assert result.status == "limit_reached"
    assert result.stop_reason == "model_call_budget_exhausted"
    assert result.model_calls == 1
    assert calls == [model]
    assert retriever.queries == []
    event = _intent_event(result)
    assert (event.decision_source, event.intent_kind, event.reason_code) == (
        "model", "knowledge", "hybrid_knowledge"
    )


@pytest.mark.parametrize("session_id", [None, "test-session"])
def test_intent_and_answer_share_budget_and_classify_once(monkeypatch, session_id) -> None:
    monkeypatch.setattr(config, "get_intent_settings", lambda: IntentSettings(mode="hybrid"))
    model, bound_model = object(), object()
    calls = []

    def invoke(actual_model, messages, **kwargs):
        calls.append(actual_model)
        if actual_model is model:
            return _knowledge_decision()
        assert actual_model is bound_model
        return AIMessage(content="Python lists preserve insertion order. [S1]")

    monkeypatch.setattr(runner, "invoke_chat_model", invoke)
    monkeypatch.setattr(preparation, "invoke_chat_model", invoke)
    monkeypatch.setattr(model_loop, "invoke_chat_model", invoke)
    monkeypatch.setattr(preparation, "bind_chat_tools", lambda model, tools: bound_model)
    retriever = StubRetriever()
    agent = runner.KnowledgeAgent(
        AgentConfig(max_model_calls=2), model=model, retriever=retriever
    )
    result = agent.ask("Explain Python lists", session_id=session_id)

    assert result.status == "success"
    assert result.model_calls == 2
    assert calls == [model, bound_model]
    assert retriever.queries == ["Explain Python lists"]
    assert _intent_event(result).decision_source == "model"
    assert sum(event.stage == "model" for event in result.trace.events) == 1


def test_failed_intent_counts_once_and_records_only_safe_error_code(monkeypatch) -> None:
    monkeypatch.setattr(config, "get_intent_settings", lambda: IntentSettings(mode="hybrid"))
    calls = []

    def invoke(model, messages, **kwargs):
        calls.append(model)
        raise RuntimeError("private-provider-detail api_key=secret")

    monkeypatch.setattr(runner, "invoke_chat_model", invoke)
    monkeypatch.setattr(preparation, "invoke_chat_model", invoke)
    monkeypatch.setattr(model_loop, "invoke_chat_model", invoke)
    retriever = StubRetriever()
    agent = runner.KnowledgeAgent(
        AgentConfig(max_model_calls=1), model=object(), retriever=retriever
    )
    result = agent.ask("Explain Python lists")

    assert result.model_calls == 1
    assert len(calls) == 1
    assert result.status == "limit_reached"
    assert retriever.queries == []
    event = _intent_event(result)
    assert event.status == "error"
    assert event.decision_source == "fallback"
    assert event.reason_code == "intent_model_error"
    assert event.error_code == "intent_model_error"
    assert "private-provider-detail" not in repr(result.trace)
    assert "secret" not in repr(result.trace)


def test_zero_budget_rejects_classifier_before_invocation(monkeypatch) -> None:
    monkeypatch.setattr(config, "get_intent_settings", lambda: IntentSettings(mode="hybrid"))
    monkeypatch.setattr(
        preparation, "invoke_chat_model", lambda *args, **kwargs: pytest.fail("unexpected model call")
    )
    retriever = StubRetriever()
    recorder = TraceRecorder()
    prepared = preparation.prepare_question(
        "Explain Python lists",
        retriever=retriever,
        max_context_chars=4000,
        model=object(),
        max_model_calls=0,
        trace_recorder=recorder,
    )

    assert prepared.terminal_result is not None
    assert prepared.terminal_result.status == "limit_reached"
    assert prepared.terminal_result.stop_reason == "model_call_budget_exhausted"
    assert prepared.state.model_calls == 0
    assert retriever.queries == []


def test_classifier_disables_transport_retries_and_counts_failed_call(monkeypatch) -> None:
    monkeypatch.setattr(config, "get_intent_settings", lambda: IntentSettings(mode="hybrid"))

    class RetryingModel:
        _knowledge_agent_max_retries = 1

        def __init__(self) -> None:
            self.attempts = 0

        def invoke(self, messages):
            self.attempts += 1
            if self.attempts == 1:
                raise TimeoutError("temporary timeout")
            return _knowledge_decision()

    model = RetryingModel()
    retriever = StubRetriever()
    agent = runner.KnowledgeAgent(
        AgentConfig(max_model_calls=1), model=model, retriever=retriever
    )
    result = agent.ask("Explain Python lists")

    assert model.attempts == 1
    assert result.model_calls == 1
    assert result.status == "limit_reached"
    assert retriever.queries == []
    event = _intent_event(result)
    assert event.decision_source == "fallback"
    assert event.status == "error"
    assert event.reason_code == "intent_model_timeout"
    assert event.error_code == "intent_model_timeout"
