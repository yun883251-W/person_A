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


def test_single_call_budget_skips_classifier_and_stops_before_retrieval(monkeypatch) -> None:
    """预算只够一次调用时不为分类消耗它：分类不发起，直接以预算耗尽结束。

    分类需要为自己之外再留一次调用（`max_model_calls - 1`）。预算为 1 时这个条件
    永远不成立，因此既没有分类调用，也不会检索 —— 一次注定无法完成的轮次不再
    白花一次提供方调用。预算为 2 时分类正常发起，见
    `test_intent_and_answer_share_budget_and_classify_once`。
    """
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
    assert result.model_calls == 0
    assert calls == []
    assert retriever.queries == []
    event = _intent_event(result)
    assert (event.decision_source, event.intent_kind, event.reason_code) == (
        "fallback", "knowledge", "model_call_budget_exhausted"
    )
    assert event.status == "error"
    assert event.error_code == "model_call_budget_exhausted"


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
    """分类失败只计一次调用、降级为 fallback，且提供方细节不进入公开追踪。

    分类失败不终止本轮：判定降级后仍照常检索和回答，所以预算要够分类加回答两次。
    """
    monkeypatch.setattr(config, "get_intent_settings", lambda: IntentSettings(mode="hybrid"))
    model, bound_model = object(), object()
    calls = []

    def invoke(actual_model, messages, **kwargs):
        calls.append(actual_model)
        if actual_model is bound_model:
            return AIMessage(content="Python lists preserve insertion order. [S1]")
        raise RuntimeError("private-provider-detail api_key=secret")

    monkeypatch.setattr(runner, "invoke_chat_model", invoke)
    monkeypatch.setattr(preparation, "invoke_chat_model", invoke)
    monkeypatch.setattr(model_loop, "invoke_chat_model", invoke)
    monkeypatch.setattr(preparation, "bind_chat_tools", lambda model, tools: bound_model)
    retriever = StubRetriever()
    agent = runner.KnowledgeAgent(
        AgentConfig(max_model_calls=2), model=model, retriever=retriever
    )
    result = agent.ask("Explain Python lists")

    assert result.status == "success"
    assert result.model_calls == 2
    assert calls == [model, bound_model]
    assert retriever.queries == ["Explain Python lists"]
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
    """分类关闭传输重试：配置允许多次尝试，分类仍只发起一次并计入预算。

    这里不替换 invoke_chat_model，走真实的调用与错误归一化；模型自身允许两次
    传输尝试，如果分类没有关闭重试，attempts 会是 2。
    """
    monkeypatch.setattr(config, "get_intent_settings", lambda: IntentSettings(mode="hybrid"))

    class TimeoutModel:
        _knowledge_agent_max_retries = 1

        def __init__(self) -> None:
            self.attempts = 0

        def invoke(self, messages):
            self.attempts += 1
            raise TimeoutError("temporary timeout")

    class BoundAnswerModel:
        def invoke(self, messages):
            return AIMessage(content="Python lists preserve insertion order. [S1]")

    model = TimeoutModel()
    monkeypatch.setattr(
        preparation, "bind_chat_tools", lambda bound, tools: BoundAnswerModel()
    )
    retriever = StubRetriever()
    agent = runner.KnowledgeAgent(
        AgentConfig(max_model_calls=2), model=model, retriever=retriever
    )
    result = agent.ask("Explain Python lists")

    assert model.attempts == 1
    assert result.model_calls == 2
    assert result.status == "success"
    assert retriever.queries == ["Explain Python lists"]
    event = _intent_event(result)
    assert event.decision_source == "fallback"
    assert event.status == "error"
    assert event.reason_code == "intent_model_timeout"
    assert event.error_code == "intent_model_timeout"
