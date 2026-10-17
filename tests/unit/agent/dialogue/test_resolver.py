from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from browser_mcp.contracts import SEARCH_BILIBILI_TOOL
from knowledge_agent.agent.dialogue.contracts import (
    BusinessPhase,
    BusinessType,
    DialogueIntent,
    ExecutionMode,
)
from knowledge_agent.agent.dialogue.resolver import resolve_dialogue
from knowledge_agent.agent.intent import McpCapability


NOW = "2026-10-17T10:00:00+08:00"
CAPABILITIES = (
    McpCapability(SEARCH_BILIBILI_TOOL, "Search Bilibili", (("query", "topic"),)),
)


def _resolve(question: str, **overrides):
    arguments = {
        "session_id": "session-1",
        "state": None,
        "assessment_state": None,
        "history": (),
        "capabilities": CAPABILITIES,
        "remaining_model_calls": 3,
        "model_invoker": None,
        "updated_at": NOW,
        "turn_id": "turn-1",
    }
    arguments.update(overrides)
    return resolve_dialogue(question, **arguments)


def _pending_state():
    return _resolve("打开 B 站").state_transition.next_state


def test_stateless_complete_and_incomplete_requests_do_not_create_state() -> None:
    complete = _resolve(
        "解释 RAG",
        session_id=None,
        turn_id=None,
    )
    assert complete.decision.business_type is BusinessType.KNOWLEDGE
    assert complete.state_transition is None

    incomplete = _resolve(
        "打开 B 站",
        session_id=None,
        turn_id=None,
    )
    assert incomplete.decision.missing_slots == ("query",)
    assert incomplete.state_transition is None


def test_pending_bilibili_topic_is_bound_to_exact_current_input() -> None:
    result = _resolve("Python decorators", state=_pending_state(), turn_id="turn-2")

    assert result.decision.intent is DialogueIntent.CONTINUE_BUSINESS
    assert dict(result.decision.confirmed_slots) == {"query": "Python decorators"}
    assert result.decision.slot_bindings[0].raw_input == "Python decorators"
    assert result.decision.slot_bindings[0].source_turn_id == "turn-2"
    assert result.state_transition.next_state.focus.phase is BusinessPhase.READY


def test_greeting_preserves_pending_and_new_knowledge_replaces_it() -> None:
    state = _pending_state()

    greeting = _resolve("你好", state=state)
    assert greeting.decision.intent is DialogueIntent.CHITCHAT
    assert greeting.state_transition is None

    knowledge = _resolve("解释 RAG", state=state)
    assert knowledge.decision.business_type is BusinessType.KNOWLEDGE
    assert knowledge.state_transition.next_state.focus.business_type is BusinessType.KNOWLEDGE
    assert knowledge.state_transition.next_state.pending is None


def test_bare_cancel_cancels_the_only_pending_business() -> None:
    result = _resolve("取消", state=_pending_state(), turn_id="turn-2")

    assert result.decision.intent is DialogueIntent.CANCEL_BUSINESS
    assert result.state_transition.reason_code == "pending_cancelled"
    assert result.state_transition.next_state.focus.phase is BusinessPhase.CANCELLED


def test_explicit_multi_task_preserves_raw_request_for_planner() -> None:
    question = "先解释 RAG，然后介绍 Agent"
    result = _resolve(question)

    assert result.decision.execution_mode is ExecutionMode.PLAN
    assert result.decision.raw_plan_request == question


def test_classifier_uses_one_call_and_accepts_only_strict_contract() -> None:
    payload = {
        "intent": "chitchat",
        "business_type": None,
        "execution_mode": "single",
        "slots": {},
        "missing_slots": [],
    }
    result = _resolve(
        "这是什么",
        model_invoker=lambda _messages: AIMessage(content=json.dumps(payload)),
    )

    assert result.model_calls == 1
    assert result.decision.intent is DialogueIntent.CHITCHAT


def test_semantic_bilibili_candidate_requires_verbatim_model_topic() -> None:
    payload = {
        "intent": "new_business",
        "business_type": "bilibili_search",
        "execution_mode": "single",
        "slots": {"query": "数据结构"},
        "missing_slots": [],
    }
    result = _resolve(
        "帮我在B站找点数据结构的视频",
        model_invoker=lambda _messages: AIMessage(
            content=json.dumps(payload, ensure_ascii=False),
        ),
    )

    assert result.model_calls == 1
    assert dict(result.decision.confirmed_slots) == {"query": "数据结构"}
    assert result.decision.allowed_tool_names == (SEARCH_BILIBILI_TOOL,)
    assert result.state_transition.next_state.focus.phase is BusinessPhase.READY


def test_classifier_preserves_budget_for_downstream_work() -> None:
    called = False

    def invoke(_messages):
        nonlocal called
        called = True
        raise AssertionError("must not be called")

    result = _resolve(
        "这是什么",
        remaining_model_calls=1,
        model_invoker=invoke,
    )

    assert called is False
    assert result.model_calls == 0
    assert result.decision.reason_code == "model_call_budget_exhausted"
