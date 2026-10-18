from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from knowledge_agent.agent import assessment, preparation
from knowledge_agent.agent.dialogue.intent_types import IntentKind
from knowledge_agent.agent.dialogue.classifier import identify_intent
from knowledge_agent.agent.state import ConversationHistory, ConversationTurn
from knowledge_agent.config import IntentSettings
from knowledge_agent.schemas import AnswerResult, RouteSummary


RULES = IntentSettings(mode="rules", browser_mode="rules")


def _turn(question: str, *, route: str = "knowledge") -> ConversationTurn:
    result = AnswerResult(
        question=question,
        answer="answer",
        status="success",
        routing=RouteSummary(
            route=route,
            intent="knowledge" if route == "knowledge" else "mcp_tool",
            decision_source="rules",
            reason_code="test",
        ),
    )
    return ConversationTurn(
        question=question,
        retrieval_question=question,
        messages=(HumanMessage(content=question), AIMessage(content="answer")),
        result=result,
    )


def test_followups_use_only_bounded_knowledge_history() -> None:
    context = [
        {"question": "oldest", "route": "knowledge", "complete": True},
        {"question": "tool action", "route": "mcp", "complete": True},
        {"question": "Python lists", "route": "knowledge", "complete": True},
    ]
    followup = identify_intent(
        "continue with that",
        settings=RULES,
        context=context,
        max_history_turns=2,
        max_context_chars=100,
    )
    switched = identify_intent(
        "new topic: explain tuples", settings=RULES, context=context
    )
    no_history = identify_intent("continue", settings=RULES)

    assert followup.kind is IntentKind.KNOWLEDGE
    assert "Python lists" in followup.retrieval_query
    assert "tool action" not in followup.retrieval_query
    assert "oldest" not in followup.retrieval_query
    assert switched.reason_code == "explicit_topic_switch"
    assert no_history.kind is IntentKind.NEEDS_CLARIFICATION


def test_tool_followup_is_not_replayed_from_history() -> None:
    decision = identify_intent(
        "continue",
        settings=RULES,
        context=[{"question": "call lookup", "route": "mcp", "complete": True}],
    )
    assert decision.kind is IntentKind.NEEDS_CLARIFICATION
    assert decision.allowed_tool_names == ()


def test_conversation_history_isolated_trimmed_and_legacy_compatible() -> None:
    history = ConversationHistory(max_turns=2)
    history.append("a", _turn("one"))
    history.append("b", _turn("other"))
    history.append("a", _turn("two"))
    history.append("a", _turn("three"))

    assert [turn.question for turn in history.turns("a")] == ["two", "three"]
    assert [turn.question for turn in history.turns("b")] == ["other"]
    assert [message.content for message in history.messages("a")] == [
        "two", "answer", "three", "answer"
    ]


def test_paused_number_is_not_submitted_and_completed_type_can_be_described() -> None:
    page = {
        "total": 48,
        "questionnaire_version": "q1",
        "scoring_version": "s1",
        "questions": [{"id": i, "text": f"Q{i}"} for i in range(12)],
        "options": [{"value": i, "label": str(i)} for i in range(1, 6)],
    }
    started = assessment.handle_request(
        "start", "start MBTI", None, lambda _name, _arguments: page
    )
    paused = assessment.handle_request(
        "pause", "pause", started.state, lambda *_args: {}
    )
    unchanged = assessment.handle_request(
        "answer", "3", paused.state, lambda *_args: {}
    )
    assert unchanged.stop_reason == "mbti_not_active"
    assert unchanged.state["answers"] == {}

    completed = dict(unchanged.state)
    completed["status"] = "completed"
    completed["answers"] = {str(i): 3 for i in range(48)}
    completed["questions"] = [{"id": i, "text": f"Q{i}"} for i in range(48)]
    completed["result"] = {
        "type": "INTJ",
        "type_name": "INTJ",
        "questionnaire_version": "q1",
        "scoring_version": "s1",
        "dimension_scores": {
            key: {left + "_pct": 50, right + "_pct": 50, "result": left}
            for key, (left, right) in {
                "EI": ("E", "I"), "SN": ("S", "N"),
                "TF": ("T", "F"), "JP": ("J", "P"),
            }.items()
        },
        "ties": {key: True for key in ("EI", "SN", "TF", "JP")},
    }
    described = assessment.handle_request(
        "describe",
        "介绍我的类型",
        completed,
        lambda name, arguments: {
            "description": {
                "name": arguments["type_code"], "traits": ["trait"],
                "strengths": [], "growth_areas": [],
            }
        },
    )
    assert described.stop_reason == "mbti_description"
    assert "INTJ" in described.answer


def test_rules_and_legacy_entry_do_not_invoke_optional_classifier(monkeypatch) -> None:
    calls = 0

    def count(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return identify_intent("hello", settings=RULES)

    monkeypatch.setattr(preparation, "identify_intent", count)
    assert preparation.classify_question("hello") == "greeting"
    assert calls == 1


def test_manual_skill_keeps_normal_knowledge_semantics() -> None:
    decision = identify_intent(
        "Explain dependency injection",
        skill_id="concept-explanation",
        settings=RULES,
        model_invoker=lambda _messages: (_ for _ in ()).throw(
            AssertionError("rules mode invoked the optional classifier")
        ),
    )
    assert decision.kind is IntentKind.KNOWLEDGE
    assert decision.retrieval_query == "Explain dependency injection"
