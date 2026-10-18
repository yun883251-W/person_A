from __future__ import annotations

import json

import pytest

import knowledge_agent.skills.selector as skill_selector
from knowledge_agent.skills.selector import (
    SkillRequest,
    SkillSelectionSettings,
    load_selected_skill,
    normalize_skill_request,
    select_skill,
    snapshot_skill_catalog,
)


CATALOG = (
    {
        "skill_id": "concept-explanation",
        "name": "Concept explanation",
        "description": "Explain one concept, its mechanism, and its purpose.",
    },
    {
        "skill_id": "concept-comparison",
        "name": "Concept comparison",
        "description": "Compare two concepts using the same criteria.",
    },
    {
        "skill_id": "code-explanation",
        "name": "Code explanation",
        "description": "Explain visible code without reading local paths.",
    },
)


class FakeModel:
    def __init__(self, skill_id: str | None) -> None:
        self.skill_id = skill_id
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, messages):
        self.calls.append(tuple(messages))
        return {
            "content": json.dumps(
                {
                    "skill_id": self.skill_id,
                    "reason_code": "skill_match" if self.skill_id else "skill_no_match",
                }
            )
        }


def test_legacy_request_defaults_and_explicit_modes() -> None:
    assert normalize_skill_request() == SkillRequest("off")
    assert normalize_skill_request(skill_id="concept-explanation") == SkillRequest(
        "manual", "concept-explanation"
    )
    assert normalize_skill_request(" AUTO ") == SkillRequest("auto")
    assert normalize_skill_request("manual", " code-explanation ") == SkillRequest(
        "manual", "code-explanation"
    )

    for mode, skill_id in (("manual", None), ("auto", "concept-explanation"), ("off", "concept-explanation")):
        with pytest.raises(ValueError):
            normalize_skill_request(mode, skill_id)


def test_off_manual_and_rules_modes_do_not_call_the_model() -> None:
    model = FakeModel("concept-explanation")

    disabled = select_skill(
        "What is Python?", request=SkillRequest("off"), catalog=CATALOG,
        model_invoker=model, remaining_model_calls=4,
    )
    manual = select_skill(
        "What is Python?", request=SkillRequest("manual", "concept-comparison"),
        catalog=CATALOG, model_invoker=model, remaining_model_calls=4,
    )
    automatic = select_skill(
        "What is Python?", request=SkillRequest("auto"), catalog=CATALOG,
        model_invoker=model, remaining_model_calls=4,
    )

    assert (disabled.selected_skill_id, disabled.decision_source) == (None, "disabled")
    assert (manual.selected_skill_id, manual.decision_source) == (
        "concept-comparison", "manual"
    )
    assert (automatic.selected_skill_id, automatic.decision_source) == (
        "concept-explanation", "rules"
    )
    assert model.calls == []


def test_catalog_is_fixed_bounded_and_allowlisted() -> None:
    settings = SkillSelectionSettings(
        max_candidates=2,
        max_catalog_chars=160,
        allowed_skill_ids=("concept-explanation", "code-explanation"),
    )
    snapshot = snapshot_skill_catalog(CATALOG, settings=settings)

    assert [entry.skill_id for entry in snapshot] == [
        "concept-explanation", "code-explanation"
    ]
    assert all(not hasattr(entry, "path") for entry in snapshot)
    assert sum(len(entry.skill_id) + len(entry.name) + len(entry.description) for entry in snapshot) <= 160

    with pytest.raises(ValueError, match="duplicate"):
        snapshot_skill_catalog((CATALOG[0], CATALOG[0]))


def test_conflicting_rules_defer_once_to_the_fake_model() -> None:
    model = FakeModel("concept-comparison")
    result = select_skill(
        "Compare what these functions do:\n```python\nleft()\nright()\n```",
        request=SkillRequest("auto"),
        catalog=CATALOG,
        model_invoker=model,
        remaining_model_calls=2,
        settings=SkillSelectionSettings(mode="hybrid"),
    )

    assert (result.selected_skill_id, result.decision_source, result.selection_model_calls) == (
        "concept-comparison", "model", 1
    )
    assert len(model.calls) == 1
    payload = json.loads(model.calls[0][1].content)
    assert [item["skill_id"] for item in payload["skills"]] == [
        item["skill_id"] for item in CATALOG
    ]
    assert all(set(item) == {"skill_id", "name", "description"} for item in payload["skills"])


def test_hybrid_budget_and_failures_have_exact_model_call_counts() -> None:
    settings = SkillSelectionSettings(mode="hybrid")
    model = FakeModel(None)

    for remaining in (0, 1):
        result = select_skill(
            "Summarize the topic", request=SkillRequest("auto"), catalog=CATALOG,
            model_invoker=model, remaining_model_calls=remaining, settings=settings,
        )
        assert (result.reason_code, result.selection_model_calls) == (
            "skill_selector_budget_exhausted", 0
        )
    assert model.calls == []

    def timeout(_messages):
        raise TimeoutError

    result = select_skill(
        "Summarize the topic", request=SkillRequest("auto"), catalog=CATALOG,
        model_invoker=timeout, remaining_model_calls=2, settings=settings,
    )
    assert (result.decision_source, result.reason_code, result.selection_model_calls) == (
        "fallback", "skill_selector_timeout", 1
    )


def test_model_sees_public_metadata_only_and_body_loads_after_selection(monkeypatch) -> None:
    catalog = tuple(
        {
            **entry,
            "path": f"private/{entry['skill_id']}/SKILL.md",
            "body": f"private body for {entry['skill_id']}",
        }
        for entry in CATALOG
    )
    model = FakeModel("concept-explanation")
    loaded = object()
    load_calls: list[str] = []

    def fake_load(skill_id: str):
        load_calls.append(skill_id)
        return loaded

    monkeypatch.setattr(skill_selector, "load_skill", fake_load)
    selection = select_skill(
        "Summarize the topic", request=SkillRequest("auto"), catalog=catalog,
        model_invoker=model, remaining_model_calls=2,
        settings=SkillSelectionSettings(mode="hybrid"),
    )

    assert load_calls == []
    payload = json.loads(model.calls[0][1].content)
    assert all(set(item) == {"skill_id", "name", "description"} for item in payload["skills"])
    assert "private/" not in model.calls[0][1].content
    assert "private body" not in model.calls[0][1].content

    snapshot = load_selected_skill(selection)
    assert snapshot.loaded_skill is loaded
    assert load_calls == ["concept-explanation"]


def test_auto_load_failure_falls_back_but_manual_failure_is_not_hidden(monkeypatch) -> None:
    def fail_load(_skill_id: str):
        raise ValueError("invalid skill")

    monkeypatch.setattr(skill_selector, "load_skill", fail_load)
    automatic = load_selected_skill(
        skill_selector.SkillSelection(
            "auto", None, "concept-explanation", "rules", "skill_rule_match"
        )
    )
    assert automatic.loaded_skill is None
    assert automatic.strategy_id == "default"
    assert (
        automatic.selection.selected_skill_id,
        automatic.selection.decision_source,
        automatic.selection.reason_code,
    ) == (None, "fallback", "skill_load_failed")

    with pytest.raises(ValueError, match="invalid skill"):
        load_selected_skill(
            skill_selector.SkillSelection(
                "manual", "concept-explanation", "concept-explanation",
                "manual", "skill_manual",
            )
        )


@pytest.mark.parametrize(
    ("question", "visible_context", "expected"),
    (
        ("什么是向量数据库？", "", "concept-explanation"),
        ("What is a vector database?", "", "concept-explanation"),
        ("对比 Redis 与 Memcached 的差异", "", "concept-comparison"),
        ("Compare Redis vs. Memcached", "", "concept-comparison"),
        ("请解释这段代码：\n```python\nitems.sort()\n```", "", "code-explanation"),
        ("Explain this function", "def load():\n    return 1", "code-explanation"),
        ("不要解释代码，只给出结果：\n```python\n1 + 1\n```", "", None),
        ("Don't compare Redis and Memcached", "", None),
    ),
)
def test_chinese_and_english_rule_boundaries(question, visible_context, expected) -> None:
    result = select_skill(
        question,
        request=SkillRequest("auto"),
        catalog=CATALOG,
        visible_context=visible_context,
    )
    assert result.selected_skill_id == expected
    assert result.decision_source == ("rules" if expected else "fallback")
