from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pytest

from knowledge_agent import evaluation
from knowledge_agent.schemas import AnswerResult


def _questions(path: Path, count: int = 3) -> Path:
    records = [
        {
            "id": f"q{index}",
            "type": "answerable",
            "question": f"question-{index}",
            "expected_sources": [{"file": f"docs/{index}.md"}],
        }
        for index in range(1, count + 1)
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def _answer(question: str, status: str = "success") -> AnswerResult:
    return AnswerResult(
        question=question,
        answer=f"answer for {question}",
        status=status,  # type: ignore[arg-type]
        model_calls=1,
        tool_calls=2,
    )


class FakeApplication:
    def __init__(self, outcomes: list[object], *, fail_cleanup: bool = False) -> None:
        self.outcomes = list(outcomes)
        self.fail_cleanup = fail_cleanup
        self.events: list[tuple[str, str | None]] = []
        self.active = False
        self.closed = False

    def start_session(self, session_id: str) -> None:
        assert not self.active
        self.active = True
        self.events.append(("start", session_id))

    def ask(self, question: str) -> AnswerResult:
        assert self.active
        self.events.append(("ask", question))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, AnswerResult)
        return outcome

    def end_session(self) -> None:
        assert self.active
        self.active = False
        self.events.append(("end", None))
        if self.fail_cleanup:
            self.fail_cleanup = False
            raise RuntimeError("cleanup secret")

    def close(self) -> None:
        self.closed = True


def test_batch_isolated_ordered_and_continues_after_case_failures(tmp_path: Path) -> None:
    app = FakeApplication(
        [
            _answer("question-1", "service_error"),
            RuntimeError("API_KEY=secret-value"),
            _answer("question-3"),
        ]
    )

    result = evaluation.run_evaluation(
        _questions(tmp_path / "questions.jsonl"),
        tmp_path / "reports",
        application_factory=lambda: app,
    )

    assert [event[0] for event in app.events] == ["start", "ask", "end"] * 3
    assert [event[1] for event in app.events if event[0] == "ask"] == [
        "question-1", "question-2", "question-3"
    ]
    assert app.closed
    assert result.summary.run_status == "completed_with_errors"
    assert result.summary.completed_cases == 3
    assert result.summary.error_cases == 2
    assert result.summary.citation_expected == 3
    assert result.summary.citation_matched == 0
    assert len(result.results_path.read_text(encoding="utf-8").splitlines()) == 3
    all_output = result.results_path.read_text(encoding="utf-8") + result.report_path.read_text(encoding="utf-8")
    assert "secret-value" not in all_output


def test_cleanup_failure_discards_application_before_next_case(tmp_path: Path) -> None:
    first = FakeApplication([_answer("question-1")], fail_cleanup=True)
    second = FakeApplication([_answer("question-2")])
    applications = iter((first, second))

    result = evaluation.run_evaluation(
        _questions(tmp_path / "questions.jsonl", 2),
        tmp_path / "reports",
        application_factory=lambda: next(applications),
    )

    assert first.closed and second.closed
    assert result.summary.completed_cases == 2
    assert result.summary.error_cases == 1


def test_initialization_failure_is_a_failed_batch_with_no_fake_cases(tmp_path: Path) -> None:
    def fail() -> object:
        raise RuntimeError("CHAT_API_KEY=secret")

    result = evaluation.run_evaluation(
        _questions(tmp_path / "questions.jsonl"),
        tmp_path / "reports",
        application_factory=fail,
    )

    assert result.summary.run_status == "failed"
    assert result.summary.run_error == "application_initialization_failed"
    assert result.summary.completed_cases == 0
    assert result.summary.error_cases == 0
    assert result.results_path.read_text(encoding="utf-8") == ""
    assert "secret" not in result.summary_path.read_text(encoding="utf-8")


def test_keyboard_interrupt_stops_and_preserves_partial_results(tmp_path: Path) -> None:
    app = FakeApplication([_answer("question-1"), KeyboardInterrupt(), _answer("question-3")])

    result = evaluation.run_evaluation(
        _questions(tmp_path / "questions.jsonl"),
        tmp_path / "reports",
        application_factory=lambda: app,
    )

    records = [json.loads(line) for line in result.results_path.read_text(encoding="utf-8").splitlines()]
    assert result.summary.run_status == "interrupted"
    assert result.summary.completed_cases == 1
    assert len(records) == 2
    assert records[-1]["interrupted"] is True
    assert "question-3" not in [event[1] for event in app.events]


def test_results_write_failure_stops_before_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FakeApplication([_answer("question-1"), _answer("question-2")])

    def fail_write(stream: object, result: object) -> None:
        raise OSError("disk secret")

    monkeypatch.setattr(evaluation, "_write_case", fail_write)
    with pytest.raises(OSError):
        evaluation.run_evaluation(
            _questions(tmp_path / "questions.jsonl", 2),
            tmp_path / "reports",
            application_factory=lambda: app,
        )

    assert [event[1] for event in app.events if event[0] == "ask"] == ["question-1"]


def test_jsonable_handles_real_result_metadata_types() -> None:
    class State(Enum):
        DONE = "done"

    value = {
        "time": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "state": State.DONE,
        "tags": frozenset({"one"}),
    }

    assert evaluation._jsonable(value) == {
        "time": "2026-01-01T00:00:00+00:00",
        "state": "done",
        "tags": ["one"],
    }

