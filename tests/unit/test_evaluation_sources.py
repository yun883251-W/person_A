from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_agent.evaluation import (
    EvaluationCaseResult,
    EvaluationExpectedSource,
    EvaluationInputError,
    EvaluationQuestion,
    _build_summary,
    check_citations,
    load_questions,
    run_evaluation,
)
from knowledge_agent.schemas import AnswerSource


def _write_jsonl(path: Path, records: list[object]) -> Path:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _valid_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "q1",
        "type": "answerable",
        "question": "Question?",
        "expected_sources": [{"file": "docs/chapter.md", "section": "Parent / Child"}],
        "expected_points": [],
        "acceptance_criteria": [],
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    "contents",
    ["", "not-json\n", '{"id":"q1"}\n'],
)
def test_load_questions_rejects_empty_malformed_and_missing_fields(
    tmp_path: Path, contents: str,
) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(EvaluationInputError):
        load_questions(path)


def test_load_questions_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "questions.jsonl", [_valid_record(), _valid_record()])

    with pytest.raises(EvaluationInputError, match="unique"):
        load_questions(path)


@pytest.mark.parametrize("source", ["../chapter.md", "/chapter.md", "C:\\chapter.md"])
def test_load_questions_rejects_unsafe_source_paths(tmp_path: Path, source: str) -> None:
    record = _valid_record(expected_sources=[{"file": source}])
    path = _write_jsonl(tmp_path / "questions.jsonl", [record])

    with pytest.raises(EvaluationInputError, match="relative path"):
        load_questions(path)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_run_evaluation_rejects_invalid_limit_before_reading_questions(limit: object) -> None:
    with pytest.raises(EvaluationInputError, match="positive integer"):
        run_evaluation("missing.jsonl", limit=limit)  # type: ignore[arg-type]


def _source(path: str, headings: tuple[str, ...], chunk_id: str = "chunk") -> AnswerSource:
    return AnswerSource(
        chunk_id=chunk_id,
        source_path=path,
        heading_path=headings,
        start_line=1,
        end_line=2,
        content_hash="hash",
        index_version="index",
    )


def test_citation_matching_uses_full_path_and_contiguous_heading_hierarchy() -> None:
    question = EvaluationQuestion(
        id="q1",
        type="answerable",
        question="Question?",
        expected_sources=(
            EvaluationExpectedSource("docs/chapter.md", "Parent / Child"),
            EvaluationExpectedSource("other/chapter.md", "Parent"),
        ),
    )
    duplicate = _source("docs\\chapter.md", ("Document", "Parent", "Child", "Detail"))

    checks = check_citations(
        question,
        (
            _source("wrong/chapter.md", ("Parent", "Child"), "wrong-file"),
            _source("docs/chapter.md", ("Parent", "Other", "Child"), "wrong-section"),
            duplicate,
            duplicate,
        ),
    )

    assert [check.matched for check in checks] == [True, False]
    assert checks[0].actual_source is duplicate
    assert sum(check.matched for check in checks) == 1
    assert not any(check.matched for check in check_citations(question, ()))


def test_only_unanswerable_role_source_is_optional() -> None:
    expected = (EvaluationExpectedSource("tutorial.md", role="teaching example"),)
    unanswerable = EvaluationQuestion("q1", "unanswerable", "Question?", expected)
    answerable = EvaluationQuestion("q2", "answerable", "Question?", expected)

    assert check_citations(unanswerable, ())[0].applicable is False
    assert check_citations(answerable, ())[0].applicable is True


def test_zero_applicable_sources_has_no_coverage_score(tmp_path: Path) -> None:
    question = EvaluationQuestion(
        "q1",
        "unanswerable",
        "Question?",
        (EvaluationExpectedSource("tutorial.md", role="teaching example"),),
    )
    case = EvaluationCaseResult(
        question=question,
        answer_result=None,
        elapsed_seconds=0.0,
        citation_checks=check_citations(question, ()),
    )

    summary = _build_summary(
        (case,),
        total_cases=1,
        elapsed_seconds=0.0,
        interrupted=False,
        run_error=None,
        started_at="2026-01-01T00:00:00+00:00",
        questions_path=tmp_path / "questions.jsonl",
        questions_sha256="hash",
        question_bank_cases=1,
        limit=None,
        system_prompt_sha256=None,
    )

    assert summary.citation_expected == 0
    assert summary.citation_matched == 0
    assert summary.citation_coverage is None
