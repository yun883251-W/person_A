from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_agent import cli, evaluation
from knowledge_agent.agent.preparation import _SOURCE_CITATION_INSTRUCTION
from knowledge_agent.agent.prompts import SYSTEM_PROMPT


def _result(tmp_path: Path, status: str, *, error_cases: int = 0, run_error: str | None = None):
    summary = evaluation.EvaluationSummary(
        run_status=status,  # type: ignore[arg-type]
        run_error=run_error,
        interrupted=status == "interrupted",
        total_cases=3,
        completed_cases=2,
        error_cases=error_cases,
    )
    return evaluation.EvaluationRunResult(
        tmp_path / "run",
        tmp_path / "run" / "results.jsonl",
        tmp_path / "run" / "summary.json",
        tmp_path / "run" / "report.md",
        summary,
    )


def test_evaluate_dispatches_arguments_and_emits_clean_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object):
        calls.append(kwargs)
        return _result(tmp_path, "completed")

    monkeypatch.setattr(evaluation, "run_evaluation", fake_run)
    exit_code = cli.main([
        "evaluate", "--questions", "evaluation/custom.jsonl",
        "--output-dir", "evaluation/output", "--limit", "2", "--json",
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert calls == [{
        "questions_path": "evaluation/custom.jsonl",
        "output_dir": "evaluation/output",
        "limit": 2,
        "progress": None,
    }]
    assert payload["quality_review"] == "pending"
    assert payload["paths"]["report"].endswith("report.md")


@pytest.mark.parametrize(
    ("status", "error_cases", "run_error", "expected"),
    [
        ("completed", 0, None, 0),
        ("completed_with_errors", 1, None, 1),
        ("failed", 0, "application_initialization_failed", 1),
        ("interrupted", 0, None, 130),
    ],
)
def test_evaluate_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    error_cases: int,
    run_error: str | None,
    expected: int,
) -> None:
    monkeypatch.setattr(
        evaluation,
        "run_evaluation",
        lambda **kwargs: _result(tmp_path, status, error_cases=error_cases, run_error=run_error),
    )

    assert cli.main(["evaluate"]) == expected
    output = capsys.readouterr().out
    assert str(tmp_path / "run" / "report.md") in output
    assert "人工质量评审：通过" not in output
    assert "待复核" in output


def test_evaluate_help_and_bad_limit_do_not_run_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_run(**kwargs: object):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(evaluation, "run_evaluation", fake_run)
    with pytest.raises(SystemExit) as help_exit:
        cli.main(["evaluate", "--help"])
    with pytest.raises(SystemExit) as bad_exit:
        cli.main(["evaluate", "--limit", "0"])

    assert help_exit.value.code == 0
    assert bad_exit.value.code == 2
    assert called is False


def test_evaluate_exception_is_redacted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        evaluation,
        "run_evaluation",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("API_KEY=secret-value")),
    )

    assert cli.main(["evaluate", "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "secret-value" not in captured.err
    assert json.loads(captured.err)["status"] == "error"


def test_system_and_turn_prompts_require_current_source_ids() -> None:
    assert "[S1]" in SYSTEM_PROMPT and "source_id" in SYSTEM_PROMPT and "本轮" in SYSTEM_PROMPT
    assert "[S1]" in _SOURCE_CITATION_INSTRUCTION
    assert "source_id" in _SOURCE_CITATION_INSTRUCTION
    assert "current reference materials" in _SOURCE_CITATION_INSTRUCTION
