"""知识库问答效果评估模块。

读取人工验收题，通过 Agent 应用接口逐题问答，保存结果和引用覆盖统计，
为人工判断答案正确性、完整性和忠实度提供报告。
本模块只统计运行与引用来源覆盖，不自动评判答案质量。

前置工作：先完成 agent/prompts.py 中修改 SYSTEM_PROMPT 第三节的 TODO，
统一为模型输出本轮 [S1] 等标记，由 Agent 校验并生成最终引用。
评估运行器已实现；对应 CLI 接入见 cli.py 的评估开发 TODO。
其余验证在批处理与 CLI 接入完成后集中执行，不逐项调用真实模型。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from time import perf_counter
from typing import Any, Callable, Literal, Mapping, Sequence, TextIO
import uuid

from .config import PROJECT_ROOT
from .schemas import AnswerResult, AnswerSource


# 第一阶段：题集、执行与报告开发。

RunStatus = Literal["completed", "completed_with_errors", "failed", "interrupted"]


@dataclass(frozen=True, slots=True)
class EvaluationExpectedSource:
    """一个人工标注的期望来源。"""

    file: str
    section: str | None = None
    role: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    """一道评估题及其人工验收标注。"""

    id: str
    type: str
    question: str
    expected_sources: tuple[EvaluationExpectedSource, ...] = ()
    expected_points: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """一个期望来源的引用核对结果。"""

    expected_source: EvaluationExpectedSource
    matched: bool
    actual_source: AnswerSource | None = None
    applicable: bool = True


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    """一道题的完整执行记录。"""

    question: EvaluationQuestion
    answer_result: AnswerResult | None
    elapsed_seconds: float
    citation_checks: tuple[CitationCheck, ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    interrupted: bool = False
    manual_review: Mapping[str, Any] = field(
        default_factory=lambda: {
            "status": "pending",
            "correctness": None,
            "completeness": None,
            "faithfulness": None,
            "notes": "",
        }
    )


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """一次评估运行的汇总数据。"""

    run_status: RunStatus
    run_error: str | None
    interrupted: bool
    total_cases: int
    completed_cases: int
    error_cases: int
    status_counts: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    citation_expected: int = 0
    citation_matched: int = 0
    citation_coverage: float | None = None
    started_at: str | None = None
    questions_path: str | None = None
    questions_sha256: str | None = None
    question_bank_cases: int | None = None
    limit: int | None = None
    system_prompt_sha256: str | None = None
    model_calls: int | None = None
    tool_calls: int | None = None
    index_versions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """一次评估运行的输出文件与汇总。"""

    run_dir: Path
    results_path: Path
    summary_path: Path
    report_path: Path
    summary: EvaluationSummary


class EvaluationInputError(ValueError):
    """题集或运行参数无效。"""


def _input_error(path: Path, line: int | None, field_name: str, message: str) -> EvaluationInputError:
    location = f"{path}:{line}" if line is not None else str(path)
    return EvaluationInputError(f"{location}: field '{field_name}': {message}")


def _string_list(value: Any, *, path: Path, line: int, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _input_error(path, line, field_name, "must be a list of strings")
    return tuple(value)


def _source_path_is_safe(value: str) -> bool:
    parsed = PurePosixPath(value.replace("\\", "/"))
    windows_path = PureWindowsPath(value)
    return (
        bool(value.strip())
        and not parsed.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and ".." not in parsed.parts
    )


def load_questions(path: str | Path) -> tuple[EvaluationQuestion, ...]:
    """读取并完整校验 UTF-8 JSONL 评估题集。"""

    source_path = Path(path)
    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    try:
        stream = source_path.open("r", encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise EvaluationInputError(f"{source_path}: unable to read question set") from exc
    with stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise _input_error(source_path, line_number, "record", "invalid JSON") from exc
            if not isinstance(record, dict):
                raise _input_error(source_path, line_number, "record", "must be an object")
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise _input_error(source_path, line_number, "id", "must be a non-empty string")
            if identifier in seen_ids:
                raise _input_error(source_path, line_number, "id", "must be unique")
            seen_ids.add(identifier)
            question_type = record.get("type")
            if not isinstance(question_type, str) or not question_type.strip():
                raise _input_error(source_path, line_number, "type", "must be a non-empty string")
            question_text = record.get("question")
            if not isinstance(question_text, str) or not question_text.strip():
                raise _input_error(source_path, line_number, "question", "must be a non-empty string")
            raw_sources = record.get("expected_sources", [])
            if not isinstance(raw_sources, list):
                raise _input_error(source_path, line_number, "expected_sources", "must be a list")
            expected_sources: list[EvaluationExpectedSource] = []
            for source_index, item in enumerate(raw_sources):
                field_prefix = f"expected_sources[{source_index}]"
                if not isinstance(item, dict):
                    raise _input_error(source_path, line_number, field_prefix, "must be an object")
                file_name = item.get("file")
                if not isinstance(file_name, str) or not _source_path_is_safe(file_name):
                    raise _input_error(
                        source_path, line_number, f"{field_prefix}.file",
                        "must be a non-empty relative path without '..'",
                    )
                section = item.get("section")
                role = item.get("role")
                if section is not None and not isinstance(section, str):
                    raise _input_error(source_path, line_number, f"{field_prefix}.section", "must be a string")
                if role is not None and not isinstance(role, str):
                    raise _input_error(source_path, line_number, f"{field_prefix}.role", "must be a string")
                expected_sources.append(EvaluationExpectedSource(file_name, section, role))
            questions.append(
                EvaluationQuestion(
                    id=identifier,
                    type=question_type,
                    question=question_text,
                    expected_sources=tuple(expected_sources),
                    expected_points=_string_list(
                        record.get("expected_points", []), path=source_path,
                        line=line_number, field_name="expected_points",
                    ),
                    acceptance_criteria=_string_list(
                        record.get("acceptance_criteria", []), path=source_path,
                        line=line_number, field_name="acceptance_criteria",
                    ),
                )
            )
    if not questions:
        raise EvaluationInputError(f"{source_path}: question set is empty")
    return tuple(questions)


def _resolve_project_path(path: str | Path | None, default: Path) -> Path:
    resolved = default if path is None else Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _normalize_path(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix().removeprefix("./")


def _section_matches(expected: str | None, actual: Sequence[str]) -> bool:
    if expected is None or not expected.strip():
        return True
    wanted = tuple(part.strip() for part in expected.split("/") if part.strip())
    headings = tuple(part.strip() for part in actual)
    width = len(wanted)
    return bool(wanted) and any(
        headings[index:index + width] == wanted
        for index in range(len(headings) - width + 1)
    )


def check_citations(
    question: EvaluationQuestion, sources: Sequence[AnswerSource],
) -> tuple[CitationCheck, ...]:
    """核对答案实际引用的来源，role 来源仅供人工参考。"""

    sources_by_path: dict[str, list[AnswerSource]] = {}
    for source in sources:
        sources_by_path.setdefault(_normalize_path(source.source_path), []).append(source)

    checks: list[CitationCheck] = []
    for expected in question.expected_sources:
        actual = next(
            (
                source
                for source in sources_by_path.get(_normalize_path(expected.file), ())
                if _section_matches(expected.section, source.heading_path)
            ),
            None,
        )
        applicable = not (question.type == "unanswerable" and expected.role is not None)
        checks.append(CitationCheck(expected, actual is not None, actual, applicable))
    return tuple(checks)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _write_case(stream: TextIO, result: EvaluationCaseResult) -> None:
    json.dump(_jsonable(result), stream, ensure_ascii=False, separators=(",", ":"))
    stream.write("\n")
    stream.flush()


def _case_is_error(result: EvaluationCaseResult) -> bool:
    return result.error_type is not None or (
        result.answer_result is not None
        and result.answer_result.status in {"service_error", "limit_reached"}
    )


def _build_summary(
    cases: Sequence[EvaluationCaseResult], *, total_cases: int, elapsed_seconds: float,
    interrupted: bool, run_error: str | None, started_at: str, questions_path: Path,
    questions_sha256: str, question_bank_cases: int, limit: int | None,
    system_prompt_sha256: str | None,
) -> EvaluationSummary:
    completed = [case for case in cases if not case.interrupted]
    status_counts = Counter(
        case.answer_result.status for case in completed if case.answer_result is not None
    )
    error_cases = sum(_case_is_error(case) for case in completed)
    expected = sum(check.applicable for case in completed for check in case.citation_checks)
    matched = sum(
        check.applicable and check.matched
        for case in completed for check in case.citation_checks
    )
    status: RunStatus
    if interrupted:
        status = "interrupted"
    elif run_error is not None:
        status = "failed"
    elif error_cases:
        status = "completed_with_errors"
    else:
        status = "completed"
    answers = [case.answer_result for case in completed if case.answer_result is not None]
    versions = {
        version
        for answer in answers
        for version in ([answer.index_version] + [source.index_version for source in answer.sources])
        if version
    }
    return EvaluationSummary(
        run_status=status, run_error=run_error, interrupted=interrupted,
        total_cases=total_cases, completed_cases=len(completed), error_cases=error_cases,
        status_counts=dict(status_counts), elapsed_seconds=elapsed_seconds,
        citation_expected=expected, citation_matched=matched,
        citation_coverage=(matched / expected if expected else None),
        started_at=started_at, questions_path=str(questions_path),
        questions_sha256=questions_sha256, question_bank_cases=question_bank_cases,
        limit=limit, system_prompt_sha256=system_prompt_sha256,
        model_calls=(None if any(case.interrupted for case in cases) else sum(answer.model_calls for answer in answers)),
        tool_calls=(None if any(case.interrupted for case in cases) else sum(answer.tool_calls for answer in answers)),
        index_versions=tuple(sorted(versions)),
    )


def _markdown_text(value: str) -> str:
    return value.replace("```", "`​``")


def _render_report(summary: EvaluationSummary, cases: Sequence[EvaluationCaseResult]) -> str:
    coverage = "不适用" if summary.citation_coverage is None else f"{summary.citation_coverage:.1%}"
    lines = [
        "# 知识库问答评估报告", "", "## 运行汇总", "",
        f"- 运行状态：`{summary.run_status}`",
        f"- 计划 / 已结束 / 未执行：{summary.total_cases} / {summary.completed_cases} / {summary.total_cases - summary.completed_cases}",
        f"- 执行失败题：{summary.error_cases}",
        f"- 引用来源覆盖：{summary.citation_matched}/{summary.citation_expected} ({coverage})",
        f"- 耗时：{summary.elapsed_seconds:.3f} 秒", "",
        "> 引用来源覆盖仅表示已引用来源对人工标注来源的覆盖，不是检索 recall，也不是答案正确率。答案质量仍待人工评审。",
    ]
    if summary.run_error:
        lines.extend(["", f"- 批次错误：{summary.run_error}"])
    for case in cases:
        question = case.question
        lines.extend([
            "", f"## {question.id} · {question.type}", "",
            f"**问题：** {_markdown_text(question.question)}", "",
        ])
        if case.interrupted:
            lines.append("**执行结果：** 用户中断，未取得完整答案。")
        elif case.error_type:
            lines.append(f"**执行结果：** {case.error_type} — {case.error_message}")
        elif case.answer_result is not None:
            lines.extend([
                f"**Agent 状态：** `{case.answer_result.status}`", "",
                "**实际回答：**", "", _markdown_text(case.answer_result.answer),
            ])
        lines.extend(["", "**引用核对：**", ""])
        if not case.citation_checks:
            lines.append("- 无标注来源。")
        for check in case.citation_checks:
            expected = check.expected_source
            label = "不适用（仅供人工参考）" if not check.applicable else ("命中" if check.matched else "未命中")
            actual = ""
            if check.actual_source is not None:
                source = check.actual_source
                actual = f"；实际 {source.source_path} / {' / '.join(source.heading_path)} / L{source.start_line}-L{source.end_line}"
            role = f"；role={expected.role}" if expected.role else ""
            lines.append(f"- [{label}] {expected.file} / {expected.section or '*'}{role}{actual}")
        lines.extend(["", "**期望要点：**", ""])
        lines.extend(f"- {_markdown_text(point)}" for point in question.expected_points)
        lines.extend(["", "**验收标准：**", ""])
        lines.extend(f"- {_markdown_text(item)}" for item in question.acceptance_criteria)
        lines.extend(["", "**人工评审：** pending；正确性、完整性、忠实度和备注待填写。"])
    return "\n".join(lines) + "\n"


def _default_application_factory() -> Any:
    from .application import create_agent_application

    return create_agent_application()


def _safe_close(application: Any) -> None:
    close = getattr(application, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def run_evaluation(
    questions_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    limit: int | None = None,
    application_factory: Callable[[], Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EvaluationRunResult:
    """执行一次隔离、增量落盘的知识库问答评估。"""

    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise EvaluationInputError("limit must be a positive integer")
    resolved_questions = _resolve_project_path(
        questions_path, PROJECT_ROOT / "evaluation" / "questions.jsonl",
    )
    all_questions = load_questions(resolved_questions)
    selected = all_questions[:limit] if limit is not None else all_questions
    questions_hash = hashlib.sha256(resolved_questions.read_bytes()).hexdigest()
    try:
        from .agent.prompts import SYSTEM_PROMPT

        prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    except (ImportError, AttributeError):
        prompt_hash = None
    report_parent = _resolve_project_path(output_dir, PROJECT_ROOT / "evaluation" / "reports")
    report_parent.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{uuid.uuid4().hex[:8]}"
    run_dir = report_parent / run_name
    run_dir.mkdir(exist_ok=False)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = perf_counter()
    cases: list[EvaluationCaseResult] = []
    application: Any | None = None
    interrupted = False
    run_error: str | None = None
    factory = application_factory or _default_application_factory

    with results_path.open("x", encoding="utf-8", newline="\n") as result_stream:
        try:
            application = factory()
        except KeyboardInterrupt:
            interrupted = True
        except Exception:
            run_error = "application_initialization_failed"

        if application is not None:
            for index, question in enumerate(selected):
                if application is None:
                    try:
                        application = factory()
                    except KeyboardInterrupt:
                        interrupted = True
                        break
                    except Exception:
                        run_error = "application_reinitialization_failed"
                        break
                case_started = perf_counter()
                session_started = False
                answer: AnswerResult | None = None
                error_type: str | None = None
                error_message: str | None = None
                case_interrupted = False
                cleanup_failed = False
                try:
                    session_id = f"evaluation-{run_dir.name}-{index + 1}-{question.id}"
                    application.start_session(session_id)
                    session_started = True
                    answer = application.ask(question.question)
                    if not isinstance(answer, AnswerResult):
                        error_type = "invalid_answer_result"
                        error_message = "Agent returned an invalid answer result."
                        answer = None
                except KeyboardInterrupt:
                    interrupted = True
                    case_interrupted = True
                except Exception:
                    error_type = "execution_error"
                    error_message = "The question could not be executed."
                finally:
                    if session_started:
                        try:
                            application.end_session()
                        except KeyboardInterrupt:
                            interrupted = True
                            case_interrupted = True
                            cleanup_failed = True
                        except Exception:
                            cleanup_failed = True
                            if error_type is None:
                                error_type = "session_cleanup_error"
                                error_message = "The isolated session could not be cleaned up."
                checks = check_citations(question, answer.sources if answer is not None else ())
                if cleanup_failed:
                    _safe_close(application)
                    application = None
                case = EvaluationCaseResult(
                    question=question, answer_result=answer,
                    elapsed_seconds=perf_counter() - case_started,
                    citation_checks=checks, error_type=error_type,
                    error_message=error_message, interrupted=case_interrupted,
                )
                cases.append(case)
                _write_case(result_stream, case)
                if not case_interrupted and progress is not None:
                    try:
                        progress(sum(not item.interrupted for item in cases), len(selected))
                    except Exception:
                        pass
                if interrupted:
                    break
    if application is not None:
        _safe_close(application)

    summary = _build_summary(
        cases, total_cases=len(selected), elapsed_seconds=perf_counter() - started_clock,
        interrupted=interrupted, run_error=run_error, started_at=started_at,
        questions_path=resolved_questions, questions_sha256=questions_hash,
        question_bank_cases=len(all_questions), limit=limit,
        system_prompt_sha256=prompt_hash,
    )
    try:
        summary_path.write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"evaluation summary could not be saved in {run_dir}") from exc
    try:
        report_path.write_text(_render_report(summary, cases), encoding="utf-8")
    except OSError:
        summary = _build_summary(
            cases, total_cases=len(selected), elapsed_seconds=perf_counter() - started_clock,
            interrupted=interrupted, run_error="report_write_failed", started_at=started_at,
            questions_path=resolved_questions, questions_sha256=questions_hash,
            question_bank_cases=len(all_questions), limit=limit,
            system_prompt_sha256=prompt_hash,
        )
        try:
            summary_path.write_text(
                json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise RuntimeError(f"evaluation report could not be saved in {run_dir}") from exc
    return EvaluationRunResult(run_dir, results_path, summary_path, report_path, summary)

# 02：读取并校验 JSONL 题集。
# 提供 load_questions(path)，逐行读取 UTF-8（兼容 BOM），允许跳过空行。
# 校验每条记录为对象、id 唯一、question 非空，type 和字符串列表类型有效。
# 校验 expected_sources 为列表，每项 file 非空；section 和 role 若提供则须为字符串。
# 来源 file 使用相对于知识库的完整路径，拒绝绝对路径和 .. 越界路径。
# 保留 expected_points、acceptance_criteria 和 role，不把它们当作模型指令。
# 读取错误提供题集路径、行号与字段名；空题集、重复 ID、坏 JSON 在调用模型前报错。
# 先校验整个题集，再应用 limit，避免未读取的坏记录被误认为有效题集。

# 03：整理路径与运行参数。
# 默认题集为 PROJECT_ROOT / "evaluation/questions.jsonl"。
# 默认报告父目录为 PROJECT_ROOT / "evaluation/reports"。
# 用户传入的相对路径均按项目根目录解析；limit 为可选正整数，不能默默接受零或负数。
# 每次在报告父目录内创建独立运行目录，名称含时间与唯一标识，不覆盖已有报告。
# 明确结果文件为 results.jsonl、summary.json、report.md；输出不得覆盖题集或索引产物。
# 在请求模型前完成题集校验和报告目录准备，目录不可写时直接结束。

# 04：记录本次评估的基本信息。
# 记录开始时间、题集路径、题集内容哈希、实际题数、limit 和系统提示词哈希。
# 题集总题数与本次选中题数分开记录，total_cases 表示本次准备执行的题数。
# 模型提供方与模型名如需记录，只保存明确允许公开的字段，不序列化整个配置或客户端。
# 索引版本从实际 AnswerResult 和 AnswerSource 收集，未返回时记为未知，不能编造。
# 模型调用次数与耗时来自实际运行；未执行的题目不填零耗时来冒充执行结果。

# 05：接入 Agent 的公开应用接口。
# 默认延迟导入并调用 create_agent_application，不能在模块导入或解析参数时调用。
# 允许 application_factory 注入假应用，供后续离线验证；评估层不直接调用聊天模型。
# 同一批次可复用一个应用的模型与检索服务，每题使用独立会话。
# 应用初始化失败属于批次级错误：记录 run_error 和未执行题数，生成失败汇总，
# 不把全部题目标为“资料不足”，也不把 error_cases 为零解释成运行成功。

# 06：完成一次隔离的题目问答。
# 按 start_session(本次运行与题目对应的唯一标识) → ask(question) → end_session() 执行。
# 只有题目中的 question 作为问题传入；期望来源、回答要点和验收标准不能泄露给模型。
# 记录 AnswerResult 中的 answer、status、sources、index_version、model_calls、
# tool_calls、stop_reason；CLI 和评估层不再次替换 [S1] 或重新生成引用编号。
# 用单调时钟记录单题耗时；保留输入顺序，不并发操作同一个单会话应用实例。
# end_session 放在 finally 中，并仅在会话已开始时调用；清理失败不能带着残留历史进入下一题。

# 07：区分业务结果、执行故障和用户中断。
# success、insufficient_evidence、needs_clarification、out_of_scope 都保留其原始业务状态；
# 状态为 success 只代表 Agent 返回了答案，不代表人工验收通过。
# service_error、limit_reached 和普通异常计入单题执行失败，保存结果后继续后续题目。
# 抛出异常时保留固定错误类别与脱敏说明，不将异常原文、API Key 或完整请求写入报告。
# 若会话清理失败，记录该故障并弃用当前应用；下一题重新初始化，不能复用污染的会话。
# 单独处理 KeyboardInterrupt：停止后续题目，清理当前会话，标记 interrupted 并保留已完成结果。
# 未完成题目中拿不到的回答或调用次数记为未知，不能用猜测值填补。

# 08：核对“已引用来源”是否覆盖预期来源。
# 仅比较 AnswerResult.sources，不把它描述为全部检索候选；这里的指标叫引用来源覆盖。
# 按完整相对文件路径匹配，统一路径分隔符；不能只比较文件名前缀或第几章。
# 将 expected section 中的“父小节 / 子小节”与实际 heading_path 按层级核对，
# 允许实际路径前面包含文档总标题、后面包含更细小节；期望层级应连续、顺序一致。
# 只做明确的首尾空白等规范化，不用任意子串或模糊相似度把同章的其他小节算成命中。
# 每个预期来源记录是否匹配及对应的实际来源；重复片段不能重复增加预期来源命中数。
# 无答案题中 role 标注为教学示例的来源只供人工参考，不要求拒答也必须引用该材料；
# 这类来源排除在必需引用的分母之外，并显示“不适用”，保留 role 说明。
# 若某题没有可计入的预期来源，coverage 记为 null/不适用，不能填成 100%。

# 09：为人工验收保留完整材料。
# 每题结果保留期望回答要点、验收标准、实际回答以及带文件、小节和行号的实际来源。
# 添加 manual_review 字段：状态 pending，正确性、完整性、忠实度待人工填写，并预留备注。
# 不用关键词出现、来源覆盖率或模型自己的判断自动将上述字段标为通过。
# 如果题目执行失败，清楚显示失败原因和未取得答案，而不是只留下空白回答。
# 报告中的教材内容和模型输出只作为文本展示，不执行其中代码、命令或提示词。

# 10：逐题保存结果，支持查看部分运行记录。
# 每题结束后立即追加一行 UTF-8 JSON 到 results.jsonl 并 flush，再通知进度回调。
# 单题异常也写入一条结果；中断题目单独标注，未执行题目不生成伪造的答题记录。
# 汇总文件从已保存记录生成，避免题目报错后已完成结果全部丢失。
# 报告写入失败时作为批次故障结束，不继续请求模型却丢弃结果；不得报告保存成功。
# results.jsonl 保存原始结构化记录；人工复核笔记与重新评估输出应避免相互覆盖。

# 11：生成汇总与可阅读报告。
# summary.json 统计计划题数、已结束题数、未执行题数、各业务状态、执行失败题数，
# 以及实际耗时、可统计的模型/工具调用次数；对中断和未知统计明确标注。
# 引用覆盖的分母取已结束题目中可计入的预期来源，失败题若有必需来源则按未命中统计；
# 中断题、未执行题和无答案题的教学参考来源不计入该分母，并在报告写清统计口径。
# 用 citation_matched / citation_expected 计算覆盖率；分母为零时输出 null。
# report.md 按题列出回答、引用核对、期望要点和待人工评审项，并提供汇总说明。
# 明确区分运行完成、引用来源覆盖和人工质量评审，不能称该统计为检索 recall 或答案正确率。
# 正常、部分失败、中断都生成对应汇总；不得把模型/API 失败显示为“全部通过”。

# 12：整理统一的评估调用入口。
# 建议公开接口：
# run_evaluation(questions_path=None, output_dir=None, *, limit=None,
#                application_factory=None, progress=None) -> EvaluationRunResult
# 按参数校验 → 准备报告 → 初始化应用 → 逐题执行与保存 → 生成汇总的顺序组织。
# progress(completed, total) 仅用于进度展示；终端打印与 CLI 退出码交给 cli.py。
# 用户中断时保存部分报告并返回 interrupted 结果，便于 CLI 显示路径后返回 130。
# 运行器不改写知识文档、现有向量索引、题集和 .env，也不自动安装或下载模型。
# 正式执行此入口会调用配置的 Ollama/DeepSeek；只有注入假应用的验证才是离线运行。
