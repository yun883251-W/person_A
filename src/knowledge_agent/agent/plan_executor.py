"""Execute a validated plan with one shared budget and explicit dependencies.

The executor receives a leaf-step callback from runner. It does not instantiate
models, discover tools, change assessment state, or persist intermediate turns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from browser_mcp.contracts import SEARCH_BILIBILI_TOOL

from ..config import AgentConfig
from ..schemas import AnswerResult, AnswerSource, AnswerStatus, RetrievalResult, RouteSummary, StepAnswerDiagnostics
from .answer_contract import ANSWER_VALIDATION_REASONS
from .answer_diagnostics import answer_diagnostics_trace_fields, public_answer_diagnostics
from .dependencies import DependencyContext, dependency_step, is_dependency_transform
from .planner import PlanStep, TaskPlan, validate_plan_summary
from .trace import TraceRecorder, finish_trace_safely


@dataclass(frozen=True, slots=True)
class StepBudget:
    model_calls: int
    tool_calls: int
    followup_searches: int
    context_chars: int


@dataclass(frozen=True, slots=True)
class StepOutcome:
    result: AnswerResult
    followup_searches: int = 0
    context_chars: int = 0
    # Runtime-only original material, never serialized into the public plan or
    # restored from a previous chat. Downstream leaves register new source IDs.
    evidence: tuple[RetrievalResult, ...] = ()


_SKIPPED_MESSAGES = {
    "plan_dependency_unfinished": "前置步骤未成功完成，本步骤未执行。",
    "plan_model_budget_exhausted": "本轮模型调用预算不足，本步骤未执行。",
    "plan_tool_budget_exhausted": "本轮工具调用预算已用完，本步骤未执行。",
    "plan_context_budget_exhausted": "本轮上下文预算已用完，本步骤未执行。",
}

def _outcome_kind(result: AnswerResult) -> str:
    """Separate evidence coverage from a failed execution or invalid answer."""
    if result.status in {"service_error", "limit_reached"} or result.stop_reason == "partial_tool_failure":
        return "execution"
    if result.stop_reason in ANSWER_VALIDATION_REASONS:
        return "validation"
    return {
        "success": "completed", "insufficient_evidence": "evidence",
        "needs_clarification": "clarification", "out_of_scope": "scope",
    }.get(result.status, "execution")


def minimum_step_budget(step: PlanStep) -> tuple[int, int]:
    """Reuse-only transformations need an answer but no fresh search call."""
    if step.kind == "knowledge" and is_dependency_transform(step.goal, bool(step.depends_on)):
        return 1, 0
    browser = step.allowed_tool_names == (SEARCH_BILIBILI_TOOL,)
    model_calls = 2 if step.kind == "mcp" and not browser else 1
    return model_calls, max(1, len(step.allowed_tool_names))


def execute_plan(
    plan: TaskPlan,
    *,
    question: str,
    config: AgentConfig,
    initial_model_calls: int,
    trace: TraceRecorder,
    run_step: Callable[[PlanStep, StepBudget, DependencyContext, TraceRecorder], StepOutcome],
) -> AnswerResult:
    """Run independent steps when possible, skipping failed dependencies.

    Results are assembled without another model rewrite, preserving each leaf's
    verified citation rendering and keeping external tool data in its own step.
    """
    results: dict[str, AnswerResult] = {}
    goals = {step.step_id: step.goal for step in plan.steps}
    kinds = {step.step_id: step.kind for step in plan.steps}
    evidence_by_step: dict[str, tuple[RetrievalResult, ...]] = {}
    summaries: list[dict] = []
    sections: list[str] = []
    sources: dict[tuple[str, str], AnswerSource] = {}
    model_calls = initial_model_calls
    tool_calls = followup_searches = 0
    limit_hit = False
    unavailable: set[str] = set()

    for position, step in enumerate(plan.steps):
        handle = trace.start_event("plan_step", "program", step_id=step.step_id)
        dependencies = [results.get(key) for key in step.depends_on]
        skip_reason = None
        if any(result is None or _outcome_kind(result) != "completed" for result in dependencies):
            skip_reason = "plan_dependency_unfinished"
        # Browser search has a program-authored receipt and can finish with a
        # single valid proposal. Its optional repair still uses remaining budget.
        # Generic MCP tools need a second model call to report their result.
        minimum_calls, minimum_tools = minimum_step_budget(step)
        if skip_reason is None and config.max_model_calls - model_calls < minimum_calls:
            skip_reason = "plan_model_budget_exhausted"
        if skip_reason is None and config.max_tool_calls - tool_calls < minimum_tools:
            skip_reason = "plan_tool_budget_exhausted"

        if skip_reason is not None:
            unavailable.add(step.step_id)
            limit_hit |= skip_reason != "plan_dependency_unfinished"
            summaries.append({
                "step_id": step.step_id, "goal": step.goal, "kind": step.kind,
                "depends_on": list(step.depends_on), "status": "skipped",
                "model_calls": 0, "tool_calls": 0, "stop_reason": skip_reason,
            })
            if step.topic_binding is not None:
                summaries[-1]["topic_binding"] = {
                    "topic": step.topic_binding.topic, "source_step_id": step.topic_binding.source_step_id,
                }
            sections.append(f"### {step.step_id} · {step.goal}\n\n{_SKIPPED_MESSAGES[skip_reason]}")
            trace.finish_event(
                handle, "skipped", reason_code=skip_reason,
                error_code=skip_reason if skip_reason != "plan_dependency_unfinished" else None,
            )
            continue

        # Transport completed knowledge answers with their actual cited source
        # snapshots. Tool results can gate a step, but are never KB evidence.
        dependency_context = DependencyContext(tuple(
            dependency_step(key, goals[key], results[key], evidence_by_step.get(key, ()))
            for key in step.depends_on
            if step.kind == "knowledge" and kinds[key] == "knowledge"
        ))

        # Leave a first answer/retrieval for every later viable step. Extra
        # selection calls, supplemental reads and repairs may use only surplus.
        reserved_models = reserved_tools = 0
        blocked = set(unavailable)
        for later in plan.steps[position + 1:]:
            if any(key in blocked for key in later.depends_on):
                blocked.add(later.step_id)
                continue
            required_models, required_tools = minimum_step_budget(later)
            reserved_models += required_models
            reserved_tools += required_tools
        remaining_models = config.max_model_calls - model_calls
        remaining_tools = config.max_tool_calls - tool_calls
        remaining = StepBudget(
            model_calls=min(remaining_models, max(minimum_calls, remaining_models - reserved_models)),
            tool_calls=min(remaining_tools, max(minimum_tools, remaining_tools - reserved_tools)),
            followup_searches=max(0, config.max_followup_searches - followup_searches),
            # Each leaf has its own message list. Previously consumed evidence
            # is not resident here; runner accounts for actual dependencies.
            context_chars=config.max_context_chars,
        )
        outcome = run_step(step, remaining, dependency_context, trace)
        result = outcome.result
        results[step.step_id] = result
        if step.kind == "knowledge":
            evidence_by_step[step.step_id] = outcome.evidence
        model_calls += result.model_calls
        tool_calls += result.tool_calls
        followup_searches += outcome.followup_searches
        outcome_kind = _outcome_kind(result)
        if outcome_kind != "completed":
            unavailable.add(step.step_id)
        limit_hit |= result.status == "limit_reached"
        step_summary = {
            "step_id": step.step_id, "goal": step.goal, "kind": step.kind,
            "depends_on": list(step.depends_on),
            "status": "service_error" if result.stop_reason == "partial_tool_failure" else result.status,
            "model_calls": result.model_calls, "tool_calls": result.tool_calls,
            "stop_reason": result.stop_reason,
        }
        diagnostic = public_answer_diagnostics(result.answer_diagnostics)
        if diagnostic is not None:
            step_summary["answer_diagnostics"] = diagnostic
        if step.topic_binding is not None:
            step_summary["topic_binding"] = {
                "topic": step.topic_binding.topic, "source_step_id": step.topic_binding.source_step_id,
            }
        summaries.append(step_summary)
        label = {
            "success": "", "insufficient_evidence": "资料覆盖不足：",
            "needs_clarification": "需要补充信息：", "service_error": "执行失败：",
            "limit_reached": "达到本轮限制：", "out_of_scope": "超出范围：",
        }[result.status]
        if outcome_kind == "validation":
            label = "回答校验未通过："
        elif result.stop_reason == "partial_tool_failure":
            label = "部分工具执行失败："
        sections.append(f"### {step.step_id} · {step.goal}\n\n{label}{result.answer}")
        for source in result.sources:
            sources[(source.index_version, source.chunk_id)] = source
        trace.finish_event(
            handle, {
                "completed": "success", "evidence": "empty",
                "clarification": "skipped", "scope": "skipped",
                "validation": "error", "execution": "error",
            }[outcome_kind],
            reason_code=result.stop_reason, route_kind=step.kind,
            result_count=len(result.sources),
            error_code=result.stop_reason if outcome_kind in {"validation", "execution"} else None,
            **answer_diagnostics_trace_fields(result.answer_diagnostics),
        )

    completed = sum(item["status"] == "success" for item in summaries)
    outcome_counts = {
        kind: sum(_outcome_kind(result) == kind for result in results.values())
        for kind in ("evidence", "clarification", "scope", "validation", "execution")
    }
    technical_failure = bool(outcome_counts["validation"] or outcome_counts["execution"] or limit_hit)
    has_content = any(result.sources or result.status == "success" for result in results.values())
    if completed == len(plan.steps):
        status: AnswerStatus = "success"
        plan_status, stop_reason = "completed", "plan_completed"
    elif has_content:
        status = "insufficient_evidence"
        plan_status = "partial"
        stop_reason = (
            "plan_partial_execution" if outcome_counts["execution"] or limit_hit
            else "plan_partial_validation" if outcome_counts["validation"]
            else "plan_insufficient_evidence" if outcome_counts["evidence"]
            else "plan_needs_clarification" if outcome_counts["clarification"]
            else "plan_partial"
        )
    elif any(item["status"] == "needs_clarification" for item in summaries):
        status = "needs_clarification"
        plan_status, stop_reason = "needs_clarification", "plan_needs_clarification"
    elif any(item["status"] == "insufficient_evidence" for item in summaries):
        status = "insufficient_evidence"
        plan_status = "partial"
        stop_reason = (
            "plan_partial_execution" if outcome_counts["execution"] or limit_hit
            else "plan_answer_validation_failed" if outcome_counts["validation"]
            else "plan_insufficient_evidence"
        )
    elif results and all(result.status == "out_of_scope" for result in results.values()):
        status = "out_of_scope"
        plan_status, stop_reason = "failed", "plan_out_of_scope"
    else:
        status = "limit_reached" if limit_hit else "service_error"
        plan_status, stop_reason = "failed", "plan_budget_exhausted" if limit_hit else "plan_failed"
    heading = f"已完成 {completed}/{len(plan.steps)} 个任务。"
    details = [
        f"{label} {outcome_counts[kind]} 步"
        for kind, label in (
            ("evidence", "资料覆盖不足"), ("clarification", "需要补充信息"),
            ("scope", "超出范围"), ("validation", "回答校验未通过"),
            ("execution", "执行异常或达到限制"),
        ) if outcome_counts[kind]
    ]
    skipped = sum(item["status"] == "skipped" for item in summaries)
    if skipped:
        details.append(f"未执行 {skipped} 步")
    if details:
        heading += "\n" + "；".join(details) + "。"
    format_warnings = sum(
        result.answer_diagnostics is not None
        and result.answer_diagnostics.validation_reason == "refusal_format_nonconforming"
        for result in results.values()
    )
    if format_warnings:
        heading += f"\n其中 {format_warnings} 步的模型回复格式异常，已保留其声明状态并使用程序说明。"
    summary = validate_plan_summary({"version": 1, "status": plan_status, "steps": summaries})
    versions = {source.index_version for source in sources.values()}
    mixed_tools = any(step.kind == "mcp" for step in plan.steps)
    return AnswerResult(
        question=question, answer=heading + "\n\n" + "\n\n".join(sections), status=status,
        sources=tuple(sources.values()), index_version=next(iter(versions)) if len(versions) == 1 else None,
        model_calls=model_calls, tool_calls=tool_calls, stop_reason=stop_reason, plan=summary,
        step_answer_diagnostics=tuple(
            StepAnswerDiagnostics(step_id, result.answer_diagnostics)
            for step_id, result in results.items() if result.answer_diagnostics is not None
        ),
        routing=RouteSummary(
            route="mcp" if mixed_tools else "knowledge",
            intent="mcp_tool" if mixed_tools else "knowledge",
            decision_source=plan.decision_source, reason_code="multi_step_plan",
        ),
        trace=finish_trace_safely(
            trace, "failed" if technical_failure else "completed",
            error_code=stop_reason if technical_failure else None,
        ),
    )
