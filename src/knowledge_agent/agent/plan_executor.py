"""Execute a validated plan with one shared budget and explicit dependencies.

The executor receives a leaf-step callback from runner. It does not instantiate
models, discover tools, change assessment state, or persist intermediate turns.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from ..config import AgentConfig
from ..schemas import AnswerResult, AnswerSource, AnswerStatus, RouteSummary
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


_SKIPPED_MESSAGES = {
    "plan_dependency_unfinished": "前置步骤未成功完成，本步骤未执行。",
    "plan_model_budget_exhausted": "本轮模型调用预算不足，本步骤未执行。",
    "plan_tool_budget_exhausted": "本轮工具调用预算已用完，本步骤未执行。",
    "plan_context_budget_exhausted": "本轮上下文预算已用完，本步骤未执行。",
}


def execute_plan(
    plan: TaskPlan,
    *,
    question: str,
    config: AgentConfig,
    initial_model_calls: int,
    trace: TraceRecorder,
    run_step: Callable[[PlanStep, StepBudget, str, TraceRecorder], StepOutcome],
) -> AnswerResult:
    """Run independent steps when possible, skipping failed dependencies.

    Results are assembled without another model rewrite, preserving each leaf's
    verified citation rendering and keeping external tool data in its own step.
    """
    results: dict[str, AnswerResult] = {}
    summaries: list[dict] = []
    sections: list[str] = []
    sources: dict[tuple[str, str], AnswerSource] = {}
    model_calls = initial_model_calls
    tool_calls = followup_searches = context_chars = 0
    limit_hit = False

    for step in plan.steps:
        handle = trace.start_event("plan_step", "program")
        dependencies = [results.get(key) for key in step.depends_on]
        skip_reason = None
        if any(result is None or result.status != "success" for result in dependencies):
            skip_reason = "plan_dependency_unfinished"
        # A generic MCP action requires one call to request tools and a second
        # to report their result. Do not begin it without both calls available.
        minimum_calls = 2 if step.kind == "mcp" else 1
        if skip_reason is None and config.max_model_calls - model_calls < minimum_calls:
            skip_reason = "plan_model_budget_exhausted"
        if skip_reason is None and tool_calls >= config.max_tool_calls:
            skip_reason = "plan_tool_budget_exhausted"

        dependency_context = ""
        if dependencies:
            dependency_context = json.dumps(
                {"completed_steps": [
                    {"step_id": key, "answer": results[key].answer}
                    for key in step.depends_on if key in results
                ]},
                ensure_ascii=False,
            )
        available_context = config.max_context_chars - context_chars - len(dependency_context)
        if skip_reason is None and available_context <= 0:
            skip_reason = "plan_context_budget_exhausted"

        if skip_reason is not None:
            limit_hit |= skip_reason != "plan_dependency_unfinished"
            summaries.append({
                "step_id": step.step_id, "goal": step.goal, "kind": step.kind,
                "depends_on": list(step.depends_on), "status": "skipped",
                "model_calls": 0, "tool_calls": 0, "stop_reason": skip_reason,
            })
            sections.append(f"### {step.step_id} · {step.goal}\n\n{_SKIPPED_MESSAGES[skip_reason]}")
            trace.finish_event(handle, "skipped", reason_code=step.step_id, error_code=skip_reason)
            continue

        remaining = StepBudget(
            model_calls=config.max_model_calls - model_calls,
            tool_calls=config.max_tool_calls - tool_calls,
            followup_searches=max(0, config.max_followup_searches - followup_searches),
            context_chars=available_context,
        )
        outcome = run_step(step, remaining, dependency_context, trace)
        result = outcome.result
        results[step.step_id] = result
        model_calls += result.model_calls
        tool_calls += result.tool_calls
        followup_searches += outcome.followup_searches
        context_chars += outcome.context_chars + len(dependency_context)
        limit_hit |= result.status == "limit_reached"
        summaries.append({
            "step_id": step.step_id, "goal": step.goal, "kind": step.kind,
            "depends_on": list(step.depends_on), "status": result.status,
            "model_calls": result.model_calls, "tool_calls": result.tool_calls,
            "stop_reason": result.stop_reason,
        })
        label = {
            "success": "", "insufficient_evidence": "资料不足：",
            "needs_clarification": "需要补充信息：", "service_error": "执行失败：",
            "limit_reached": "达到本轮限制：", "out_of_scope": "超出范围：",
        }[result.status]
        sections.append(f"### {step.step_id} · {step.goal}\n\n{label}{result.answer}")
        for source in result.sources:
            sources[(source.index_version, source.chunk_id)] = source
        trace.finish_event(
            handle, "success" if result.status == "success" else "error",
            reason_code=step.step_id, route_kind=step.kind,
            result_count=len(result.sources),
            error_code=result.stop_reason if result.status != "success" else None,
        )

    completed = sum(item["status"] == "success" for item in summaries)
    has_content = any(result.sources or result.status == "success" for result in results.values())
    if completed == len(plan.steps):
        status: AnswerStatus = "success"
        plan_status, stop_reason = "completed", "plan_completed"
    elif has_content:
        status = "insufficient_evidence"
        plan_status, stop_reason = "partial", "plan_partial"
    elif any(item["status"] == "needs_clarification" for item in summaries):
        status = "needs_clarification"
        plan_status, stop_reason = "needs_clarification", "plan_needs_clarification"
    elif any(item["status"] == "insufficient_evidence" for item in summaries):
        status = "insufficient_evidence"
        plan_status, stop_reason = "partial", "plan_insufficient_evidence"
    else:
        status = "limit_reached" if limit_hit else "service_error"
        plan_status, stop_reason = "failed", "plan_budget_exhausted" if limit_hit else "plan_failed"
    heading = f"已完成 {completed}/{len(plan.steps)} 个任务。"
    summary = validate_plan_summary({"version": 1, "status": plan_status, "steps": summaries})
    versions = {source.index_version for source in sources.values()}
    mixed_tools = any(step.kind == "mcp" for step in plan.steps)
    return AnswerResult(
        question=question, answer=heading + "\n\n" + "\n\n".join(sections), status=status,
        sources=tuple(sources.values()), index_version=next(iter(versions)) if len(versions) == 1 else None,
        model_calls=model_calls, tool_calls=tool_calls, stop_reason=stop_reason, plan=summary,
        routing=RouteSummary(
            route="mcp" if mixed_tools else "knowledge",
            intent="mcp_tool" if mixed_tools else "knowledge",
            decision_source="model", reason_code="multi_step_plan",
        ),
        trace=finish_trace_safely(
            trace, "completed" if status == "success" else "failed",
            error_code=None if status == "success" else stop_reason,
        ),
    )
