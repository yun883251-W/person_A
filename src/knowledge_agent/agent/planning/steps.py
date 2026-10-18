"""计划叶子步骤执行：按预算准备资料、调用模型/工具并校验答案。

依赖由调用方显式传入，不读取 Agent 实例，不提交子轮次或业务状态。
前置步骤的事实只通过 DependencyContext 中的证据快照传递。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from langchain_core.messages import HumanMessage
from browser_mcp.contracts import SEARCH_BILIBILI_TOOL

from knowledge_agent.config import AgentConfig, get_intent_settings
from knowledge_agent.knowledge.retrieval import KnowledgeRetriever
from knowledge_agent.models.provider import ChatModelInvocationError
from knowledge_agent.schemas import AnswerResult
from knowledge_agent.skills.selector import SkillSelectionSettings
from knowledge_agent.tools.registry import InMemoryToolRegistry
from knowledge_agent.observability.trace import TraceRecorder
from ..answering.answers import finalize_answer, finalize_mcp_answer
from ..dialogue.classifier import identify_intent
from ..dialogue.intent_types import IntentDecision, IntentKind
from ..model_loop import _model_error_code, run_model_loop
from ..preparation import (
    _PLAN_STEP_CONTEXT_INSTRUCTION, PreparedContextLimitError,
    prepare_question, prepare_model_turn, prepare_mcp_model_turn,
)
from ..state import TurnState
from . import planner
from .dependencies import DependencyContext, cited_evidence, is_dependency_transform
from .executor import StepBudget, StepOutcome
from .topic_references import validate_topic_binding


def run_planned_step(
    step: planner.PlanStep, budget: StepBudget, dependency_context: DependencyContext,
    trace: TraceRecorder, *, original_question: str, skill_id: str | None,
    model: object, retriever: KnowledgeRetriever,
    registry: InMemoryToolRegistry[TurnState], config: AgentConfig,
    system_prompt: str, selection_settings: SkillSelectionSettings,
    skill_mode: str | None = None,
    conversation_summary: str = "",
    knowledge_history: Sequence[object] = (),
) -> StepOutcome:
    """用剩余预算执行一个叶子；绝不提交子轮次。"""
    state = TurnState(original_question=step.goal, trace_recorder=trace, step_id=step.step_id)
    try:
        if step.kind == "knowledge":
            # 每个步骤只看到自己的问题、相关知识
            # 历史和显式依赖，绝不看到兄弟步骤的指令。
            background_limit = min(2000, max(1, budget.context_chars // 4))
            summary_text = conversation_summary[:background_limit]
            context_data: dict[str, Any] = {}
            clause_history = list(knowledge_history[-config.max_history_turns:])
            dependency_transform = is_dependency_transform(step.goal, bool(step.depends_on))
            if dependency_context.steps:
                bounded_dependencies = []
                for dependency in dependency_context.steps:
                    goal = dependency.goal[:1000]
                    bounded_dependencies.append({
                        "step_id": dependency.step_id, "goal": goal,
                        "answer": dependency.answer[:background_limit],
                        "may_be_truncated": len(dependency.answer) > background_limit,
                    })
                    clause_history.append({"question": goal, "route": "knowledge", "complete": True})
                context_data["completed_steps"] = bounded_dependencies
            decision = identify_intent(
                step.goal, context=tuple(clause_history),
                settings=replace(get_intent_settings(), mode="rules", browser_mode="rules"),
                max_context_chars=background_limit,
                max_history_turns=config.max_history_turns,
            )
            if dependency_transform and decision.kind in {
                IntentKind.KNOWLEDGE, IntentKind.NEEDS_CLARIFICATION,
            }:
                # 显式计划引用解析指代对象；只有下面的
                # 知识快照才能提供事实证据。
                # 若前置步骤仅涉及 MCP，准备阶段会报告
                # 缺少证据，而不是消耗未预留的搜索。
                decision = replace(
                    decision, kind=IntentKind.KNOWLEDGE, clarification_question="",
                    retrieval_query=step.goal, reason_code="plan_dependency_transform",
                )
            if decision.kind is IntentKind.MCP_TOOL:
                decision = IntentDecision(
                    kind=IntentKind.NEEDS_CLARIFICATION,
                    clarification_question="此步骤没有获得工具操作授权，请单独明确需要执行的操作。",
                    reason_code="plan_unconfirmed_clause_action",
                )
            if summary_text:
                context_data["session_summary_data"] = summary_text
                if decision.reason_code == "followup_without_context":
                    decision = replace(
                        decision, kind=IntentKind.KNOWLEDGE, clarification_question="",
                        retrieval_query=step.goal + "\n历史知识背景：" + summary_text,
                        reason_code="knowledge_summary_followup",
                    )
            if decision.reason_code == "knowledge_followup":
                context_data["recent_knowledge_questions"] = decision.retrieval_query
            context = json.dumps(context_data, ensure_ascii=False) if context_data else ""
            history = (HumanMessage(content="PLAN_CONTEXT_DATA\n" + context + "\nEND_PLAN_CONTEXT_DATA"),) if context else ()
            prepared = prepare_question(
                step.goal, retriever=retriever,
                max_context_chars=budget.context_chars,
                trace_recorder=trace, intent_decision=decision,
                max_model_calls=budget.model_calls,
                max_tool_calls=budget.tool_calls,
                max_followup_searches=budget.followup_searches,
                skill_id=skill_id,
                skill_mode=skill_mode,
                model=model,
                selection_settings=selection_settings,
                step_id=step.step_id,
                visible_context=context,
                dependency_context=dependency_context,
                dependency_transform=dependency_transform,
            )
            state = prepared.state
            if prepared.terminal_result is not None:
                return StepOutcome(prepared.terminal_result, state.followup_searches, state.context_chars)
            turn = prepare_model_turn(
                prepared, model=model, retriever=retriever,
                system_prompt=system_prompt + _PLAN_STEP_CONTEXT_INSTRUCTION,
                history=history,
                skill_id=skill_id, max_retries=0,
                allow_tool_calls=(
                    budget.model_calls - state.model_calls > 1
                    and budget.tool_calls > state.tool_calls
                ),
                max_context_chars=budget.context_chars,
            )
        else:
            if step.intent is not None and (
                step.intent.kind is not IntentKind.MCP_TOOL
                or step.intent.authorized_user_input != step.goal
                or step.intent.allowed_tool_names != step.allowed_tool_names
            ):
                raise ValueError("Plan step differs from the authorized user clause")
            if step.topic_binding is not None:
                event = trace.start_event("intent", "program", step_id=step.step_id)
                valid_binding = (
                    step.topic_binding.original_clause == step.goal
                    and step.allowed_tool_names == (SEARCH_BILIBILI_TOOL,)
                    and step.intent is not None
                    and validate_topic_binding(step.topic_binding, planner.raw_plan_clauses(original_question))
                )
                trace.finish_event(
                    event, "success" if valid_binding else "error", decision_source="rules",
                    intent_kind="mcp_tool", route_kind="mcp",
                    reason_code="browser_bilibili_topic_reference" if valid_binding else "plan_topic_reference_invalid",
                    error_code=None if valid_binding else "plan_topic_reference_invalid",
                )
                if not valid_binding:
                    return StepOutcome(AnswerResult(
                        question=step.goal, answer="搜索主题绑定无法核验，尚未执行浏览器操作。",
                        status="service_error", stop_reason="plan_topic_reference_invalid",
                    ))
                state.browser_search_topic = step.topic_binding.topic
            turn = prepare_mcp_model_turn(
                state, model=model, registry=registry,
                allowed_tool_names=step.allowed_tool_names,
                max_retries=0,
            )
        model_result = run_model_loop(
            turn, state, max_model_calls=budget.model_calls,
            max_tool_calls=budget.tool_calls,
            max_followup_searches=budget.followup_searches,
            max_context_chars=budget.context_chars,
        )
        result = (
            finalize_answer(model_result, state) if step.kind == "knowledge"
            else finalize_mcp_answer(model_result, state)
        )
        # 混合的工具结果不算完成的前置步骤。
        if result.stop_reason == "partial_tool_failure" and result.status == "success":
            result = replace(result, status="insufficient_evidence")
    except PreparedContextLimitError:
        result = AnswerResult(
            question=step.goal, answer="本步骤的资料与背景超出上下文限制，无法保留完整的可引用资料。",
            status="limit_reached", model_calls=state.model_calls,
            tool_calls=state.tool_calls, stop_reason="plan_context_budget_exhausted",
        )
    except ChatModelInvocationError as exc:
        result = AnswerResult(
            question=step.goal, answer="本步骤的模型调用失败，未自动重试。",
            status="service_error", model_calls=state.model_calls,
            tool_calls=state.tool_calls, stop_reason=_model_error_code(exc),
        )
    except Exception:
        # 不要重试失败的工具步骤：超时可能已经执行了它。
        result = AnswerResult(
            question=step.goal, answer="本步骤执行失败，未自动重试。",
            status="service_error", model_calls=state.model_calls,
            tool_calls=state.tool_calls, stop_reason="plan_step_failed",
        )
    return StepOutcome(
        result, state.followup_searches, state.context_chars,
        evidence=cited_evidence(result, state.supplied_evidence) if step.kind == "knowledge" else (),
    )

