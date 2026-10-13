"""知识问答、受控工具调用、MBTI 会话与多步骤任务的编排入口。

state 保存阶段数据，preparation 准备检索与提示词，model_loop 执行模型工具循环，
answers 校验引用并生成结果，应用生命周期由 knowledge_agent.application 管理。
本模块保留原公共类型与函数导入入口，已有 CLI/API 调用方式保持兼容。
验证 TODO 18—21 留待统一执行；本次整理不运行测试或真实模型联调。
"""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..config import AgentConfig, get_skill_selection_settings
from ..knowledge.retrieval import KnowledgeRetriever
from ..models import ChatModelInvocationError, initialize_chat_model, invoke_chat_model
from ..schemas import AnswerResult, AnswerStatus, RouteSummary
from ..session_store import SessionStore, SessionStoreError, StoredTurn
from ..skill_loader import load_skill
from ..tools.registry import InMemoryToolRegistry
from . import assessment, planner
from .answers import _attach_completed_trace, _attach_routing, finalize_answer, finalize_mcp_answer
from .intent import IntentDecision, IntentKind, IntentModelBudgetError, identify_intent
from .model_loop import (
    _model_error_code,
    continue_model_loop,
    execute_tool_calls,
    invoke_model_turn,
    run_model_loop,
)
from .plan_executor import StepBudget, StepOutcome, execute_plan
from .preparation import (
    _PLAN_STEP_CONTEXT_INSTRUCTION,
    classify_question,
    create_knowledge_tools,
    create_section_executor,
    prepare_evidence_context,
    prepare_mcp_model_turn,
    prepare_model_turn,
    prepare_question,
)
from .prompts import SYSTEM_PROMPT
from .router import RouteDecision, RouteKind, resolve_route
from .skill_selector import SkillSelectionSettings, normalize_skill_request
from .state import (
    ConversationHistory,
    ConversationTurn,
    KnowledgeTools,
    ModelTurnKind,
    ModelTurnResult,
    PreparedModelTurn,
    PreparedQuestion,
    QuestionKind,
    RetrievalStatus,
    ToolCallResult,
    ToolCallStatus,
    ToolExecutor,
    TurnPersistenceError,
    TurnState,
    _complete_history_messages,
)
from .trace import TraceRecorder, finish_trace_safely


def resolve_followup_question(
    question: str,
    turns: Sequence[ConversationTurn],
    *,
    max_history_turns: int = 6,
    max_context_chars: int = 4000,
) -> str:
    """Resolve a referential follow-up into a self-contained retrieval query."""
    decision = identify_intent(
        question,
        context=turns,
        max_history_turns=max_history_turns,
        max_context_chars=max_context_chars,
    )
    return decision.retrieval_query or question.strip()


class KnowledgeAgent:
    """Connect the chat client, system prompt, and lazy retrieval service."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        model: object | None = None,
        retriever: KnowledgeRetriever | None = None,
        session_store: SessionStore | None = None,
        mcp_registry: InMemoryToolRegistry[TurnState] | None = None,
        skill_selection_settings: SkillSelectionSettings | None = None,
    ) -> None:
        self.config = config if config is not None else AgentConfig()
        self.model = model if model is not None else initialize_chat_model()
        self.system_prompt = SYSTEM_PROMPT
        self.skill_selection_settings = (
            skill_selection_settings if skill_selection_settings is not None
            else get_skill_selection_settings()
        )
        self.retriever = retriever if retriever is not None else KnowledgeRetriever()
        self.history = ConversationHistory(self.config.max_history_turns)
        self.session_store = session_store
        self.assessment_states: dict[str, dict[str, Any]] = {}
        self.mcp_registry = (
            mcp_registry if mcp_registry is not None else InMemoryToolRegistry()
        )

    def _mcp_capabilities(self) -> tuple[dict[str, object], ...]:
        """Return the safe metadata snapshot used by intent and routing."""
        capabilities: list[dict[str, object]] = []
        for definition in self.mcp_registry.list_definitions():
            schema: object = {}
            raw_schema = getattr(definition, "args_schema", None)
            if isinstance(raw_schema, Mapping):
                schema = copy.deepcopy(dict(raw_schema))
            else:
                try:
                    schema = definition.get_input_schema().model_json_schema()
                except (AttributeError, TypeError, ValueError):
                    pass
            capabilities.append(
                {
                    "name": definition.name,
                    "description": definition.description or "",
                    "input_schema": schema,
                }
            )
        return tuple(capabilities)

    def prepare_question(
        self,
        question: str,
        *,
        skill_id: str | None = None,
        skill_mode: str | None = None,
        retrieval_question: str | None = None,
        trace_recorder: TraceRecorder | None = None,
        context: Sequence[object] = (),
        intent_decision: IntentDecision | None = None,
        intent_model_calls: int = 0,
    ) -> PreparedQuestion:
        """Validate and classify a question, proactively retrieving when required."""
        return prepare_question(
            question,
            retriever=self.retriever,
            max_context_chars=self.config.max_context_chars,
            retrieval_question=retrieval_question,
            trace_recorder=trace_recorder,
            context=context,
            model=self.model,
            max_model_calls=self.config.max_model_calls,
            max_tool_calls=self.config.max_tool_calls,
            max_followup_searches=self.config.max_followup_searches,
            max_history_turns=self.config.max_history_turns,
            intent_decision=intent_decision,
            intent_model_calls=intent_model_calls,
            skill_id=skill_id,
            skill_mode=skill_mode,
            selection_settings=self.skill_selection_settings,
        )

    def prepare_model_turn(
        self,
        prepared: PreparedQuestion,
        *,
        history: Sequence[BaseMessage] = (),
        skill_id: str | None = None,
    ) -> PreparedModelTurn:
        """Build the messages and tool-bound runnable for a prepared question."""
        return prepare_model_turn(
            prepared,
            model=self.model,
            retriever=self.retriever,
            system_prompt=self.system_prompt,
            history=history,
            skill_id=skill_id,
        )

    def invoke_model_turn(
        self,
        turn: PreparedModelTurn,
        state: TurnState,
    ) -> ModelTurnResult:
        """Invoke one prepared model turn and classify its complete response."""
        return invoke_model_turn(turn, state)

    def run_model_loop(
        self,
        turn: PreparedModelTurn,
        state: TurnState,
    ) -> ModelTurnResult | None:
        """Run a complete model/tool loop within this agent's configured limits."""
        return run_model_loop(
            turn,
            state,
            max_model_calls=self.config.max_model_calls,
            max_tool_calls=self.config.max_tool_calls,
            max_followup_searches=self.config.max_followup_searches,
            max_context_chars=self.config.max_context_chars,
        )

    def ask(
        self,
        question: str,
        session_id: str | None = None,
        *,
        skill_id: str | None = None,
        skill_mode: str | None = None,
    ) -> AnswerResult:
        """Dispatch an ordinary request or one bounded, explicitly multi-step task."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        request = normalize_skill_request(skill_mode, skill_id)
        if request.mode == "manual":
            load_skill(request.skill_id)
        snapshot = self.assessment_states.get(session_id) if session_id is not None else None
        if (
            planner.should_plan(question)
            and assessment.request_action(question, snapshot) is None
            and assessment.request_clarification(question, snapshot) is None
        ):
            return self._ask_plan(question.strip(), session_id, skill_id=request.skill_id, skill_mode=request.mode)
        return self._ask_single(question, session_id, skill_id=request.skill_id, skill_mode=request.mode)

    def _ask_plan(
        self, question: str, session_id: str | None, *, skill_id: str | None,
        skill_mode: str | None = None,
    ) -> AnswerResult:
        trace = TraceRecorder(requested_skill_id=skill_id)
        event = trace.start_event("planning", "program")
        planning_calls = 0

        def invoke_planner(messages: Sequence[BaseMessage]) -> object:
            nonlocal planning_calls
            if planning_calls >= self.config.max_model_calls:
                raise planner.PlanValidationError("plan_model_budget_exhausted")
            planning_calls += 1
            # A planning retry is another model request; this first version
            # makes exactly one attempt so its shared call budget stays exact.
            return invoke_chat_model(self.model, messages, max_retries=0)

        failure_status: AnswerStatus = "needs_clarification"
        try:
            if self.config.max_model_calls < 2:
                raise planner.PlanValidationError("plan_model_budget_exhausted")
            plan = planner.build_plan(
                question,
                context=self.history.turns(session_id) if session_id is not None else (),
                available_tools=self._mcp_capabilities(),
                model_invoker=invoke_planner, max_steps=3,
                max_context_chars=self.config.max_context_chars,
            )
        except planner.PlanValidationError as exc:
            code = exc.code
            answer = (
                "这条请求暂不能生成可执行计划。请拆成独立问题，或明确每一步的目标、"
                "工具名称及参数。MBTI 开始、作答、暂停和评分请单独操作。"
            )
            if code == "plan_model_budget_exhausted":
                failure_status = "limit_reached"
                answer = "本轮模型调用预算不足以规划并执行任务，请拆成单独问题。"
        except ChatModelInvocationError as exc:
            code = _model_error_code(exc)
            failure_status = "service_error"
            answer = "任务规划模型暂不可用，尚未执行任何步骤。请稍后重试。"
        except Exception:
            code = "plan_preparation_failed"
            failure_status = "service_error"
            answer = "任务规划暂未完成，尚未执行任何步骤。请检查模型与能力配置。"
        else:
            trace.finish_event(event, "success", decision_source="model", result_count=len(plan.steps))
            result = execute_plan(
                plan, question=question, config=self.config,
                initial_model_calls=planning_calls, trace=trace,
                run_step=lambda step, budget, context, recorder: self._run_planned_step(
                    step, budget, context, recorder, original_question=question, skill_id=skill_id,
                    skill_mode=skill_mode,
                ),
            )
            if session_id is not None:
                self._commit_turn(session_id, TurnState(original_question=question), result, (
                    HumanMessage(content=question), AIMessage(content=result.answer),
                ))
            return result

        trace.finish_event(event, "error", error_code=code)
        result = AnswerResult(
            question=question, answer=answer, status=failure_status,
            model_calls=planning_calls, stop_reason=code,
            routing=RouteSummary(
                route="direct", intent="needs_clarification", decision_source="fallback", reason_code=code,
            ),
            plan=planner.validate_plan_summary({
                "version": 1,
                "status": "needs_clarification" if failure_status == "needs_clarification" else "failed",
                "steps": [],
            }),
            trace=finish_trace_safely(trace, "failed", error_code=code),
        )
        if session_id is not None:
            self._commit_turn(session_id, TurnState(original_question=question), result, (
                HumanMessage(content=question), AIMessage(content=answer),
            ))
        return result

    def _run_planned_step(
        self, step: planner.PlanStep, budget: StepBudget, dependency_context: str,
        trace: TraceRecorder, *, original_question: str, skill_id: str | None,
        skill_mode: str | None = None,
    ) -> StepOutcome:
        """Execute one leaf with the remaining budget; never commit a child turn."""
        state = TurnState(original_question=step.goal, trace_recorder=trace, step_id=step.step_id)
        context = json.dumps({
            "original_user_request": original_question,
            "completed_dependency_data": dependency_context,
        }, ensure_ascii=False)
        try:
            if step.kind == "knowledge":
                decision = IntentDecision(
                    kind=IntentKind.KNOWLEDGE, retrieval_query=step.goal,
                    decision_source="model", reason_code="planned_knowledge_step",
                )
                prepared = prepare_question(
                    step.goal, retriever=self.retriever, max_context_chars=budget.context_chars,
                    trace_recorder=trace, intent_decision=decision,
                    max_model_calls=budget.model_calls,
                    max_tool_calls=budget.tool_calls,
                    max_followup_searches=budget.followup_searches,
                    skill_id=skill_id,
                    skill_mode=skill_mode,
                    model=self.model,
                    selection_settings=self.skill_selection_settings,
                    step_id=step.step_id,
                    visible_context=original_question,
                )
                state = prepared.state
                if prepared.terminal_result is not None:
                    return StepOutcome(prepared.terminal_result, state.followup_searches, state.context_chars)
                turn = prepare_model_turn(
                    prepared, model=self.model, retriever=self.retriever,
                    system_prompt=self.system_prompt + _PLAN_STEP_CONTEXT_INSTRUCTION,
                    history=(HumanMessage(content="PLAN_CONTEXT_DATA\n" + context),),
                    skill_id=skill_id, max_retries=0,
                )
            else:
                turn = prepare_mcp_model_turn(
                    state, model=self.model, registry=self.mcp_registry,
                    allowed_tool_names=step.allowed_tool_names,
                    max_retries=0, context=context,
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
            # A mixed tool outcome is not a completed prerequisite.
            if result.stop_reason == "partial_tool_failure" and result.status == "success":
                result = replace(result, status="insufficient_evidence")
        except Exception:
            # Do not retry a failed tool step: a timeout may have executed it.
            result = AnswerResult(
                question=step.goal, answer="本步骤执行失败，未自动重试。",
                status="service_error", model_calls=state.model_calls,
                tool_calls=state.tool_calls, stop_reason="plan_step_failed",
            )
        return StepOutcome(result, state.followup_searches, state.context_chars)

    def _ask_single(
        self,
        question: str,
        session_id: str | None = None,
        *,
        skill_id: str | None = None,
        skill_mode: str | None = None,
    ) -> AnswerResult:
        """Run one complete question-answer turn and return its public result."""
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not question.strip():
            raise ValueError("question must not be blank")
        request = normalize_skill_request(skill_mode, skill_id)
        trace_recorder = TraceRecorder(requested_skill_id=skill_id)
        turns: tuple[ConversationTurn, ...] = ()
        history: tuple[BaseMessage, ...] = ()
        if session_id is not None:
            turns = self.history.turns(session_id)

        intent_model_calls = 0
        mcp_capabilities = self._mcp_capabilities()
        assessment_state = self.assessment_states.get(session_id) if session_id is not None else None
        assessment_action = assessment.request_action(question, assessment_state)

        def invoke_intent_model(messages: Sequence[BaseMessage]) -> object:
            nonlocal intent_model_calls
            if intent_model_calls >= self.config.max_model_calls:
                raise IntentModelBudgetError()
            intent_model_calls += 1
            return invoke_chat_model(self.model, messages, max_retries=0)

        intent_event = trace_recorder.start_event("intent", "program")
        try:
            intent = identify_intent(
                question, context=turns, model_invoker=invoke_intent_model,
                available_tools=mcp_capabilities,
                max_history_turns=self.config.max_history_turns,
                max_context_chars=self.config.max_context_chars,
                assessment_state=assessment_state,
            )
        except Exception:
            trace_recorder.finish_event(
                intent_event, "error", error_code="intent_identification_failed"
            )
            raise
        fallback = intent.decision_source == "fallback"
        trace_recorder.finish_event(
            intent_event, "error" if fallback else "success",
            decision_source=intent.decision_source, intent_kind=intent.kind.value,
            reason_code=intent.reason_code,
            error_code=intent.reason_code if fallback else None,
        )
        route_event = trace_recorder.start_event("route", "program")
        try:
            route = resolve_route(
                intent,
                available_tools=mcp_capabilities,
                skill_id=skill_id,
                assessment_action=assessment_action,
            )
        except Exception:
            trace_recorder.finish_event(
                route_event, "error", error_code="route_resolution_failed"
            )
            raise
        trace_recorder.finish_event(
            route_event, "success",
            decision_source=route.intent.decision_source,
            intent_kind=route.intent.kind.value,
            route_kind=route.route.value,
            reason_code=route.reason_code,
        )
        if session_id is not None and route.route is RouteKind.KNOWLEDGE:
            history = self.history.messages(session_id, route=RouteKind.KNOWLEDGE)
        if assessment_action is not None and route.route is RouteKind.MCP:
            return self._ask_assessment(
                question, session_id, assessment_action, assessment_state, route,
                trace_recorder, intent_model_calls,
            )
        if route.route is RouteKind.MCP:
            state = TurnState(
                original_question=question.strip(),
                trace_recorder=trace_recorder,
            )
            state.model_calls = intent_model_calls
            turn = prepare_mcp_model_turn(
                state,
                model=self.model,
                registry=self.mcp_registry,
                allowed_tool_names=route.allowed_tool_names,
            )
            model_result = self.run_model_loop(turn, state)
            result = _attach_completed_trace(
                _attach_routing(finalize_mcp_answer(model_result, state), route),
                state,
            )
            if session_id is not None:
                self._commit_turn(session_id, state, result, state.messages[1:])
            return result
        prepared = self.prepare_question(
            question,
            skill_id=skill_id,
            skill_mode=request.mode,
            trace_recorder=trace_recorder,
            context=turns,
            intent_decision=route.intent,
            intent_model_calls=intent_model_calls,
        )
        if prepared.terminal_result is not None:
            result = _attach_completed_trace(
                _attach_routing(prepared.terminal_result, route), prepared.state
            )
            if session_id is not None:
                messages: tuple[BaseMessage, ...] = (
                    HumanMessage(content=prepared.state.original_question),
                    AIMessage(content=result.answer),
                )
                self._commit_turn(session_id, prepared.state, result, messages)
            return result

        turn = self.prepare_model_turn(prepared, history=history, skill_id=skill_id)
        model_result = self.run_model_loop(turn, prepared.state)
        result = _attach_completed_trace(
            _attach_routing(self.finalize_answer(model_result, prepared.state), route),
            prepared.state,
        )
        if session_id is not None:
            current_messages = prepared.state.messages[1 + len(history) :]
            self._commit_turn(session_id, prepared.state, result, current_messages)
        return result

    def _ask_assessment(
        self, question: str, session_id: str | None, action: str,
        previous: dict[str, Any] | None, route: RouteDecision,
        trace_recorder: TraceRecorder, model_calls: int,
    ) -> AnswerResult:
        state = TurnState(original_question=question.strip(), trace_recorder=trace_recorder)
        state.model_calls = model_calls

        def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            handle = trace_recorder.start_event("tool", "program", tool_name=name)
            registration = (
                self.mcp_registry.get(name)
                if name in assessment.TOOLS and name in route.allowed_tool_names else None
            )
            if registration is None:
                trace_recorder.finish_event(handle, "skipped", error_code="mbti_tool_unavailable")
                raise assessment.AssessmentToolError("mbti_tool_unavailable")
            if state.tool_calls >= self.config.max_tool_calls:
                trace_recorder.finish_event(handle, "skipped", error_code="tool_budget_exhausted")
                raise assessment.AssessmentToolError("tool_budget_exhausted")
            try:
                if registration.argument_validator is not None:
                    registration.argument_validator(arguments)
            except Exception:
                trace_recorder.finish_event(handle, "skipped", error_code="invalid_tool_arguments")
                raise assessment.AssessmentToolError("invalid_tool_arguments") from None
            state.tool_calls += 1
            output = self.mcp_registry.execute(name, arguments, context=state)
            if not isinstance(output, dict) or output.get("status") != "success":
                error = output.get("error", {}) if isinstance(output, dict) else {}
                code = error.get("code", "mbti_tool_failed") if isinstance(error, dict) else "mbti_tool_failed"
                trace_recorder.finish_event(handle, "error", error_code=str(code))
                raise assessment.AssessmentToolError(str(code))
            payload = output.get("structured_content")
            if not isinstance(payload, dict) and not output.get("truncated"):
                try:
                    payload = json.loads(output.get("text", ""))
                except (TypeError, ValueError):
                    payload = None
            if not isinstance(payload, dict):
                trace_recorder.finish_event(handle, "error", error_code="mbti_invalid_tool_result")
                raise assessment.AssessmentToolError("mbti_invalid_tool_result")
            size = len(json.dumps(payload, ensure_ascii=False))
            if state.context_chars + size > self.config.max_context_chars:
                trace_recorder.finish_event(handle, "error", error_code="context_budget_exhausted")
                raise assessment.AssessmentToolError("context_budget_exhausted")
            state.context_chars += size
            trace_recorder.finish_event(handle, "success")
            return payload

        try:
            if session_id is None and action != "describe":
                reply = assessment.AssessmentReply(
                    "请先创建会话，再开始 MBTI 测试。", status="needs_clarification",
                    stop_reason="mbti_session_required",
                )
            else:
                reply = assessment.handle_request(action, question, previous, call_tool)
        except assessment.AssessmentToolError as exc:
            reply = assessment.AssessmentReply(
                "MBTI 工具暂不可用，请确认 MCP 配置、mcp 可选依赖及内置题库文件。已有答题进度未改变。",
                previous, "service_error", exc.code,
            )
        except (TypeError, ValueError, KeyError):
            reply = assessment.AssessmentReply(
                "MBTI 工具返回的数据与当前问卷不兼容，已有进度未改变。",
                previous, "service_error", "mbti_invalid_tool_result",
            )
        snapshot = assessment.validate_snapshot(reply.state) if reply.state is not None else None
        result = AnswerResult(
            question=state.original_question, answer=reply.answer,
            status=reply.status, model_calls=state.model_calls, tool_calls=state.tool_calls,
            stop_reason=reply.stop_reason, assessment_state=snapshot,
        )
        result = _attach_completed_trace(_attach_routing(result, route), state)
        if session_id is not None:
            self._commit_turn(session_id, state, result, (
                HumanMessage(content=state.original_question), AIMessage(content=result.answer),
            ))
        return result

    def _commit_turn(
        self,
        session_id: str,
        state: TurnState,
        result: AnswerResult,
        messages: Sequence[object],
    ) -> None:
        """Commit only a completed turn; failures and partial tool exchanges are dropped."""
        if result.status in {"service_error", "limit_reached"}:
            return
        complete_messages = _complete_history_messages(messages)
        if not complete_messages:
            return
        turn = ConversationTurn(
            question=state.original_question,
            retrieval_question=state.retrieval_question or state.original_question,
            messages=complete_messages,
            result=result,
        )
        if self.session_store is not None:
            try:
                self.session_store.append_turn(
                    session_id,
                    StoredTurn(
                        turn_id=uuid.uuid4().hex,
                        turn_no=0,
                        question=turn.question,
                        retrieval_question=turn.retrieval_question,
                        messages=turn.messages,
                        result=turn.result,
                    ),
                )
            except SessionStoreError as exc:
                raise TurnPersistenceError(result) from exc
        if result.assessment_state is not None:
            self.assessment_states[session_id] = assessment.validate_snapshot(result.assessment_state)
        self.history.append(session_id, turn)

    def get_session_history(self, session_id: str) -> tuple[ConversationTurn, ...]:
        """Return an immutable snapshot of the session's retained complete turns."""
        return self.history.turns(session_id)

    def clear_session(self, session_id: str) -> None:
        """Remove all retained turns for one session."""
        self.history.clear(session_id)
        self.assessment_states.pop(session_id, None)

    def finalize_answer(
        self,
        result: ModelTurnResult | None,
        state: TurnState,
    ) -> AnswerResult:
        """Turn the model's terminal message into a verified public answer."""
        return finalize_answer(result, state)


# 旧应用级导入仍可用；延迟读取避免 application -> runner -> application 循环依赖。
_APPLICATION_EXPORTS = frozenset({
    "AgentApplicationError", "SessionNotStartedError", "ApplicationBusyError",
    "KnowledgeAgentApplication", "create_agent_application",
})


def __getattr__(name: str) -> Any:
    if name in _APPLICATION_EXPORTS:
        from .. import application

        return getattr(application, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 第三阶段：开发全部完成后统一验证。以下 TODO 暂不执行。

# TODO 18：集中执行可控的单轮流程测试。
# 使用假模型与假检索服务覆盖：先搜索后回答、追加小节读取、多个工具请求及空结果。
# 检查 AIMessage 与 ToolMessage 的调用 ID 配对、调用顺序、次数统计及结构化输出。
# 检查知识问题不会未经检索直接返回事实答案，范围说明与澄清请求可正常结束。
# 测试数据独立于真实知识库，不调用 DeepSeek 或 Ollama，也不修改真实索引。

# TODO 19：集中验证异常、引用和资料边界。
# 确认 SYSTEM_PROMPT 与附加引用指令均只要求模型输出本轮 [S1] 等来源标记。
# 检查合法标记由程序生成完整引用；未知标记、旧轮次来源或模型手写引用不能作为可信来源。
# 覆盖未知工具、非法参数、重复调用 ID、缺少工具结果、循环上限和上下文超限。
# 覆盖模型失败、检索失败、资料不足等状态，确认它们不会都被当作正常空答案。
# 覆盖未知来源、伪造行号、仅命中但未引用的片段，以及两个 26- 文件的区分。
# 使用包含角色指令和虚构业务制度的教材片段，检查不会执行其中指令或冒称真实政策。
# 检查错误与执行摘要不包含密钥，内部推理字段不被当作最终答复展示。

# TODO 20：集中执行真实 DeepSeek 与 Ollama 联调。
# 准备真实可用索引，使用配置的 Ollama Embedding 与 DeepSeek 对话模型跑通一次问答。
# 检查知识检索、工具请求、结果回填、补查及最终中文回答都能按约定串联。
# 使用 evaluation/questions.jsonl 的问题核对实际命中来源和答案依据。
# 逐条核对关键结论与引用片段是否相符，不能只因回答流畅或带有引用就判定通过。
# 当前索引、模型或服务尚未就绪的项目标记为未执行，不能填写通过结论。

# TODO 21：集中验证多轮追问、版本一致性与结束行为。
# 覆盖连续追问、跨主题提问、历史裁剪、清空会话和不同会话之间的状态隔离。
# 覆盖问答间刷新索引，确认新一轮使用新版本，单轮回答不混用不同批次资料。
# 覆盖达到调用上限、模型中断和失败恢复，确认下一轮不会继承残缺工具消息。
# 记录各场景的回答状态、来源准确性、调用次数和耗时，汇总待修复项。
# 全部验证完成后再说明哪些能力已经可用，不将已写代码等同于联调成功。
