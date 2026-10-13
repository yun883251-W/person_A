"""受预算约束的模型与工具循环，以及工具结果和来源 ID 的回填。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from ..models import (
    ChatModelAuthenticationError,
    ChatModelInvocationError,
    ChatModelNotFoundError,
    ChatModelRequestError,
    ChatModelServiceError,
    ChatModelTimeoutError,
    invoke_chat_model,
)
from ..schemas import AnswerSource
from ..tools.registry import create_tool_registry
from .router import RouteKind
from .state import KnowledgeTools, ModelTurnResult, PreparedModelTurn, ToolCallResult, TurnState


def _message_text(message: AIMessage) -> str:
    """Return only user-visible text content, excluding reasoning metadata."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts).strip()


def invoke_model_turn(
    turn: PreparedModelTurn,
    state: TurnState,
) -> ModelTurnResult:
    """Synchronously invoke the model and choose the next agent action."""
    state.model_calls += 1
    model_event = (
        state.trace_recorder.start_event(
            "model", "program", selected_skill_id=turn.skill_id, step_id=state.step_id,
        )
        if state.trace_recorder is not None
        else None
    )
    if turn.skill_id is not None and state.trace_recorder is not None:
        state.trace_recorder.mark_skill_applied(turn.skill_id)
    try:
        message = invoke_chat_model(
            turn.model,
            tuple(state.messages),
            max_retries=turn.max_retries,
        )
    except ChatModelInvocationError as exc:
        if state.trace_recorder is not None and model_event is not None:
            state.trace_recorder.finish_event(
                model_event,
                "error",
                error_code=_model_error_code(exc),
            )
        raise
    state.messages.append(message)

    if message.tool_calls:
        if state.trace_recorder is not None and model_event is not None:
            state.trace_recorder.finish_event(model_event, "success")
        return ModelTurnResult(kind="tool_requests", message=message)

    answer = _message_text(message)
    if not answer:
        if state.trace_recorder is not None and model_event is not None:
            state.trace_recorder.finish_event(model_event, "empty")
        return ModelTurnResult(kind="empty_response", message=message)
    if state.trace_recorder is not None and model_event is not None:
        state.trace_recorder.finish_event(model_event, "success")
    return ModelTurnResult(kind="final_answer", message=message, answer=answer)


def _tool_call_error(code: str, message: str) -> dict[str, Any]:
    return {"status": "error", "error": {"code": code, "message": message}}


def execute_tool_calls(
    message: AIMessage,
    tools: KnowledgeTools,
    state: TurnState,
    *,
    max_tool_calls: int,
    max_followup_searches: int | None = None,
) -> tuple[ToolCallResult, ...]:
    """Validate and sequentially execute the tool calls in one model message."""
    registry = tools.registry
    if registry is None:
        # Compatibility for callers that still instantiate KnowledgeTools
        # directly through the old two-field API.
        contextual_executors = {
            name: (
                lambda _context, _executor=executor, **arguments: _executor(
                    **arguments
                )
            )
            for name, executor in tools.executors.items()
        }
        registry = create_tool_registry(tools.definitions, contextual_executors)

    results = registry.execute_tool_calls(
        message.tool_calls,
        context=state,
        completed_tool_call_ids=state.completed_tool_call_ids,
        max_tool_calls=max_tool_calls,
        max_followup_searches=max_followup_searches,
    )
    return tuple(
        ToolCallResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            output=result.output,
        )
        for result in results
    )

def _register_new_sources(
    state: TurnState,
    tool_results: Sequence[ToolCallResult],
) -> dict[str, str]:
    """Register newly retrieved chunks and return their source IDs by chunk ID."""
    source_ids = {
        source.chunk_id: source_id for source_id, source in state.source_map.items()
    }
    returned_chunk_ids: list[str] = []
    seen_chunk_ids: set[str] = set()

    def collect_chunk_ids(value: Any) -> None:
        if isinstance(value, Mapping):
            chunk_id = value.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                returned_chunk_ids.append(chunk_id)
            for child in value.values():
                collect_chunk_ids(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect_chunk_ids(child)

    for tool_result in tool_results:
        if tool_result.status == "success":
            collect_chunk_ids(tool_result.output)

    used_numbers = {
        int(source_id[1:])
        for source_id in state.source_map
        if source_id.startswith("S") and source_id[1:].isdigit()
    }
    next_number = 1

    for chunk_id in returned_chunk_ids:
        if chunk_id in source_ids:
            continue
        result = state.evidence.get(chunk_id)
        if result is None:
            continue
        while next_number in used_numbers:
            next_number += 1
        source_id = f"S{next_number}"
        state.source_map[source_id] = AnswerSource.from_retrieval_result(result)
        source_ids[chunk_id] = source_id
        used_numbers.add(next_number)
        next_number += 1
    return source_ids


def _tool_message_output(output: Any, source_ids: Mapping[str, str]) -> Any:
    """Copy a tool result while annotating returned chunks with citable IDs."""
    if isinstance(output, Mapping):
        copied = {key: _tool_message_output(value, source_ids) for key, value in output.items()}
        chunk_id = copied.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id in source_ids:
            copied["source_id"] = source_ids[chunk_id]
        return copied
    if isinstance(output, list):
        return [_tool_message_output(value, source_ids) for value in output]
    if isinstance(output, tuple):
        return [_tool_message_output(value, source_ids) for value in output]
    return output


def continue_model_loop(
    turn: PreparedModelTurn,
    state: TurnState,
    result: ModelTurnResult,
    *,
    max_model_calls: int,
    max_tool_calls: int,
    max_followup_searches: int = 2,
    max_context_chars: int = 24_000,
) -> ModelTurnResult:
    """Backfill tool results and continue invoking the model until it stops."""
    if isinstance(max_model_calls, bool) or not isinstance(max_model_calls, int):
        raise TypeError("max_model_calls must be an integer")
    if max_model_calls < 1:
        raise ValueError("max_model_calls must be greater than 0")
    if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int):
        raise TypeError("max_tool_calls must be an integer")
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be greater than 0")
    if isinstance(max_followup_searches, bool) or not isinstance(
        max_followup_searches, int
    ):
        raise TypeError("max_followup_searches must be an integer")
    if max_followup_searches < 0:
        raise ValueError("max_followup_searches must not be negative")
    if isinstance(max_context_chars, bool) or not isinstance(max_context_chars, int):
        raise TypeError("max_context_chars must be an integer")
    if max_context_chars < 1:
        raise ValueError("max_context_chars must be greater than 0")

    while result.kind == "tool_requests":
        # A tool result requires another model call to turn it into an answer.
        # Stop before executing more tools when that final call cannot be made.
        if state.model_calls >= max_model_calls:
            state.status = "limit_reached"
            state.stop_reason = "model_call_budget_exhausted"
            return result
        tool_results = execute_tool_calls(
            result.message,
            turn.tools,
            state,
            max_tool_calls=max_tool_calls,
            max_followup_searches=max_followup_searches,
        )
        if any(not tool_result.tool_call_id for tool_result in tool_results):
            state.status = "service_error"
            state.stop_reason = "invalid_tool_call_id"
            return result
        budget_exhausted: str | None = None
        for tool_result in tool_results:
            output = tool_result.output
            message_status = tool_result.status
            if isinstance(output, Mapping):
                error = output.get("error")
                code = error.get("code") if isinstance(error, Mapping) else None
                if tool_result.status == "error" and isinstance(code, str):
                    state.last_tool_error_code = code
                if code in {
                    "tool_budget_exhausted",
                    "followup_search_budget_exhausted",
                }:
                    budget_exhausted = str(code)

            previous_source_ids = set(state.source_map)
            if tool_result.status == "success" and turn.route is RouteKind.KNOWLEDGE:
                source_ids = _register_new_sources(state, (tool_result,))
                output = _tool_message_output(output, source_ids)
            serialized = json.dumps(output, ensure_ascii=False, default=str)
            if tool_result.status == "success":
                state.successful_tool_calls += 1
                if state.context_chars + len(serialized) > max_context_chars:
                    for source_id in set(state.source_map) - previous_source_ids:
                        del state.source_map[source_id]
                    output = _tool_call_error(
                        "context_budget_exhausted",
                        "The evidence context budget for this turn has been exhausted.",
                    )
                    serialized = json.dumps(output, ensure_ascii=False, default=str)
                    message_status = "error"
                    state.successful_tool_calls -= 1
                    state.failed_tool_calls += 1
                    state.last_tool_error_code = "context_budget_exhausted"
                    budget_exhausted = "context_budget_exhausted"
                else:
                    state.context_chars += len(serialized)
            else:
                state.failed_tool_calls += 1
            state.messages.append(
                ToolMessage(
                    content=serialized,
                    tool_call_id=tool_result.tool_call_id,
                    name=tool_result.name,
                    status=message_status,
                )
            )

        if budget_exhausted is not None:
            state.status = "limit_reached"
            state.stop_reason = budget_exhausted
            return result
        try:
            result = invoke_model_turn(turn, state)
        except ChatModelInvocationError as exc:
            _record_model_failure(state, exc)
            return result

    if result.kind == "empty_response":
        state.status = "service_error"
        state.stop_reason = "model_empty_response"
    return result

def _model_error_code(exc: ChatModelInvocationError) -> str:
    """Return the stable, non-sensitive public category for a model failure."""
    if isinstance(exc, ChatModelAuthenticationError):
        return "model_authentication_failed"
    if isinstance(exc, ChatModelTimeoutError):
        return "model_timeout"
    if isinstance(exc, ChatModelNotFoundError):
        return "model_not_found"
    if isinstance(exc, ChatModelRequestError):
        return "model_request_invalid"
    if isinstance(exc, ChatModelServiceError):
        return "model_service_unavailable"
    return "model_invocation_failed"


def _record_model_failure(state: TurnState, exc: ChatModelInvocationError) -> None:
    """Record a stable, non-sensitive public category for a model failure."""
    state.status = "service_error"
    state.stop_reason = _model_error_code(exc)


def run_model_loop(
    turn: PreparedModelTurn,
    state: TurnState,
    *,
    max_model_calls: int,
    max_tool_calls: int,
    max_followup_searches: int = 2,
    max_context_chars: int = 24_000,
) -> ModelTurnResult | None:
    """Run one bounded model/tool loop, returning failures through ``state``."""
    try:
        if isinstance(max_model_calls, bool) or not isinstance(max_model_calls, int):
            raise TypeError
        if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int):
            raise TypeError
        if isinstance(max_followup_searches, bool) or not isinstance(
            max_followup_searches, int
        ):
            raise TypeError
        if isinstance(max_context_chars, bool) or not isinstance(max_context_chars, int):
            raise TypeError
        if (
            max_model_calls < 1
            or max_tool_calls < 1
            or max_followup_searches < 0
            or max_context_chars < 1
        ):
            raise ValueError
        if state.model_calls >= max_model_calls:
            state.status = "limit_reached"
            state.stop_reason = "model_call_budget_exhausted"
            return None
        result = invoke_model_turn(turn, state)
        return continue_model_loop(
            turn,
            state,
            result,
            max_model_calls=max_model_calls,
            max_tool_calls=max_tool_calls,
            max_followup_searches=max_followup_searches,
            max_context_chars=max_context_chars,
        )
    except ChatModelInvocationError as exc:
        _record_model_failure(state, exc)
    except (TypeError, ValueError):
        state.status = "service_error"
        state.stop_reason = "invalid_agent_parameters"
    return None
