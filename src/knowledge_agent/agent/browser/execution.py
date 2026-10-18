"""B 站搜索的单动作执行：模型生成参数，双端校验，程序报告打开结果。

最多允许首次参数生成和一次修正；任意真实工具尝试后都不再调用模型或重开浏览器。
复用 model_loop 注入的模型调用、注册器和 trace，避免另建模型客户端。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from browser_mcp.contracts import (
    SEARCH_BILIBILI_TOOL,
    build_search_url,
    validate_search_arguments,
)

from knowledge_agent.schemas import AnswerResult, AnswerStatus
from knowledge_agent.agent.router import RouteKind
from knowledge_agent.agent.state import ModelTurnResult, PreparedModelTurn, ToolCallResult, TurnState


class _TopicBindingMismatch(ValueError):
    """The model proposed a different subject than the host-resolved user topic."""


def is_browser_search_turn(turn: PreparedModelTurn) -> bool:
    """The special workflow owns exactly one approved browser capability."""
    return (
        turn.route is RouteKind.MCP
        and tuple(tool.name for tool in turn.tools.definitions) == (SEARCH_BILIBILI_TOOL,)
    )


def _finish(
    state: TurnState, answer: str, *, status: AnswerStatus, code: str,
) -> ModelTurnResult:
    message = AIMessage(content=answer)
    state.messages.append(message)
    state.status = status
    state.stop_reason = code
    state.browser_action_result = AnswerResult(
        question=state.original_question,
        answer=answer,
        status=status,
        model_calls=state.model_calls,
        tool_calls=state.tool_calls,
        stop_reason=code,
    )
    if state.trace_recorder is not None:
        handle = state.trace_recorder.start_event("finalize", "program", step_id=state.step_id)
        state.trace_recorder.finish_event(
            handle, "success" if status == "success" else "error",
            reason_code=code, error_code=code if status != "success" else None,
        )
    return ModelTurnResult(kind="final_answer", message=message, answer=answer)


def _reject_parameters(state: TurnState, code: str) -> None:
    """Record a rejected proposal, never the raw arguments or provider content."""
    if state.trace_recorder is not None:
        handle = state.trace_recorder.start_event(
            "tool", "model", tool_name=SEARCH_BILIBILI_TOOL, step_id=state.step_id,
        )
        state.trace_recorder.finish_event(handle, "skipped", error_code=code)


def _validated_call(message: AIMessage, state: TurnState) -> tuple[Mapping[str, Any], dict[str, str]]:
    # A malformed JSON call may be placed in invalid_tool_calls by LangChain.
    # Never silently ignore it and execute a different valid call in the batch.
    if message.invalid_tool_calls or len(message.tool_calls) != 1:
        raise ValueError("one_valid_tool_call_required")
    call = message.tool_calls[0]
    if call.get("name") != SEARCH_BILIBILI_TOOL:
        raise ValueError("unexpected_tool")
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id.strip() or call_id in state.completed_tool_call_ids:
        raise ValueError("invalid_tool_call_id")
    arguments = validate_search_arguments(call.get("args"))
    if state.browser_search_topic is not None and arguments["query"] != state.browser_search_topic:
        raise _TopicBindingMismatch("browser_search_topic_mismatch")
    return call, arguments


def _verified_payload(output: object, arguments: Mapping[str, str]) -> dict[str, Any] | None:
    """Accept only this tool's exact result contract for the submitted query."""
    if not isinstance(output, Mapping) or output.get("status") != "success" or output.get("truncated"):
        return None
    if output.get("tool_name") != SEARCH_BILIBILI_TOOL or output.get("server_name") != "browser":
        return None
    payload = output.get("structured_content")
    if payload is None:
        try:
            payload = json.loads(output.get("text", ""))
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, dict) or set(payload) != {
        "status", "action", "site", "query", "url", "browser_launch_requested", "page_verified",
    }:
        return None
    if (
        payload.get("status") != "success"
        or payload.get("action") != "browser_launch_requested"
        or payload.get("site") != "bilibili"
        or payload.get("query") != arguments["query"]
        or payload.get("url") != build_search_url(arguments["query"])
        or payload.get("browser_launch_requested") is not True
        or payload.get("page_verified") is not False
    ):
        return None
    return payload


def run_browser_search(
    turn: PreparedModelTurn,
    state: TurnState,
    *,
    invoke_model: Callable[..., ModelTurnResult],
    execute_tools: Callable[..., tuple[ToolCallResult, ...]],
    max_model_calls: int,
    max_tool_calls: int,
    max_context_chars: int,
) -> ModelTurnResult:
    """Generate one schema-valid search request and make at most one dispatch."""
    initial_messages = tuple(state.messages)
    arguments: dict[str, str] | None = None
    proposal: ModelTurnResult | None = None
    for attempt in range(2):
        if state.model_calls >= max_model_calls:
            state.messages[:] = initial_messages
            return _finish(
                state, "本轮模型调用预算不足，尚未请求打开浏览器。",
                status="limit_reached", code="model_call_budget_exhausted",
            )
        proposal = invoke_model(turn, state)
        try:
            _call, arguments = _validated_call(proposal.message, state)
        except (TypeError, ValueError) as exc:
            code = "browser_search_topic_mismatch" if isinstance(exc, _TopicBindingMismatch) else "browser_invalid_tool_arguments"
            _reject_parameters(state, code)
            # Drop malformed assistant calls rather than send an orphan call ID
            # back to the provider or persist them as a completed conversation.
            state.messages[:] = initial_messages
            if attempt == 0 and state.model_calls < max_model_calls:
                topic_repair = (
                    " The query must exactly match this host-resolved topic: "
                    + json.dumps(state.browser_search_topic, ensure_ascii=False) + "."
                    if state.browser_search_topic is not None else ""
                )
                state.messages.append(HumanMessage(content=(
                    "STRUCTURED_ARGUMENT_REPAIR: Your previous output was rejected before execution. "
                    "Call only mcp__browser__search_bilibili exactly once using native tool calling. "
                    "Its arguments must be a JSON object with exactly one field, query: a nonblank "
                    "string of 1 to 200 characters without control characters. Extract the search "
                    "topic from the original user request; do not invent a different topic. "
                    "Do not output JSON as plain answer text or issue additional calls." + topic_repair
                )))
                continue
            return _finish(
                state, (
                    "模型生成的搜索词与已确定的主题不一致，尚未请求打开浏览器。请重试。"
                    if code == "browser_search_topic_mismatch" else
                    "未能生成符合要求的搜索参数，尚未请求打开浏览器。请重试。"
                ),
                status="service_error", code=code,
            )
        break

    if proposal is None or arguments is None:
        raise RuntimeError("browser_parameter_generation_incomplete")
    url = build_search_url(arguments["query"])
    if state.tool_calls >= max_tool_calls:
        state.messages[:] = initial_messages
        return _finish(
            state, "本轮工具调用预算不足，尚未请求打开浏览器。",
            status="limit_reached", code="tool_budget_exhausted",
        )
    # Bound the stored tool message before an external action. The canonical
    # result is small and the final answer needs no further model context.
    canonical = {
        "status": "success", "action": "browser_launch_requested", "site": "bilibili",
        "query": arguments["query"], "url": url,
        "browser_launch_requested": True, "page_verified": False,
    }
    if state.context_chars + len(json.dumps(canonical, ensure_ascii=False)) > max_context_chars:
        state.messages[:] = initial_messages
        return _finish(
            state, "本轮上下文预算不足，尚未请求打开浏览器。",
            status="limit_reached", code="context_budget_exhausted",
        )

    # Retain the actual accepted proposal and matching result, not internal
    # repair prompts. All rejected model attempts remain counted in the trace.
    state.messages[:] = [*initial_messages, proposal.message]
    results = execute_tools(
        proposal.message, turn.tools, state,
        max_tool_calls=max_tool_calls,
    )
    result = results[0] if len(results) == 1 else None
    payload = _verified_payload(result.output, arguments) if result is not None and result.status == "success" else None
    if payload is not None:
        state.successful_tool_calls += 1
        public_output: dict[str, Any] = payload
    else:
        state.failed_tool_calls += 1
        code = "browser_result_unconfirmed"
        output = result.output if result is not None else None
        if isinstance(output, Mapping) and output.get("status") == "error":
            error = output.get("error")
            if isinstance(error, Mapping) and error.get("code") in {
                "timeout", "tool_timeout", "connection_lost", "unavailable", "remote_error",
                "invalid_tool_arguments", "invalid_arguments", "tool_budget_exhausted",
            }:
                code = str(error["code"])
        state.last_tool_error_code = code
        public_output = {"status": "error", "error": {"code": code}}

    serialized = json.dumps(public_output, ensure_ascii=False)
    state.context_chars += len(serialized)
    state.messages.append(ToolMessage(
        content=serialized,
        tool_call_id=proposal.message.tool_calls[0]["id"],
        name=SEARCH_BILIBILI_TOOL,
        status="success" if payload is not None else "error",
    ))
    if payload is not None:
        return _finish(
            state,
            "已向本机默认浏览器发送 B 站搜索打开请求。\n\n"
            f"关键词：{arguments['query']}\n\n[B站搜索页]({url})\n\n"
            "尚未检查页面是否加载，也未读取视频结果。",
            status="success", code="browser_launch_requested",
        )
    return _finish(
        state,
        "浏览器打开请求未获成功确认，本轮没有重复调用。请先查看浏览器窗口；"
        f"也可以使用此链接：[B站搜索页]({url})。",
        status="service_error", code=state.last_tool_error_code or "browser_result_unconfirmed",
    )
