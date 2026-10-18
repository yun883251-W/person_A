"""工具注册与统一执行模块。

模块功能：管理提供给模型的工具定义，以及服务端实际执行的函数。
输入：工具定义、执行函数、模型工具调用、当前问答轮状态与调用预算。
输出：模型可见的工具列表，以及关联 tool_call_id 的结构化调用结果。

agent/model_loop.py 通过此注册器执行工具；Skill 与 MCP 复用同一能力边界。
工具定义与参数校验复用 LangChain StructuredTool。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableSet
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from langchain_core.tools import BaseTool

ExecutionContextT = TypeVar("ExecutionContextT")

DEFAULT_TOOL_NAMES = frozenset({"search_knowledge", "read_knowledge_section"})


class ToolExecutor(Protocol[ExecutionContextT]):
    """在某一轮次的执行上下文中实现工具的调用方。"""

    def __call__(
        self,
        context: ExecutionContextT,
        /,
        **arguments: Any,
    ) -> Any: ...


ArgumentValidator = Callable[[Mapping[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ToolRegistration(Generic[ExecutionContextT]):
    """模型可见的工具定义与其服务端执行器的配对。"""

    name: str
    definition: BaseTool
    executor: ToolExecutor[ExecutionContextT]
    argument_validator: ArgumentValidator | None = None


ToolCallStatus = Literal["success", "error"]


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """按请求顺序排列的单次模型工具调用结果。"""

    tool_call_id: str | None
    name: str | None
    status: ToolCallStatus
    output: Any


def _tool_call_error(code: str, message: str) -> dict[str, Any]:
    return {"status": "error", "error": {"code": code, "message": message}}


def _tool_call_result(
    tool_call_id: str | None,
    name: str | None,
    output: Any,
) -> ToolCallResult:
    """构造结果，同时保留 ``empty`` 等非错误工具状态。"""
    status: ToolCallStatus = (
        "error"
        if isinstance(output, Mapping) and output.get("status") == "error"
        else "success"
    )
    return ToolCallResult(
        tool_call_id=tool_call_id,
        name=name,
        status=status,
        output=output,
    )


def _execute_registration(
    registration: ToolRegistration[ExecutionContextT],
    context: ExecutionContextT,
    arguments: Mapping[str, Any],
) -> Any:
    """执行工具，但不向模型暴露异常细节。"""
    try:
        return registration.executor(context, **arguments)
    except Exception:  # noqa: BLE001 - tool failures are isolated per call
        return _tool_call_error(
            "tool_execution_failed",
            "The requested tool could not be completed.",
        )


def _validate_arguments(
    registration: ToolRegistration[ExecutionContextT],
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """校验参数，同时不丢失非 Pydantic schema 中的字段。"""
    validator = registration.argument_validator
    if validator is not None:
        validated = validator(arguments)
        if not isinstance(validated, dict):
            raise TypeError("argument validator must return a dictionary")
        return validated
    return (
        registration.definition.get_input_schema()
        .model_validate(dict(arguments))
        .model_dump()
    )


def _start_tool_trace(
    context: object,
    *,
    name: str | None,
    call_id: str | None,
) -> tuple[object | None, int | None]:
    recorder = getattr(context, "trace_recorder", None)
    if recorder is None:
        return None, None
    return recorder, recorder.start_event(
        "tool",
        "model",
        tool_name=name,
        tool_call_id=call_id,
    )


def _finish_tool_trace(
    recorder: object | None,
    handle: int | None,
    output: Any,
    *,
    executed: bool,
) -> None:
    if recorder is None or handle is None:
        return

    output_status = output.get("status") if isinstance(output, Mapping) else None
    error = output.get("error") if isinstance(output, Mapping) else None
    error_code = error.get("code") if isinstance(error, Mapping) else None
    result_count: int | None = None
    index_version: str | None = None
    if isinstance(output, Mapping):
        raw_results = output.get("results")
        raw_chunks = output.get("chunks")
        if isinstance(raw_results, list):
            result_count = len(raw_results)
        elif isinstance(raw_chunks, list):
            result_count = len(raw_chunks)
        raw_index_version = output.get("index_version")
        if raw_index_version is not None:
            index_version = str(raw_index_version)

    status = (
        "skipped"
        if not executed
        else "error"
        if output_status == "error"
        else "empty"
        if output_status == "empty"
        else "success"
    )
    recorder.finish_event(
        handle,
        status,
        result_count=result_count,
        index_version=index_version,
        error_code=str(error_code) if error_code is not None else None,
    )


@runtime_checkable
class ToolRegistry(Protocol[ExecutionContextT]):
    """注册、暴露并执行已批准工具的接口。"""

    def register(self, registration: ToolRegistration[ExecutionContextT]) -> None:
        """添加一个工具注册项。"""

    def get(self, name: str) -> ToolRegistration[ExecutionContextT] | None:
        """返回 ``name`` 对应的注册项（如果存在）。"""

    def list_definitions(self) -> tuple[BaseTool, ...]:
        """按注册顺序返回模型可见的定义。"""

    def list_registrations(self) -> tuple[ToolRegistration[ExecutionContextT], ...]:
        """按注册顺序返回注册项。"""

    def filter_for_skill(
        self,
        allowed_tools: Iterable[str] | None = None,
    ) -> ToolRegistry[ExecutionContextT]:
        """返回仅包含本轮可用工具的注册器。"""

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        context: ExecutionContextT,
    ) -> Any:
        """在给定的轮次局部上下文中执行已注册工具。"""

    def execute_tool_calls(
        self,
        calls: Iterable[object],
        *,
        context: ExecutionContextT,
        completed_tool_call_ids: MutableSet[str],
        max_tool_calls: int,
        max_followup_searches: int | None = None,
    ) -> tuple[ToolCallResult, ...]:
        """按请求顺序校验并执行模型请求的调用。"""


# 第一阶段：注册与执行开发。

class InMemoryToolRegistry(Generic[ExecutionContextT]):
    """显式批准工具的高速、保持插入顺序的注册器。"""

    __slots__ = ("_registrations",)

    def __init__(
        self,
        registrations: Iterable[ToolRegistration[ExecutionContextT]] = (),
    ) -> None:
        self._registrations: dict[str, ToolRegistration[ExecutionContextT]] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: ToolRegistration[ExecutionContextT]) -> None:
        name = registration.name
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool name must be a non-empty string")
        if name in self._registrations:
            raise ValueError(f"tool {name!r} is already registered")
        if registration.definition.name != name:
            raise ValueError("tool definition name must match registration name")
        if not callable(registration.executor):
            raise TypeError("tool executor must be callable")
        if (
            registration.argument_validator is not None
            and not callable(registration.argument_validator)
        ):
            raise TypeError("tool argument validator must be callable")

        self._registrations[name] = registration

    def get(self, name: str) -> ToolRegistration[ExecutionContextT] | None:
        return self._registrations.get(name)

    def list_definitions(self) -> tuple[BaseTool, ...]:
        return tuple(
            registration.definition for registration in self._registrations.values()
        )

    def list_registrations(self) -> tuple[ToolRegistration[ExecutionContextT], ...]:
        """暴露安全的注册项枚举，供组合注册器使用。"""
        return tuple(self._registrations.values())

    def filter_for_skill(
        self,
        allowed_tools: Iterable[str] | None = None,
    ) -> InMemoryToolRegistry[ExecutionContextT]:
        allowed_names = (
            DEFAULT_TOOL_NAMES if allowed_tools is None else frozenset(allowed_tools)
        )
        return type(self)(
            registration
            for name, registration in self._registrations.items()
            if name in allowed_names
        )

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        context: ExecutionContextT,
    ) -> Any:
        registration = self._registrations.get(name)
        if registration is None:
            return _tool_call_error(
                "unknown_tool", "The requested tool is not available."
            )
        if not isinstance(arguments, Mapping):
            return _tool_call_error(
                "invalid_tool_arguments", "Tool arguments must be an object."
            )
        try:
            validated_arguments = _validate_arguments(registration, arguments)
        except Exception:  # noqa: BLE001 - schemas may raise custom errors
            return _tool_call_error(
                "invalid_tool_arguments",
                "Tool arguments do not match the required schema.",
            )
        return _execute_registration(registration, context, validated_arguments)

    def execute_tool_calls(
        self,
        calls: Iterable[object],
        *,
        context: ExecutionContextT,
        completed_tool_call_ids: MutableSet[str],
        max_tool_calls: int,
        max_followup_searches: int | None = None,
    ) -> tuple[ToolCallResult, ...]:
        if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int):
            raise TypeError("max_tool_calls must be an integer")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be greater than 0")
        if max_followup_searches is not None and (
            isinstance(max_followup_searches, bool)
            or not isinstance(max_followup_searches, int)
        ):
            raise TypeError("max_followup_searches must be an integer")
        if max_followup_searches is not None and max_followup_searches < 0:
            raise ValueError("max_followup_searches must not be negative")

        results: list[ToolCallResult] = []

        for call in calls:
            if not isinstance(call, Mapping):
                recorder, trace_handle = _start_tool_trace(
                    context, name=None, call_id=None
                )
                output = _tool_call_error(
                    "invalid_tool_call", "The tool call must be an object."
                )
                results.append(ToolCallResult(None, None, "error", output))
                _finish_tool_trace(recorder, trace_handle, output, executed=False)
                continue

            raw_id = call.get("id")
            call_id = raw_id.strip() if isinstance(raw_id, str) else None
            raw_name = call.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) else None
            recorder, trace_handle = _start_tool_trace(
                context, name=name, call_id=call_id
            )

            if not call_id:
                output = _tool_call_error(
                    "invalid_tool_call_id",
                    "The tool call ID must be a non-empty string.",
                )
                results.append(ToolCallResult(None, name, "error", output))
                _finish_tool_trace(recorder, trace_handle, output, executed=False)
                continue

            if call_id in completed_tool_call_ids:
                output = _tool_call_error(
                    "duplicate_tool_call_id",
                    "This tool call ID has already been handled in the current turn.",
                )
                results.append(ToolCallResult(call_id, name, "error", output))
                _finish_tool_trace(recorder, trace_handle, output, executed=False)
                continue

            # 可用的 ID 即使后续校验失败，也视为已终结处理。
            completed_tool_call_ids.add(call_id)

            registration = self._registrations.get(name) if name else None
            if registration is None:
                output = _tool_call_error(
                    "unknown_tool", "The requested tool is not available."
                )
                results.append(ToolCallResult(call_id, name, "error", output))
                _finish_tool_trace(recorder, trace_handle, output, executed=False)
                continue

            raw_arguments = call.get("args")
            if not isinstance(raw_arguments, Mapping):
                output = _tool_call_error(
                    "invalid_tool_arguments",
                    "Tool arguments must be an object.",
                )
                results.append(ToolCallResult(call_id, name, "error", output))
                _finish_tool_trace(recorder, trace_handle, output, executed=False)
                continue

            try:
                arguments = _validate_arguments(registration, raw_arguments)
            except Exception:  # noqa: BLE001 - schemas may raise custom errors
                output = _tool_call_error(
                    "invalid_tool_arguments",
                    "Tool arguments do not match the required schema.",
                )
                results.append(ToolCallResult(call_id, name, "error", output))
                _finish_tool_trace(recorder, trace_handle, output, executed=False)
                continue

            tool_calls = getattr(context, "tool_calls", None)
            followup_searches = getattr(context, "followup_searches", None)
            if isinstance(tool_calls, bool) or not isinstance(tool_calls, int):
                raise TypeError("context.tool_calls must be an integer")
            if (
                isinstance(followup_searches, bool)
                or not isinstance(followup_searches, int)
            ):
                raise TypeError("context.followup_searches must be an integer")

            if tool_calls >= max_tool_calls:
                output = _tool_call_error(
                    "tool_budget_exhausted",
                    "The tool-call budget for this turn has been exhausted.",
                )
                results.append(ToolCallResult(call_id, name, "error", output))
                _finish_tool_trace(recorder, trace_handle, output, executed=False)
                continue

            if (
                name == "search_knowledge"
                and max_followup_searches is not None
                and followup_searches >= max_followup_searches
            ):
                output = _tool_call_error(
                    "followup_search_budget_exhausted",
                    "The follow-up search budget for this turn has been exhausted.",
                )
                results.append(ToolCallResult(call_id, name, "error", output))
                _finish_tool_trace(recorder, trace_handle, output, executed=False)
                continue

            setattr(context, "tool_calls", tool_calls + 1)
            if name == "search_knowledge":
                setattr(context, "followup_searches", followup_searches + 1)

            output = _execute_registration(registration, context, arguments)
            results.append(_tool_call_result(call_id, name, output))
            _finish_tool_trace(recorder, trace_handle, output, executed=True)

        return tuple(results)

def create_tool_registry(
    definitions: Iterable[BaseTool],
    executors: Mapping[str, ToolExecutor[ExecutionContextT]],
) -> InMemoryToolRegistry[ExecutionContextT]:
    """构建同时用于模型绑定与执行的单一注册器。

    这个小型适配器让现有工具工厂继续分别返回定义和执行器，
    同时让 Agent 逐步迁移到注册器。两个集合必须描述完全相同的工具，
    防止某个工具暴露给模型却没有执行器（或相反）。
    """
    tool_definitions = tuple(definitions)
    definition_names = [definition.name for definition in tool_definitions]
    if len(definition_names) != len(set(definition_names)):
        raise ValueError("tool definitions must have unique names")

    executor_names = set(executors)
    if set(definition_names) != executor_names:
        raise ValueError("tool definitions and executors must have matching names")

    return InMemoryToolRegistry(
        ToolRegistration(
            name=definition.name,
            definition=definition,
            executor=executors[definition.name],
        )
        for definition in tool_definitions
    )


def combine_tool_registries(
    *registries: ToolRegistry[ExecutionContextT],
) -> InMemoryToolRegistry[ExecutionContextT]:
    """合并完整注册项，并拒绝重名。

    复制注册项而不只是定义，可保留自定义参数校验器，
    并让模型可见性与服务端执行保持一致。
    """
    combined: InMemoryToolRegistry[ExecutionContextT] = InMemoryToolRegistry()
    for registry in registries:
        for registration in registry.list_registrations():
            combined.register(registration)
    return combined


# 第二阶段：开发完成后统一验证，本阶段不执行。

# TODO 08：验证工具筛选、参数错误、重复调用与预算边界。
# 使用可控执行函数确认被拒绝的调用没有副作用，失败调用不会绕过次数上限。
# 确认隐藏工具不能被手工构造的模型调用执行。

# TODO 09：验证原有知识库问答行为。
# 对比迁移前后的正常回答、补充检索、引用和调用次数，覆盖 CLI 与 API 入口。
