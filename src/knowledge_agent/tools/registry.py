"""工具注册与统一执行模块（开发中）。

模块功能：管理提供给模型的工具定义，以及服务端实际执行的函数。
输入：工具定义、执行函数、模型工具调用、当前问答轮状态与调用预算。
输出：模型可见的工具列表，以及关联 tool_call_id 的结构化调用结果。

当前接口不替换 agent.py 现有注册和执行逻辑。
第一版复用现有 LangChain StructuredTool 与参数校验方式。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableSet
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
    """Callable implementing a tool against one turn's execution context."""

    def __call__(
        self,
        context: ExecutionContextT,
        /,
        **arguments: Any,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class ToolRegistration(Generic[ExecutionContextT]):
    """A model-visible tool definition paired with its server-side executor."""

    name: str
    definition: BaseTool
    executor: ToolExecutor[ExecutionContextT]


ToolCallStatus = Literal["success", "error"]


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """The outcome of one model-requested tool call, in request order."""

    tool_call_id: str | None
    name: str | None
    status: ToolCallStatus
    output: Any


def _tool_call_error(code: str, message: str) -> dict[str, Any]:
    return {"status": "error", "error": {"code": code, "message": message}}


@runtime_checkable
class ToolRegistry(Protocol[ExecutionContextT]):
    """Interface for registering, exposing, and executing approved tools."""

    def register(self, registration: ToolRegistration[ExecutionContextT]) -> None:
        """Add one tool registration."""

    def get(self, name: str) -> ToolRegistration[ExecutionContextT] | None:
        """Return the registration for ``name``, if present."""

    def list_definitions(self) -> tuple[BaseTool, ...]:
        """Return model-visible definitions in registration order."""

    def filter_for_skill(
        self,
        allowed_tools: Iterable[str] | None = None,
    ) -> ToolRegistry[ExecutionContextT]:
        """Return a registry restricted to the tools available this turn."""

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        context: ExecutionContextT,
    ) -> Any:
        """Execute a registered tool within the supplied turn-local context."""

    def execute_tool_calls(
        self,
        calls: Iterable[object],
        *,
        context: ExecutionContextT,
        completed_tool_call_ids: MutableSet[str],
    ) -> tuple[ToolCallResult, ...]:
        """Validate and execute model-requested calls in request order."""


# 第一阶段：注册与执行开发。

class InMemoryToolRegistry(Generic[ExecutionContextT]):
    """Fast, insertion-ordered registry of explicitly approved tools."""

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

        self._registrations[name] = registration

    def get(self, name: str) -> ToolRegistration[ExecutionContextT] | None:
        return self._registrations.get(name)

    def list_definitions(self) -> tuple[BaseTool, ...]:
        return tuple(
            registration.definition for registration in self._registrations.values()
        )

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
        return registration.executor(context, **arguments)

    def execute_tool_calls(
        self,
        calls: Iterable[object],
        *,
        context: ExecutionContextT,
        completed_tool_call_ids: MutableSet[str],
    ) -> tuple[ToolCallResult, ...]:
        results: list[ToolCallResult] = []

        for call in calls:
            if not isinstance(call, Mapping):
                results.append(
                    ToolCallResult(
                        tool_call_id=None,
                        name=None,
                        status="error",
                        output=_tool_call_error(
                            "invalid_tool_call", "The tool call must be an object."
                        ),
                    )
                )
                continue

            raw_id = call.get("id")
            call_id = raw_id.strip() if isinstance(raw_id, str) else None
            raw_name = call.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) else None

            if not call_id:
                results.append(
                    ToolCallResult(
                        tool_call_id=None,
                        name=name,
                        status="error",
                        output=_tool_call_error(
                            "invalid_tool_call_id",
                            "The tool call ID must be a non-empty string.",
                        ),
                    )
                )
                continue

            if call_id in completed_tool_call_ids:
                results.append(
                    ToolCallResult(
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                        output=_tool_call_error(
                            "duplicate_tool_call_id",
                            "This tool call ID has already been handled in the current turn.",
                        ),
                    )
                )
                continue

            # A usable ID is terminally handled even if later validation fails.
            completed_tool_call_ids.add(call_id)

            registration = self._registrations.get(name) if name else None
            if registration is None:
                results.append(
                    ToolCallResult(
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                        output=_tool_call_error(
                            "unknown_tool", "The requested tool is not available."
                        ),
                    )
                )
                continue

            raw_arguments = call.get("args")
            if not isinstance(raw_arguments, Mapping):
                results.append(
                    ToolCallResult(
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                        output=_tool_call_error(
                            "invalid_tool_arguments",
                            "Tool arguments must be an object.",
                        ),
                    )
                )
                continue

            try:
                arguments = (
                    registration.definition.get_input_schema()
                    .model_validate(dict(raw_arguments))
                    .model_dump()
                )
            except Exception:  # noqa: BLE001 - schemas may raise custom errors
                results.append(
                    ToolCallResult(
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                        output=_tool_call_error(
                            "invalid_tool_arguments",
                            "Tool arguments do not match the required schema.",
                        ),
                    )
                )
                continue

            try:
                output = registration.executor(context, **arguments)
            except Exception:  # noqa: BLE001 - tool failures are isolated per call
                output = _tool_call_error(
                    "tool_execution_failed",
                    "The requested tool could not be completed.",
                )

            status: ToolCallStatus = (
                "error"
                if isinstance(output, Mapping) and output.get("status") == "error"
                else "success"
            )
            results.append(
                ToolCallResult(
                    tool_call_id=call_id,
                    name=name,
                    status=status,
                    output=output,
                )
            )

        return tuple(results)

# TODO 05：统一管理工具调用次数与补充检索次数。
# 将两个知识库 executor 内自行累加的逻辑迁移到统一执行入口，避免新增工具漏计。
# 明确口径：参数合法且获准开始执行时计一次，执行失败也计入预算。
# 无效参数、未知工具和被预算拒绝的调用不实际执行；重复 ID 不重复计数。
# search_knowledge 继续遵守 max_followup_searches，全部工具遵守 max_tool_calls。
# 同步梳理 prepare_question 的首次检索路径，保留原有计数语义，避免遗漏或双计。

# TODO 06：规范工具返回值和执行错误。
# 复用现有 ToolCallResult 的调用 ID、名称、状态、输出语义，必要时提取共享结构。
# 向模型提供可处理的错误信息，日志不输出 API 密钥或完整用户问答正文。
# 搜索失败、没有命中和达到调用上限分别处理，不混用为“资料不足”。

# TODO 07：接入现有 Agent 工具循环。
# 让工具绑定和 execute_tool_calls 使用同一份注册器配置，保留现有模型循环。
# KnowledgeTools 等旧入口如需过渡，使用兼容适配，避免一次迁移破坏 CLI/API。
# 本文件仅规划接入；实现本模块时不要提前删除旧工具函数。


# 第二阶段：开发完成后统一验证，本阶段不执行。

# TODO 08：验证工具筛选、参数错误、重复调用与预算边界。
# 使用可控执行函数确认被拒绝的调用没有副作用，失败调用不会绕过次数上限。
# 确认隐藏工具不能被手工构造的模型调用执行。

# TODO 09：验证原有知识库问答行为。
# 对比迁移前后的正常回答、补充检索、引用和调用次数，覆盖 CLI 与 API 入口。
