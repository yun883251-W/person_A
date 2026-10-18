"""将一个不可变的 MCP 工具快照适配到共享工具注册器。

适配器借用已配置好的运行时。它绝不启动、关闭或重新发现该运行时，
因此每个注册器都是某一轮次的工具快照。
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeVar

from langchain_core.tools import StructuredTool

from ..tools.registry import InMemoryToolRegistry, ToolRegistration
from .client import McpToolSpec

_PUBLIC_NAME_RE = re.compile(r"^mcp__[A-Za-z0-9_-]{1,64}__[A-Za-z0-9_-]{1,64}$")
_DRAFT_2020_12_URIS = frozenset(
    {
        "https://json-schema.org/draft/2020-12/schema",
        "https://json-schema.org/draft/2020-12/schema#",
    }
)
_MAX_SCHEMA_CHARS = 32_000
_MAX_ARGUMENT_CHARS = 128_000
_MAX_JSON_DEPTH = 64
_MAX_DESCRIPTION_CHARS = 4_000
_EXTERNAL_RESULT_NOTICE = (
    " Results come from an external MCP service, not from the local knowledge "
    "base, and do not provide local source-line citations."
)

ExecutionContextT = TypeVar("ExecutionContextT")


class McpRuntimePort(Protocol):
    """该适配器使用的同步运行时接口。"""

    def list_tools(self) -> Sequence[McpToolSpec]: ...

    def call_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]: ...


class McpAdapterError(ValueError):
    """稳定、安全的注册错误，不包含远端 schema 内容。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _adapter_error(code: str, message: str) -> McpAdapterError:
    return McpAdapterError(code, message)


def _safe_error(code: str, message: str) -> dict[str, Any]:
    return {"status": "error", "error": {"code": code, "message": message}}


def _copy_bounded_json(
    value: Any,
    *,
    max_chars: int,
    invalid_code: str,
) -> Any:
    """在大小/深度/取值检查后返回保留类型的 JSON 副本。"""

    def visit(item: Any, depth: int) -> None:
        if depth > _MAX_JSON_DEPTH:
            raise _adapter_error(invalid_code, "JSON nesting exceeds the allowed limit.")
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise _adapter_error(invalid_code, "JSON contains a non-finite number.")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise _adapter_error(invalid_code, "JSON object keys must be strings.")
                visit(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)
            return
        raise _adapter_error(invalid_code, "Value is not representable as JSON.")

    visit(value, 0)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise _adapter_error(invalid_code, "Value is not representable as JSON.") from exc
    if len(encoded) > max_chars:
        raise _adapter_error(invalid_code, "JSON exceeds the allowed size limit.")
    return copy.deepcopy(value)


def _reject_external_references(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef"} and (
                not isinstance(child, str) or not child.startswith("#")
            ):
                raise _adapter_error(
                    "unsupported_schema_reference",
                    "MCP schemas may only use local fragment references.",
                )
            _reject_external_references(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_external_references(child)


def _compile_argument_validator(raw_schema: Mapping[str, Any]):
    """编译 Draft 2020-12 校验器，仅使用无网络的本地引用。"""
    try:
        schema = _copy_bounded_json(
            raw_schema,
            max_chars=_MAX_SCHEMA_CHARS,
            invalid_code="invalid_tool_schema",
        )
        if not isinstance(schema, dict):
            raise _adapter_error(
                "invalid_tool_schema", "MCP input schema must be a JSON object."
            )
        dialect = schema.get("$schema")
        if dialect is not None and dialect not in _DRAFT_2020_12_URIS:
            raise _adapter_error(
                "unsupported_schema_dialect",
                "Only JSON Schema Draft 2020-12 is supported.",
            )
        _reject_external_references(schema)
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
        compiled = Draft202012Validator(schema)
    except McpAdapterError:
        raise
    except ImportError as exc:
        raise _adapter_error(
            "schema_validator_unavailable",
            "MCP JSON Schema validation is not installed.",
        ) from exc
    except Exception as exc:
        raise _adapter_error(
            "invalid_tool_schema", "MCP tool input schema is invalid."
        ) from exc

    def validate(arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            copied = _copy_bounded_json(
                arguments,
                max_chars=_MAX_ARGUMENT_CHARS,
                invalid_code="invalid_tool_arguments",
            )
            if not isinstance(copied, dict):
                raise _adapter_error(
                    "invalid_tool_arguments", "Tool arguments must be a JSON object."
                )
            compiled.validate(copied)
            return copied
        except McpAdapterError:
            raise
        except Exception as exc:
            raise _adapter_error(
                "invalid_tool_arguments",
                "Tool arguments do not match the required schema.",
            ) from exc

    return schema, validate


def _safe_description(description: object) -> str:
    text = description if isinstance(description, str) else ""
    text = " ".join(text.split())
    available = _MAX_DESCRIPTION_CHARS - len(_EXTERNAL_RESULT_NOTICE)
    if len(text) > available:
        text = text[:available].rstrip()
    return (text or "Call an approved external MCP tool.") + _EXTERNAL_RESULT_NOTICE


def _call_runtime(
    runtime: McpRuntimePort,
    public_name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        result = runtime.call_tool(public_name, arguments)
    except Exception:  # noqa: BLE001 - never expose remote/runtime diagnostics
        return _safe_error("connection_lost", "The MCP service connection was lost.")
    if not isinstance(result, dict):
        return _safe_error(
            "invalid_tool_result", "The MCP service returned an invalid result."
        )
    return result


def _registration_for_spec(
    runtime: McpRuntimePort,
    spec: McpToolSpec,
) -> ToolRegistration[ExecutionContextT]:
    public_name = spec.name
    if not isinstance(public_name, str) or _PUBLIC_NAME_RE.fullmatch(public_name) is None:
        raise _adapter_error("invalid_tool_name", "MCP tool has an invalid public name.")
    if not isinstance(spec.remote_name, str) or not spec.remote_name:
        raise _adapter_error(
            "invalid_remote_tool_name", "MCP tool has an invalid remote name."
        )

    schema, validate = _compile_argument_validator(spec.input_schema)

    def invoke_definition(**arguments: Any) -> dict[str, Any]:
        """直接调用使用与注册器相同的校验和运行时路径。"""
        try:
            checked = validate(arguments)
        except Exception:
            return _safe_error(
                "invalid_tool_arguments",
                "Tool arguments do not match the required schema.",
            )
        return _call_runtime(runtime, public_name, checked)

    definition = StructuredTool.from_function(
        func=invoke_definition,
        name=public_name,
        description=_safe_description(spec.description),
        args_schema=copy.deepcopy(schema),
        infer_schema=False,
    )

    def execute(
        context: ExecutionContextT,  # noqa: ARG001 - required registry protocol
        /,
        **arguments: Any,
    ) -> dict[str, Any]:
        return _call_runtime(runtime, public_name, arguments)

    return ToolRegistration(
        name=public_name,
        definition=definition,
        executor=execute,
        argument_validator=validate,
    )


def create_mcp_tool_registry(
    runtime: McpRuntimePort,
) -> InMemoryToolRegistry[ExecutionContextT]:
    """根据运行时当前已批准工具的快照创建注册器。

    此过程不执行任何生命周期操作。未配置、已禁用、已关闭或尚未就绪的
    运行时自然产生空快照。运行时变化后，调用方需为下一轮重新构建注册器。
    """
    try:
        snapshot = tuple(runtime.list_tools())
    except Exception as exc:
        raise _adapter_error(
            "tool_snapshot_unavailable", "MCP tool snapshot is unavailable."
        ) from exc

    registrations = tuple(_registration_for_spec(runtime, spec) for spec in snapshot)
    try:
        return InMemoryToolRegistry(registrations)
    except (TypeError, ValueError) as exc:
        raise _adapter_error(
            "duplicate_tool_name", "MCP tool names must be unique."
        ) from exc


__all__ = ["McpAdapterError", "McpRuntimePort", "create_mcp_tool_registry"]
