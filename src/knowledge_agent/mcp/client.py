"""Bounded synchronous facade over optional local stdio MCP servers.

The MCP SDK is imported only when an enabled runtime starts, so ordinary
knowledge queries do not depend on the optional MCP packages.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import json
import math
import os
import re
import threading
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_SERVERS = 16
_MAX_ALLOWED_TOOLS = 128
_MAX_DISCOVERED_TOOLS = 256
_MAX_TOOL_PAGES = 20
_MAX_SCHEMA_CHARS = 32_000
_MAX_DESCRIPTION_CHARS = 4_000
_MAX_ARGUMENT_CHARS = 128_000
_MAX_TIMEOUT_SECONDS = 300.0
_MAX_OUTPUT_CHARS = 1_000_000
_STOP = object()
_DRAFT_2020_12_URIS = frozenset({
    "https://json-schema.org/draft/2020-12/schema",
    "https://json-schema.org/draft/2020-12/schema#",
})


class McpConfigurationError(ValueError):
    """Safe configuration error which never includes configuration contents."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class McpServerState(str, Enum):
    DISABLED = "disabled"
    STARTING = "starting"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    server_name: str
    enabled: bool
    command: str
    args: tuple[str, ...]
    cwd: Path | None
    env_from: Mapping[str, str]
    allowed_tools: tuple[str, ...]
    startup_timeout_seconds: float
    call_timeout_seconds: float
    max_output_chars: int


@dataclass(frozen=True, slots=True)
class McpToolSpec:
    name: str
    server_name: str
    remote_name: str
    description: str
    input_schema: Mapping[str, Any]
    annotations: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class McpServerStatus:
    server_name: str
    state: McpServerState
    error_code: str | None = None

    @property
    def status(self) -> McpServerState:
        """Compatibility alias for callers that describe the field as status."""
        return self.state


@dataclass(slots=True)
class _CallRequest:
    public_name: str
    remote_name: str
    arguments: dict[str, Any]
    deadline: float
    response: asyncio.Future[dict[str, Any]]


@dataclass(slots=True)
class _ServerRuntime:
    config: McpServerConfig
    queue: asyncio.Queue[Any]
    tools: dict[str, McpToolSpec] = field(default_factory=dict)


def _config_error(code: str, message: str) -> McpConfigurationError:
    return McpConfigurationError(code, message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _config_error("duplicate_json_key", "MCP 配置包含重复的 JSON 键")
        result[key] = value
    return result


def _positive_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _config_error("invalid_config", f"{field_name} 必须是正数")
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= _MAX_TIMEOUT_SECONDS:
        raise _config_error("invalid_config", f"{field_name} 必须在有效范围内")
    return number


def _name(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _NAME_RE.fullmatch(value) is None:
        raise _config_error("invalid_config", f"{field_name} 格式无效")
    return value


def _only_keys(value: Mapping[str, Any], allowed: set[str], location: str) -> None:
    if any(not isinstance(key, str) for key in value) or set(value) - allowed:
        raise _config_error("invalid_config", f"{location} 包含未知字段")


def load_mcp_config(config_path: Path | str) -> tuple[McpServerConfig, ...]:
    """Read and strictly validate a version-1 MCP configuration."""
    path = Path(config_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise _config_error("config_not_found", "MCP 配置文件不存在")
    try:
        with path.open(encoding="utf-8-sig") as stream:
            document = json.load(stream, object_pairs_hook=_unique_object)
    except McpConfigurationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise _config_error("invalid_config", "无法读取 MCP 配置文件") from None
    if not isinstance(document, dict):
        raise _config_error("invalid_config", "MCP 配置顶层必须是对象")
    _only_keys(document, {"version", "servers"}, "MCP 配置")
    version = document.get("version")
    if isinstance(version, bool) or version != 1:
        raise _config_error("unsupported_config_version", "不支持该 MCP 配置版本")
    servers = document.get("servers")
    if not isinstance(servers, dict):
        raise _config_error("invalid_config", "servers 必须是对象")
    if len(servers) > _MAX_SERVERS:
        raise _config_error("invalid_config", "MCP Server 数量超过限制")

    fields = {
        "enabled", "command", "args", "cwd", "env_from", "allowed_tools",
        "startup_timeout_seconds", "call_timeout_seconds", "max_output_chars",
    }
    configs: list[McpServerConfig] = []
    for raw_name, raw in servers.items():
        server_name = _name(raw_name, "server_name")
        if not isinstance(raw, dict):
            raise _config_error("invalid_config", "Server 配置必须是对象")
        _only_keys(raw, fields, f"Server {server_name}")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise _config_error("invalid_config", "enabled 必须是布尔值")
        command = raw.get("command", "")
        if not isinstance(command, str) or not command.strip() or "\0" in command:
            raise _config_error("invalid_config", "command 必须是单个程序名或路径")
        args = raw.get("args", [])
        if (not isinstance(args, list) or len(args) > 64 or any(
            not isinstance(item, str) or "\0" in item or len(item) > 4096 for item in args
        )):
            raise _config_error("invalid_config", "args 必须是有效字符串数组")
        raw_cwd = raw.get("cwd")
        if raw_cwd is not None and (not isinstance(raw_cwd, str) or not raw_cwd.strip()):
            raise _config_error("invalid_config", "cwd 必须是非空路径")
        cwd: Path | None = None
        if raw_cwd is not None:
            cwd = Path(raw_cwd).expanduser()
            if not cwd.is_absolute():
                cwd = path.parent / cwd
            cwd = cwd.resolve(strict=False)
            if not cwd.is_dir():
                raise _config_error("invalid_cwd", "MCP Server cwd 不存在或不是目录")
        env_from = raw.get("env_from", {})
        if not isinstance(env_from, dict) or len(env_from) > 64:
            raise _config_error("invalid_config", "env_from 必须是对象")
        env_pairs: list[tuple[str, str]] = []
        for child_name, host_name in env_from.items():
            if (not isinstance(child_name, str) or _ENV_NAME_RE.fullmatch(child_name) is None
                    or not isinstance(host_name, str) or _ENV_NAME_RE.fullmatch(host_name) is None):
                raise _config_error("invalid_config", "env_from 变量名无效")
            env_pairs.append((child_name, host_name))
        allowed = raw.get("allowed_tools")
        if (not isinstance(allowed, list) or len(allowed) > _MAX_ALLOWED_TOOLS
                or any(not isinstance(item, str) for item in allowed)):
            raise _config_error("invalid_config", "allowed_tools 必须显式提供字符串数组")
        allowed_tools = tuple(_name(item, "allowed_tools") for item in allowed)
        if len(set(allowed_tools)) != len(allowed_tools):
            raise _config_error("invalid_config", "allowed_tools 不能重复")
        if any(len(f"mcp__{server_name}__{item}") > 64 for item in allowed_tools):
            raise _config_error("invalid_config", "模型工具名超过长度限制")
        output_limit = raw.get("max_output_chars", 12_000)
        if (isinstance(output_limit, bool) or not isinstance(output_limit, int)
                or not 1 <= output_limit <= _MAX_OUTPUT_CHARS):
            raise _config_error("invalid_config", "max_output_chars 必须是有效正整数")
        configs.append(McpServerConfig(
            server_name=server_name, enabled=enabled, command=command.strip(),
            args=tuple(args), cwd=cwd, env_from=MappingProxyType(dict(env_pairs)),
            allowed_tools=allowed_tools,
            startup_timeout_seconds=_positive_number(
                raw.get("startup_timeout_seconds", 15), "startup_timeout_seconds"),
            call_timeout_seconds=_positive_number(
                raw.get("call_timeout_seconds", 30), "call_timeout_seconds"),
            max_output_chars=output_limit,
        ))
    return tuple(configs)


def _error(code: str, message: str) -> dict[str, Any]:
    return {"status": "error", "error": {"code": code, "message": message}}


def _environment(config: McpServerConfig) -> dict[str, str]:
    base_names = ("PATH", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "HOME")
    result = {name: os.environ[name] for name in base_names if name in os.environ}
    for child_name, host_name in config.env_from.items():
        if host_name not in os.environ:
            raise KeyError(host_name)
        result[child_name] = os.environ[host_name]
    return result


def _plain_json(value: Any, depth: int = 0) -> Any:
    if depth > 20:
        raise ValueError("too_deep")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non_finite")
        return value
    if isinstance(value, Mapping):
        if len(value) > 1000:
            raise ValueError("too_many_items")
        if any(not isinstance(key, str) for key in value):
            raise ValueError("non_string_key")
        return {key: _plain_json(item, depth + 1) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 1000:
            raise ValueError("too_many_items")
        return [_plain_json(item, depth + 1) for item in value]
    if hasattr(value, "model_dump"):
        return _plain_json(value.model_dump(mode="json"), depth + 1)
    raise ValueError("not_json")


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def _field(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    return default


def _has_external_schema_reference(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef"} and (
                not isinstance(child, str) or not child.startswith("#")
            ):
                return True
            if _has_external_schema_reference(child):
                return True
    elif isinstance(value, list):
        return any(_has_external_schema_reference(child) for child in value)
    return False


def _checked_schema(schema: Any) -> dict[str, Any]:
    try:
        plain = _plain_json(schema)
        if not isinstance(plain, dict) or _json_size(plain) > _MAX_SCHEMA_CHARS:
            raise ValueError("invalid_schema")
        dialect = plain.get("$schema")
        if dialect is not None and dialect not in _DRAFT_2020_12_URIS:
            raise ValueError("unsupported_schema_dialect")
        if _has_external_schema_reference(plain):
            raise ValueError("external_schema_reference")
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(plain)
        return plain
    except Exception as exc:
        raise ValueError("invalid_schema") from exc


def _checked_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> dict[str, Any]:
    try:
        plain = _plain_json(arguments)
        if not isinstance(plain, dict) or _json_size(plain) > _MAX_ARGUMENT_CHARS:
            raise ValueError("invalid_arguments")
        from jsonschema import Draft202012Validator
        Draft202012Validator(dict(schema)).validate(plain)
        return plain
    except Exception as exc:
        raise ValueError("invalid_arguments") from exc


class McpRuntime:
    """Own enabled MCP subprocesses and expose a safe synchronous interface."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        self._configs = () if config_path is None else load_mcp_config(config_path)
        self._lock = threading.RLock()
        self._statuses = {
            item.server_name: McpServerStatus(
                item.server_name,
                McpServerState.DISABLED if not item.enabled else McpServerState.CLOSED)
            for item in self._configs
        }
        self._tools: dict[str, McpToolSpec] = {}
        self._servers: dict[str, _ServerRuntime] = {}
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._startup_event = threading.Event()
        self._accepting = False
        self._started = False
        self._closing = False

    def start(self) -> None:
        with self._lock:
            if self._started and not self._closing:
                return
            if self._closing:
                raise RuntimeError("MCP runtime is closing")
            enabled = tuple(item for item in self._configs if item.enabled)
            self._started = True
            if not enabled:
                self._accepting = True
                return
            self._startup_event.clear()
            for config in enabled:
                self._set_status(config.server_name, McpServerState.STARTING)
            self._thread = threading.Thread(
                target=self._thread_main, args=(enabled,), name="mcp-runtime", daemon=True)
            self._thread.start()
        if not self._startup_event.wait(max(x.startup_timeout_seconds for x in enabled) + 2):
            self.close()
            raise RuntimeError("MCP runtime startup timed out")
        with self._lock:
            self._accepting = True

    def close(self) -> None:
        with self._lock:
            self._accepting = False
            self._closing = True
            thread = self._thread
            if self._loop is not None and self._stop_event is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._stop_event.set)
        if thread is not None and thread is not threading.current_thread():
            maximum = max(
                (max(item.call_timeout_seconds, item.startup_timeout_seconds)
                 for item in self._configs if item.enabled),
                default=1.0,
            )
            thread.join(timeout=maximum + 2.0)
        with self._lock:
            self._tools.clear()
            self._servers.clear()
            for item in self._configs:
                state = McpServerState.DISABLED if not item.enabled else McpServerState.CLOSED
                self._statuses[item.server_name] = McpServerStatus(item.server_name, state)
            self._thread = None
            self._loop = None
            self._stop_event = None
            self._started = False
            self._closing = False

    def __enter__(self) -> "McpRuntime":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def list_tools(self) -> tuple[McpToolSpec, ...]:
        with self._lock:
            return tuple(McpToolSpec(
                spec.name, spec.server_name, spec.remote_name, spec.description,
                copy.deepcopy(dict(spec.input_schema)),
                copy.deepcopy(dict(spec.annotations)) if spec.annotations else None,
            ) for spec in sorted(self._tools.values(), key=lambda item: item.name))

    def list_server_statuses(self) -> tuple[McpServerStatus, ...]:
        with self._lock:
            return tuple(self._statuses[name] for name in sorted(self._statuses))

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not any(item.enabled for item in self._configs):
                return _error("disabled", "MCP 功能未启用")
            if not self._accepting or self._closing:
                return _error("unavailable", "MCP 运行时当前不可用")
            spec = self._tools.get(name) if isinstance(name, str) else None
            if spec is None:
                return _error("unknown_tool", "未找到或未授权该 MCP 工具")
            server = self._servers.get(spec.server_name)
            status = self._statuses.get(spec.server_name)
            loop = self._loop
            if (server is None or loop is None or status is None
                    or status.state is not McpServerState.READY):
                return _error("unavailable", "MCP Server 当前不可用")
            timeout = server.config.call_timeout_seconds
        if not isinstance(arguments, Mapping):
            return _error("invalid_arguments", "工具参数必须是 JSON 对象")
        try:
            checked = _checked_arguments(spec.input_schema, arguments)
        except ValueError:
            return _error("invalid_arguments", "工具参数未通过校验")
        deadline = time.monotonic() + timeout
        pending = asyncio.run_coroutine_threadsafe(
            self._submit(server, spec.name, spec.remote_name, checked, deadline), loop)
        try:
            return pending.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            pending.cancel()
            return _error("timeout", "MCP 工具调用超时，远端操作结果未知")
        except Exception:
            return _error("connection_lost", "MCP Server 连接已中断")

    async def _submit(self, server: _ServerRuntime, public_name: str, remote_name: str,
                      arguments: dict[str, Any], deadline: float) -> dict[str, Any]:
        response: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        await server.queue.put(_CallRequest(public_name, remote_name, arguments, deadline, response))
        try:
            return await asyncio.shield(response)
        except asyncio.CancelledError:
            response.cancel()
            raise

    def _thread_main(self, enabled: tuple[McpServerConfig, ...]) -> None:
        try:
            asyncio.run(self._async_main(enabled))
        except Exception:
            for config in enabled:
                current = self._statuses.get(config.server_name)
                if current is None or current.state in {McpServerState.STARTING, McpServerState.READY}:
                    self._set_status(config.server_name, McpServerState.UNAVAILABLE, "runtime_error")
        finally:
            self._startup_event.set()

    async def _async_main(self, enabled: tuple[McpServerConfig, ...]) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        started: asyncio.Queue[str] = asyncio.Queue()
        tasks: list[asyncio.Task[None]] = []
        for config in enabled:
            server = _ServerRuntime(config, asyncio.Queue())
            self._servers[config.server_name] = server
            tasks.append(asyncio.create_task(self._server_owner(server, started)))
        for _ in enabled:
            await started.get()
        self._startup_event.set()
        await self._stop_event.wait()
        for server in tuple(self._servers.values()):
            await server.queue.put(_STOP)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _server_owner(self, server: _ServerRuntime, started: asyncio.Queue[str]) -> None:
        config = server.config
        reported = False
        try:
            try:
                environment = _environment(config)
            except KeyError:
                self._set_status(config.server_name, McpServerState.UNAVAILABLE, "missing_environment")
                return
            try:
                import anyio
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client
                from jsonschema import Draft202012Validator as _Validator  # noqa: F401
            except ImportError:
                self._set_status(config.server_name, McpServerState.UNAVAILABLE, "optional_dependency_missing")
                return
            async with AsyncExitStack() as stack:
                owner_task = asyncio.current_task()
                startup_expired = False

                def cancel_startup() -> None:
                    nonlocal startup_expired
                    startup_expired = True
                    if owner_task is not None:
                        owner_task.cancel()

                timeout_handle = asyncio.get_running_loop().call_later(
                    config.startup_timeout_seconds, cancel_startup)
                try:
                    stderr = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
                    streams = await stack.enter_async_context(stdio_client(
                        StdioServerParameters(command=config.command, args=list(config.args),
                                              env=environment, cwd=config.cwd),
                        errlog=stderr))
                    session = await stack.enter_async_context(ClientSession(*streams))
                    await session.initialize()
                    await self._discover_tools(server, session)
                except asyncio.CancelledError:
                    if not startup_expired:
                        raise
                    self._set_status(config.server_name, McpServerState.UNAVAILABLE, "startup_timeout")
                    return
                except Exception:
                    self._set_status(config.server_name, McpServerState.UNAVAILABLE, "startup_failed")
                    return
                finally:
                    timeout_handle.cancel()
                self._set_status(config.server_name, McpServerState.READY)
                await started.put(config.server_name)
                reported = True
                while True:
                    item = await server.queue.get()
                    if item is _STOP:
                        break
                    if not isinstance(item, _CallRequest) or item.response.cancelled():
                        continue
                    if self._closing:
                        self._finish(item.response, _error("unavailable", "MCP 运行时正在关闭"))
                        continue
                    remaining = item.deadline - time.monotonic()
                    if remaining <= 0:
                        self._finish(item.response, _error("timeout", "MCP 工具调用超时，远端操作结果未知"))
                        continue
                    try:
                        with anyio.fail_after(remaining):
                            result = await session.call_tool(item.remote_name, item.arguments)
                        converted = self._convert_result(config, item.public_name, result)
                    except TimeoutError:
                        converted = _error("timeout", "MCP 工具调用超时，远端操作结果未知")
                    except Exception:
                        converted = _error("connection_lost", "MCP Server 连接已中断")
                        self._remove_server_tools(config.server_name)
                        self._set_status(config.server_name, McpServerState.UNAVAILABLE, "connection_lost")
                    self._finish(item.response, converted)
        finally:
            self._remove_server_tools(config.server_name)
            if not reported:
                await started.put(config.server_name)

    async def _discover_tools(self, server: _ServerRuntime, session: Any) -> None:
        cursor: str | None = None
        discovered = 0
        approved = set(server.config.allowed_tools)
        for _ in range(_MAX_TOOL_PAGES):
            result = await session.list_tools(cursor=cursor)
            raw_tools = _field(result, "tools", default=[])
            if (not isinstance(raw_tools, Sequence)
                    or isinstance(raw_tools, (str, bytes, bytearray))):
                raise ValueError("invalid_tool_list")
            for raw_tool in raw_tools:
                discovered += 1
                if discovered > _MAX_DISCOVERED_TOOLS:
                    raise ValueError("too_many_tools")
                remote_name = _field(raw_tool, "name")
                if remote_name not in approved or not isinstance(remote_name, str):
                    continue
                public_name = f"mcp__{server.config.server_name}__{remote_name}"
                if (_NAME_RE.fullmatch(remote_name) is None or len(public_name) > 64
                        or _NAME_RE.fullmatch(public_name) is None):
                    continue
                description = _field(raw_tool, "description", default="")
                if description is None:
                    description = ""
                if not isinstance(description, str) or len(description) > _MAX_DESCRIPTION_CHARS:
                    continue
                try:
                    schema = _checked_schema(_field(raw_tool, "inputSchema", "input_schema", default={}))
                except ValueError:
                    continue
                annotations = None
                raw_annotations = _field(raw_tool, "annotations")
                if raw_annotations is not None:
                    try:
                        candidate = _plain_json(raw_annotations)
                        if isinstance(candidate, dict) and _json_size(candidate) <= 4_000:
                            annotations = candidate
                    except ValueError:
                        pass
                spec = McpToolSpec(public_name, server.config.server_name, remote_name,
                                   description, schema, annotations)
                with self._lock:
                    if public_name in self._tools:
                        raise ValueError("tool_name_collision")
                    self._tools[public_name] = spec
                    server.tools[public_name] = spec
            next_cursor = _field(result, "nextCursor", "next_cursor")
            if next_cursor is None:
                return
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor:
                raise ValueError("invalid_cursor")
            cursor = next_cursor
        raise ValueError("too_many_tool_pages")

    def _convert_result(self, config: McpServerConfig, public_name: str,
                        result: Any) -> dict[str, Any]:
        if bool(_field(result, "isError", "is_error", default=False)):
            return _error("remote_error", "MCP 工具返回错误")
        content = _field(result, "content", default=[])
        if (not isinstance(content, Sequence)
                or isinstance(content, (str, bytes, bytearray))):
            return _error("remote_error", "MCP 工具返回了无效结果")
        text_parts: list[str] = []
        text_length = 0
        truncated = False
        ignored = 0
        for block in list(content)[:100]:
            if _field(block, "type") == "text" and isinstance(_field(block, "text"), str):
                piece = ("\n" if text_parts else "") + _field(block, "text")
                remaining = config.max_output_chars - text_length
                if len(piece) > remaining:
                    text_parts.append(piece[:max(0, remaining)])
                    text_length = config.max_output_chars
                    truncated = True
                elif remaining > 0:
                    text_parts.append(piece)
                    text_length += len(piece)
                else:
                    truncated = True
            else:
                ignored += 1
        ignored += max(0, len(content) - 100)
        combined = "".join(text_parts)
        payload: dict[str, Any] = {
            "status": "success", "server_name": config.server_name,
            "tool_name": public_name, "text": combined, "truncated": truncated,
        }
        if ignored:
            payload["ignored_content_blocks"] = ignored
        structured = _field(result, "structuredContent", "structured_content")
        if structured is not None:
            try:
                plain = _plain_json(structured)
                remaining = max(0, config.max_output_chars - len(combined))
                if _json_size(plain) <= remaining:
                    payload["structured_content"] = plain
                else:
                    payload["structured_content_omitted"] = "output_too_large"
            except (ValueError, TypeError):
                payload["structured_content_omitted"] = "invalid_json"
        return payload

    @staticmethod
    def _finish(future: asyncio.Future[dict[str, Any]], value: dict[str, Any]) -> None:
        if not future.done():
            future.set_result(value)

    def _remove_server_tools(self, server_name: str) -> None:
        with self._lock:
            for name in [name for name, spec in self._tools.items()
                         if spec.server_name == server_name]:
                self._tools.pop(name, None)

    def _set_status(self, server_name: str, state: McpServerState,
                    error_code: str | None = None) -> None:
        with self._lock:
            self._statuses[server_name] = McpServerStatus(server_name, state, error_code)


__all__ = [
    "McpConfigurationError", "McpRuntime", "McpServerConfig", "McpServerState",
    "McpServerStatus", "McpToolSpec", "load_mcp_config",
]
