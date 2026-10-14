"""发布给模型的工具 Schema 必须留在已确认的保守子集内。

背景：`mcp__browser__search_bilibili` 曾把 `[\\ud800-\\udfff]`（UTF-16 代理区
范围）写进 `pattern`。本地一切都通过 —— `jsonschema` 接受、运行时发现工具、
22 个用例全绿 —— 但 DeepSeek 的请求校验拒绝整个请求，功能 100% 不可用。

本文件是那一层的离线守卫，不需要网络和凭据。它检查的是**真正发给提供方的载荷**
（`convert_to_openai_tool` 的输出），而不是项目内部的 Schema 副本。

在线确认真实提供方是否接受，由同一目录的 test_live_provider_schema.py 负责；
本文件的断言全部是提供方无关的硬约束，可以在 CI 中无条件运行。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

# 已确认被真实提供方（DeepSeek）接受的关键字。这是经验清单，不是规范子集：
# 加入新关键字前，必须先用在线探测确认真实提供方接受，再在此登记。
_KNOWN_GOOD_KEYWORDS = frozenset(
    {
        # 结构与标注
        "$schema", "type", "title", "description", "default",
        # 对象
        "properties", "required", "additionalProperties", "patternProperties",
        # 数组
        "items", "prefixItems", "minItems", "maxItems",
        # 字符串
        "pattern", "format", "minLength", "maxLength",
        # 数值
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "multipleOf",
        # 取值
        "enum", "const",
        # 组合
        "allOf", "anyOf", "oneOf", "not",
    }
)

# 值为「子 Schema 映射」的关键字：其键是名称，不是关键字。
_SUBSCHEMA_MAP_KEYWORDS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)
# 值为「单个子 Schema」的关键字。
_SUBSCHEMA_SINGLE_KEYWORDS = frozenset(
    {
        "items", "not", "if", "then", "else", "contains",
        "propertyNames", "additionalProperties",
    }
)
# 值为「子 Schema 数组」的关键字。
_SUBSCHEMA_LIST_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})

# 正则源码里的 UTF-16 代理区转义，例如 `\ud800`。这正是当时被拒的写法。
_SURROGATE_ESCAPE_RE = re.compile(r"\\u[dD][89abABcCdDeEfF][0-9a-fA-F]{2}")


def iter_schema_keywords(schema: Any, path: str = "<schema>"):
    """Yield (path, keyword) for every JSON Schema keyword in the payload."""
    if not isinstance(schema, Mapping):
        return
    for keyword, value in schema.items():
        if not isinstance(keyword, str):
            continue
        here = f"{path}.{keyword}"
        yield here, keyword
        if keyword in _SUBSCHEMA_MAP_KEYWORDS and isinstance(value, Mapping):
            for name, child in value.items():
                yield from iter_schema_keywords(child, f"{here}[{name!r}]")
        elif keyword in _SUBSCHEMA_SINGLE_KEYWORDS and isinstance(value, Mapping):
            yield from iter_schema_keywords(value, here)
        elif keyword in _SUBSCHEMA_LIST_KEYWORDS and isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from iter_schema_keywords(child, f"{here}[{index}]")


def iter_string_values(value: Any, path: str = "<schema>"):
    """Yield (path, text) for every string in the payload, including object keys."""
    if isinstance(value, str):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str):
                yield f"{path}.{key}", key
            yield from iter_string_values(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from iter_string_values(child, f"{path}[{index}]")


def iter_regex_sources(value: Any, path: str = "<schema>"):
    """Yield (path, regex_source) for `pattern` and `patternProperties` keys."""
    if isinstance(value, Mapping):
        for keyword, child in value.items():
            if keyword == "pattern" and isinstance(child, str):
                yield f"{path}.pattern", child
            elif keyword == "patternProperties" and isinstance(child, Mapping):
                for name in child:
                    if isinstance(name, str):
                        yield f"{path}.patternProperties[{name!r}]", name
            yield from iter_regex_sources(child, f"{path}.{keyword}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from iter_regex_sources(child, f"{path}[{index}]")


def find_schema_problems(schema: Any) -> list[str]:
    """Return every reason this schema must not be published to a model."""
    problems: list[str] = []

    for path, text in iter_string_values(schema):
        for character in text:
            if 0xD800 <= ord(character) <= 0xDFFF:
                problems.append(
                    f"{path}: 含孤立代理字符 U+{ord(character):04X}，无法编码为 UTF-8"
                )
                break
        match = _SURROGATE_ESCAPE_RE.search(text)
        if match is not None:
            problems.append(
                f"{path}: 含代理区转义 {match.group(0)}，真实提供方会拒绝整个请求"
            )

    for path, source in iter_regex_sources(schema):
        try:
            re.compile(source)
        except re.error as exc:
            problems.append(f"{path}: 正则无法编译 ({exc})")

    for path, keyword in iter_schema_keywords(schema):
        if keyword not in _KNOWN_GOOD_KEYWORDS:
            problems.append(
                f"{path}: 关键字 {keyword!r} 未登记。"
                "请先用在线探测确认真实提供方接受，再加入 _KNOWN_GOOD_KEYWORDS。"
            )

    return problems


def provider_parameters(definition: Any) -> dict[str, Any]:
    """Return the exact parameter object the provider receives for one tool."""
    converted = convert_to_openai_tool(definition)
    return converted["function"]["parameters"]


def test_surrogate_escape_rule_catches_the_historical_regression() -> None:
    """The lint is only useful if it flags the exact schema that caused the outage."""
    regression = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "allOf": [
                    {"pattern": r"\S"},
                    {"not": {"pattern": r"[\u0000-\u001f\ud800-\udfff]"}},
                ],
            },
        },
        "required": ["query"],
    }
    problems = find_schema_problems(regression)
    assert any("代理区转义" in problem for problem in problems), problems

    # 同一条规则不能误伤已经对照过真实提供方的当前写法。
    fixed = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "allOf": [
                    {"pattern": r"\S"},
                    {"not": {"pattern": r"[\u0000-\u001f\u007f-\u009f]"}},
                ],
            },
        },
        "required": ["query"],
    }
    assert find_schema_problems(fixed) == []


def test_lint_ignores_property_names_that_look_like_keywords() -> None:
    """`properties` 的键是参数名，不应被当成 Schema 关键字登记检查。"""
    schema = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}, "allOf": {"type": "integer"}},
        "required": ["pattern"],
    }
    assert find_schema_problems(schema) == []


def test_lint_flags_unregistered_keywords() -> None:
    """新增关键字必须显式登记，否则下一次「只有真实提供方能发现」的问题会重演。"""
    schema = {"type": "object", "unevaluatedProperties": False}
    problems = find_schema_problems(schema)
    assert any("unevaluatedProperties" in problem for problem in problems), problems


def test_published_tool_payload_is_provider_compatible(published_mcp_tools) -> None:
    """每个已发布工具的提供方载荷都必须通过保守子集检查。"""
    failures: list[str] = []
    for registration in published_mcp_tools:
        parameters = provider_parameters(registration.definition)
        for problem in find_schema_problems(parameters):
            failures.append(f"{registration.name}: {problem}")
    assert not failures, "发布的工具 Schema 不满足保守子集:\n" + "\n".join(failures)


def test_published_tool_payload_is_utf8_serializable(published_mcp_tools) -> None:
    """整个工具载荷（含描述）必须能原样编码为 UTF-8 并发往提供方。"""
    for registration in published_mcp_tools:
        payload = convert_to_openai_tool(registration.definition)
        text = json.dumps(payload, ensure_ascii=False)
        assert text.encode("utf-8").decode("utf-8") == text, registration.name


def test_every_configured_strategy_reaches_the_model(published_mcp_tools) -> None:
    """配置里 enabled 的服务器都必须真正发布出工具。

    覆盖另一种静默失败：服务器启动超时或发现阶段出错时，配置看起来是启用状态，
    但模型侧一个工具都看不到，用户只会得到「不知道能做这个」的回答。
    """
    from knowledge_agent.config import get_mcp_settings
    from knowledge_agent.mcp.client import load_mcp_config

    settings = get_mcp_settings()
    assert settings.config_path is not None
    expected = {
        f"mcp__{config.server_name}__{tool}"
        for config in load_mcp_config(settings.config_path)
        if config.enabled
        for tool in config.allowed_tools
    }
    published = {registration.name for registration in published_mcp_tools}
    assert published == expected


def test_browser_query_contract_still_rejects_surrogates() -> None:
    """发布面收窄后，真正执行动作的 Python 校验仍必须拒绝代理字符。

    发布给模型的正则已不再声明代理区（那是被拒的写法），但闸门在 Python 侧；
    这条断言防止有人「顺手」把服务端校验也一起放宽，从而把不安全输入放进 URL。
    """
    from browser_mcp.contracts import InvalidSearchArguments, validate_search_arguments

    with pytest.raises(InvalidSearchArguments):
        validate_search_arguments({"query": "a\ud800b"})
    assert validate_search_arguments({"query": "  B 站  "}) == {"query": "B 站"}
