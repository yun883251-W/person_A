"""B 站搜索的固定契约，供宿主和 MCP 服务共享；仅依赖标准库。

参数只能描述搜索词，不能提供 URL、浏览器命令、程序路径或执行脚本。
成功结果仅证明本机浏览器启动接口接受了请求，不证明网页加载或搜索结果可用。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlencode

REMOTE_SEARCH_TOOL = "search_bilibili"
SEARCH_BILIBILI_TOOL = "mcp__browser__search_bilibili"
BILIBILI_SEARCH_ENDPOINT = "https://search.bilibili.com/all"
MAX_QUERY_LENGTH = 200
_INVALID_QUERY_CHARS = r"[\u0000-\u001f\u007f-\u009f\ud800-\udfff]"
_INVALID_QUERY_RE = re.compile(_INVALID_QUERY_CHARS)


class InvalidSearchArguments(ValueError):
    """固定的公开参数错误；不在异常消息中复制原始用户输入。"""

    code = "invalid_search_arguments"

    def __init__(self) -> None:
        super().__init__("只接受 query：1 至 200 个字符的非空搜索词，不能包含控制字符。")


def validate_search_arguments(arguments: Mapping[str, Any]) -> dict[str, str]:
    """严格检查参数对象，再去除搜索词两端空白；不会执行浏览器动作。"""
    if not isinstance(arguments, Mapping) or set(arguments) != {"query"}:
        raise InvalidSearchArguments()
    query = arguments["query"]
    if (
        type(query) is not str
        or not 1 <= len(query) <= MAX_QUERY_LENGTH
        or _INVALID_QUERY_RE.search(query) is not None
        or not query.strip()
    ):
        raise InvalidSearchArguments()
    return {"query": query.strip()}


def build_search_url(query: str) -> str:
    """将已校验的搜索词编码为固定站点的 keyword 参数，不拼接原始 URL。"""
    normalized = validate_search_arguments({"query": query})["query"]
    return BILIBILI_SEARCH_ENDPOINT + "?" + urlencode({"keyword": normalized})


def search_input_schema() -> dict[str, Any]:
    """返回新对象，避免调用方修改共享 Schema；约束与 Python 校验保持一致。"""
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_QUERY_LENGTH,
                "allOf": [
                    {"pattern": r"\S"},
                    {"not": {"pattern": _INVALID_QUERY_CHARS}},
                ],
                "description": "用户明确要求在 B 站搜索的关键词，不包含控制字符。",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
