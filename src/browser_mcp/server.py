"""本机 B 站搜索的 stdio MCP 入口；stdout 专用于协议，诊断写 stderr。

使用官方 MCP 1.x 的 lowlevel Server 显式声明根参数对象，禁止额外参数。
SDK 依赖仅在显式创建/启动服务时加载，不在包导入时打开浏览器或启动进程。
该工具会请求打开浏览器标签页，因此不标注为只读或幂等工具。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .contracts import REMOTE_SEARCH_TOOL, InvalidSearchArguments, search_input_schema, validate_search_arguments
from .service import BrowserLaunchError, BrowserService


def create_server(service: BrowserService | None = None) -> Any:
    """创建工具注册表而不运行服务；业务执行器可替换为受控测试替身。"""
    from mcp.server.lowlevel import Server
    from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

    executor = service if service is not None else BrowserService()
    server = Server(
        "Person A Local Browser",
        version="0.1.0",
        instructions=(
            "仅在用户明确要求时调用 search_bilibili，在本机默认浏览器请求打开 B 站搜索页。"
            "成功仅表示浏览器启动接口接受请求；page_verified 始终为 false。"
            "不会读取搜索结果、获取视频内容、登录或下载。一次真实尝试之后不得自动重试。"
        ),
    )

    @server.list_tools()
    async def list_tools() -> list[Any]:
        return [
            Tool(
                name=REMOTE_SEARCH_TOOL,
                description=(
                    "在运行本 MCP 的电脑上请求默认浏览器打开 B 站搜索页。"
                    "只接收 query 搜索词，用于用户明确提出的 B 站搜索操作。"
                    "不会读取搜索结果或视频；返回 browser_launch_requested=true 仅表示"
                    "启动接口接受请求，page_verified=false 表示没有确认页面加载。"
                ),
                inputSchema=search_input_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            ),
        ]

    def error_result(code: str, message: str) -> Any:
        payload = {
            "status": "error",
            "site": "bilibili",
            "page_verified": False,
            "error": {"code": code, "message": message},
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
            structuredContent=payload,
            isError=True,
        )

    # 发布的 Schema 供宿主严格校验；服务独立使用同一 Python 契约再次检查。
    # 不使用 SDK 的默认校验错误文本，因为其错误可能包含完整原始参数。
    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
        if name != REMOTE_SEARCH_TOOL:
            return error_result("unknown_tool", "请求的工具不存在。")
        try:
            validated = validate_search_arguments(arguments)
            # 默认浏览器调用可能阻塞；只提交一次，不阻塞 stdio 协议循环。
            # 若等待方取消，已经提交的本机动作仍可能完成，宿主不能自动重发。
            result = await asyncio.to_thread(executor.search_bilibili, validated["query"])
        except InvalidSearchArguments as exc:
            return error_result(exc.code, str(exc))
        except BrowserLaunchError as exc:
            return error_result(exc.code, str(exc))
        except Exception:
            return error_result("browser_tool_failed", "浏览器工具执行失败；未自动重试，也未验证页面。")
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
            structuredContent=result,
            isError=False,
        )

    return server


async def _serve_stdio(server: Any) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="通过 stdio 暴露本机 B 站浏览器搜索工具")
    parser.parse_args(argv)
    try:
        server = create_server()
        asyncio.run(_serve_stdio(server))
    except KeyboardInterrupt:
        return 0
    except Exception:
        print(
            "browser_mcp_failed: 请确认已安装 Person_A 的 mcp 可选依赖，且本机执行环境可用。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
