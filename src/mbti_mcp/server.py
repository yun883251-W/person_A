"""MindCare MBTI 的 stdio MCP 入口；stdout 仅承载协议消息。"""

from __future__ import annotations

import argparse
import sys
from typing import Annotated, Any

from pydantic import Field, StrictStr

from .service import MbtiService, default_mindcare_root

QuestionOffset = Annotated[int, Field(strict=True, ge=0, le=47)]
PageLimit = Annotated[int, Field(strict=True, ge=1, le=12)]
AnswerScore = Annotated[int, Field(strict=True, ge=1, le=5)]
QuestionId = Annotated[str, Field(strict=True, pattern=r"^(?:[0-9]|[1-3][0-9]|4[0-7])$")]


def create_server(service: MbtiService):
    """工厂不启动服务；MCP 依赖仅在显式创建服务时导入。"""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    server = FastMCP(
        "MindCare MBTI",
        instructions=(
            "提供 MindCare 的 48 题自评问卷、确定性评分和类型描述。"
            "只使用用户明确提交的答案，不补答、不推测；"
            "百分比为得分占比，不是置信度。答题进度由宿主保存。"
        ),
    )
    annotations = ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False,
    )

    @server.tool(annotations=annotations)
    def get_mbti_questionnaire(offset: QuestionOffset = 0, limit: PageLimit = 12) -> dict[str, Any]:
        """分页获取 MBTI 自评问卷，最多 12 题。保留版本和题目 ID；按 options 的整数值收集用户答案。"""
        return service.get_mbti_questionnaire(offset, limit)

    @server.tool(annotations=annotations)
    def score_mbti_answers(answers: dict[QuestionId, AnswerScore], questionnaire_version: StrictStr) -> dict[str, Any]:
        """对已答完的 48 题评分。键必须为字符串 0 至 47，值为整数 1 至 5，版本必须与问卷一致。不得代用户补答或猜测答案。"""
        return service.score_mbti_answers(answers, questionnaire_version)

    @server.tool(annotations=annotations)
    def get_mbti_description(type_code: StrictStr) -> dict[str, Any]:
        """获取 INTJ 等四字母类型的一般性描述。用于解释类型；不能据此认定用户人格或进行诊断。"""
        return service.get_mbti_description(type_code)

    # 工具校验/业务异常交由 SDK 转为 isError 工具结果，绝不伪造成功数据。
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="通过 stdio 暴露 MindCare MBTI 纯 Python 模块")
    parser.add_argument("--mindcare-root", default=str(default_mindcare_root()), help="MindCare 项目根目录，包含 backend/mbti")
    args = parser.parse_args(argv)
    try:
        service = MbtiService(args.mindcare_root)
        server = create_server(service)
    except Exception:
        # 不向 stdout 写日志，也不在启动错误中泄露配置路径或环境内容。
        print("mbti_mcp_startup_failed: 请确认 MindCare 路径、源模块和 Person_A 的 mcp 可选依赖。", file=sys.stderr)
        return 1
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
