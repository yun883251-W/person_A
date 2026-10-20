"""普通 JSON 问答与实时阶段进度订阅路由。"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from ..errors import _answer_service_error
from ..runtime import ServiceRuntime
from ..schemas import AnswerResponse, AskRequest, ErrorResponse, SessionId
from ..serialization import _answer_response
from ..streaming import create_progress_response


def register_routes(application: FastAPI) -> None:
    """注册问答路由，同时保持单工作线程的忙碌策略。"""
    @application.post(
        "/sessions/{session_id}/messages",
        response_model=AnswerResponse,
        summary="在会话中提问",
        description=(
            "在已有会话中执行一轮问答，完成后一次性返回完整结果（非流式）。status 是"
            "业务状态：success 表示已回答，其他状态可能表示资料不足、需要澄清、超出"
            "范围或达到调用限制，应与 HTTP 状态分别判断。成功完成的问答轮会写入历史。"
            "trace 为可选的执行摘要，记录实际阶段与耗时，不包含内部消息或推理内容。"
        ),
        responses={
            404: {"model": ErrorResponse, "description": "会话不存在"},
            409: {"model": ErrorResponse, "description": "会话正在处理另一轮问答"},
            422: {"model": ErrorResponse, "description": "会话 ID 或问题不合法"},
            500: {
                "model": ErrorResponse,
                "description": "内部错误，或答案已生成但未能持久化",
            },
            503: {
                "model": ErrorResponse,
                "description": "服务忙、索引未构建、存储或模型服务不可用",
            },
            504: {"model": ErrorResponse, "description": "模型响应超时"},
        },
    )
    async def ask_session(
        session_id: SessionId,
        request: AskRequest,
    ) -> AnswerResponse:
        runtime: ServiceRuntime = application.state.runtime
        try:
            skill_options: dict[str, Any] = {"response_mode": request.response_mode}
            if request.skill_id is not None:
                skill_options["skill_id"] = request.skill_id
            if request.skill_mode is not None:
                skill_options["skill_mode"] = request.skill_mode
            result = await runtime.ask_session(
                session_id, request.question, **skill_options,
            )
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown skill: {request.skill_id}",
            ) from None
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
        response = _answer_response(session_id, result)
        if result.status != "service_error":
            return response
        return _answer_service_error(response)

    @application.post(
        "/sessions/{session_id}/messages/stream",
        response_class=StreamingResponse,
        summary="提问并订阅实时阶段进度",
        description=(
            "使用 POST fetch 读取 text/event-stream。progress 为公开阶段状态；result 为"
            "完整 AnswerResponse；error 使用既有 ErrorResponse，可携带未保存的 result。"
            "流开始后的业务错误通过 error 事件返回；流开始前的参数和忙碌错误仍为 HTTP JSON。"
            "不会发送模型推理、Token 或检索原文，也不提供自动重试或请求级断线恢复。"
        ),
        responses={
            422: {"model": ErrorResponse, "description": "请求参数不合法"},
            503: {"model": ErrorResponse, "description": "服务忙或不可用"},
        },
    )
    async def ask_session_stream(
        session_id: SessionId,
        request: AskRequest,
        http_request: Request,
    ) -> StreamingResponse:
        runtime: ServiceRuntime = application.state.runtime
        return await create_progress_response(runtime, session_id, request, http_request)
