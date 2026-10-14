"""会话创建、查询、历史读取、清空与删除路由。"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response, status

from ..runtime import ServiceRuntime
from ..schemas import (
    CreateSessionRequest,
    ErrorResponse,
    PageLimit,
    PageOffset,
    SessionId,
    SessionListResponse,
    SessionResponse,
    SessionSummaryResponse,
    TurnListResponse,
)
from ..serialization import _session_response, _turn_response


def register_routes(application: FastAPI) -> None:
    """Register session and persisted-history endpoints."""
    @application.post(
        "/sessions",
        response_model=SessionResponse,
        status_code=status.HTTP_201_CREATED,
        summary="创建会话",
        description="创建一个可持久化问答历史的新会话，并返回服务生成的会话 ID。",
        responses={
            422: {"model": ErrorResponse, "description": "请求体不合法"},
            500: {"model": ErrorResponse, "description": "会话存储失败"},
            503: {"model": ErrorResponse, "description": "服务忙或存储暂时不可用"},
        },
    )
    async def create_session(request: CreateSessionRequest) -> SessionResponse:
        runtime: ServiceRuntime = application.state.runtime
        record = await runtime.run_store(
            lambda store: store.create_session(title=request.title)
        )
        return _session_response(record)

    @application.get(
        "/sessions",
        response_model=SessionListResponse,
        summary="分页列出会话",
        description=(
            "按存储层的最近更新时间顺序列出会话。limit 为本页上限（1—100），"
            "offset 为从结果开头跳过的会话数；响应回显两者。"
        ),
        responses={
            422: {"model": ErrorResponse, "description": "分页参数不合法"},
            500: {"model": ErrorResponse, "description": "会话存储失败"},
            503: {"model": ErrorResponse, "description": "服务忙或存储暂时不可用"},
        },
    )
    async def list_sessions(
        limit: PageLimit = 50,
        offset: PageOffset = 0,
    ) -> SessionListResponse:
        runtime: ServiceRuntime = application.state.runtime
        records = await runtime.run_store(
            lambda store: store.list_sessions(limit=limit, offset=offset)
        )
        return SessionListResponse(
            items=[_session_response(record) for record in records],
            limit=limit,
            offset=offset,
        )

    @application.get(
        "/sessions/{session_id}",
        response_model=SessionResponse,
        summary="查询会话信息",
        description="读取一个会话的标题和创建、更新时间，不返回问答历史。",
        responses={
            404: {"model": ErrorResponse, "description": "会话不存在"},
            422: {"model": ErrorResponse, "description": "会话 ID 不合法"},
            500: {"model": ErrorResponse, "description": "会话存储失败"},
            503: {"model": ErrorResponse, "description": "服务忙或存储暂时不可用"},
        },
    )
    async def get_session(session_id: SessionId) -> SessionResponse:
        runtime: ServiceRuntime = application.state.runtime
        record = await runtime.run_store(
            lambda store: store.get_session(session_id)
        )
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
        return _session_response(record)

    @application.get(
        "/sessions/{session_id}/summary",
        response_model=SessionSummaryResponse,
        summary="读取早期知识对话摘要",
        description="只读取已保存的摘要与处理进度，不生成摘要、不调用模型；完整问答历史仍可单独读取。",
        responses={404: {"model": ErrorResponse, "description": "会话不存在"}},
    )
    async def get_summary(session_id: SessionId) -> SessionSummaryResponse:
        runtime: ServiceRuntime = application.state.runtime

        def read_summary(store):
            if store.get_session(session_id) is None:
                return None
            record = store.load_session_summary(session_id)
            return SessionSummaryResponse(
                session_id=session_id,
                summary=record.summary if record is not None else "",
                covered_turn_no=record.covered_turn_no if record is not None else 0,
                updated_at=record.updated_at if record is not None else None,
            )

        result = await runtime.run_store(read_summary)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        return result

    @application.get(
        "/sessions/{session_id}/messages",
        response_model=TurnListResponse,
        summary="分页读取问答历史",
        description=(
            "按时间正序返回已完成的完整问答轮，每个 item 同时包含问题、完整回答及引用，"
            "而不是把单条消息拆开分页。limit 为本页上限（1—100），offset 为跳过的"
            "问答轮数；响应回显两者。trace 返回已保存的执行摘要，旧记录可能为 null。"
        ),
        responses={
            404: {"model": ErrorResponse, "description": "会话不存在"},
            422: {"model": ErrorResponse, "description": "会话 ID 或分页参数不合法"},
            500: {"model": ErrorResponse, "description": "会话存储失败"},
            503: {"model": ErrorResponse, "description": "服务忙或存储暂时不可用"},
        },
    )
    async def list_messages(
        session_id: SessionId,
        limit: PageLimit = 50,
        offset: PageOffset = 0,
    ) -> TurnListResponse:
        runtime: ServiceRuntime = application.state.runtime
        turns = await runtime.load_turns(
            session_id,
            limit=limit,
            offset=offset,
        )
        if turns is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
        return TurnListResponse(
            items=[_turn_response(turn) for turn in turns],
            limit=limit,
            offset=offset,
        )

    @application.delete(
        "/sessions/{session_id}/messages",
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
        summary="清空会话历史",
        description="永久删除该会话的全部已完成问答轮，但保留会话本身。成功时无响应正文。",
        responses={
            404: {"model": ErrorResponse, "description": "会话不存在"},
            409: {"model": ErrorResponse, "description": "会话正在处理问答，无法清空"},
            422: {"model": ErrorResponse, "description": "会话 ID 不合法"},
            500: {"model": ErrorResponse, "description": "会话存储失败"},
            503: {"model": ErrorResponse, "description": "服务忙或存储暂时不可用"},
        },
    )
    async def clear_messages(session_id: SessionId) -> Response:
        runtime: ServiceRuntime = application.state.runtime
        if not await runtime.clear_session(session_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.delete(
        "/sessions/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
        summary="删除会话",
        description="永久删除会话及其全部问答历史。成功时无响应正文。",
        responses={
            404: {"model": ErrorResponse, "description": "会话不存在"},
            409: {"model": ErrorResponse, "description": "会话正在处理问答，无法删除"},
            422: {"model": ErrorResponse, "description": "会话 ID 不合法"},
            500: {"model": ErrorResponse, "description": "会话存储失败"},
            503: {"model": ErrorResponse, "description": "服务忙或存储暂时不可用"},
        },
    )
    async def delete_session(session_id: SessionId) -> Response:
        runtime: ServiceRuntime = application.state.runtime
        if not await runtime.delete_session(session_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
