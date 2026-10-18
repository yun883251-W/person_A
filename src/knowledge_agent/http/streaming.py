"""有界 SSE 进度通道与响应生成；客户端离开不会取消后台业务任务。"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from threading import Lock
from typing import Any, Literal

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from .errors import _answer_service_error, internal_error_handler
from .runtime import ServiceRuntime
from .schemas import AskRequest
from .serialization import _answer_response

class _ProgressChannel:
    """单条已连接 HTTP 流的有界工作线程→事件循环通道。

    上限同时约束已调度回调和排队载荷。最终结果经由工作线程 Future 传递，
    因此慢客户端不会因为进度队列已满而丢失回答。断开连接不会取消业务操作。
    """

    MAX_EVENTS = 256

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._lock = Lock()
        self._attached = True
        self._sent = 0
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.MAX_EVENTS)

    def publish(self, payload: dict[str, Any]) -> None:
        with self._lock:
            if not self._attached or self._sent >= self.MAX_EVENTS:
                return
            self._sent += 1
            # 载荷只来自 trace 的公开字段序列化器，
            # 绝不来自模型消息、工具参数或检索原文。
            try:
                self._loop.call_soon_threadsafe(self._enqueue, dict(payload))
            except RuntimeError:
                self._attached = False

    def _enqueue(self, payload: dict[str, Any]) -> None:
        with self._lock:
            attached = self._attached
        if attached and not self.queue.full():
            self.queue.put_nowait(payload)

    def detach(self) -> None:
        with self._lock:
            self._attached = False


def _sse_event(event: Literal["progress", "result", "error"], value: dict[str, Any]) -> str:
    # JSON 会转义内嵌换行，因此文本无法注入额外的 SSE 帧。
    return f"event: {event}\ndata: {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}\n\n"


async def create_progress_response(
    runtime: ServiceRuntime,
    session_id: str,
    request: AskRequest,
    http_request: Request,
) -> StreamingResponse:
    """只提交一次，随后发送公开进度与已校验的最终响应。"""
    channel = _ProgressChannel(asyncio.get_running_loop())
    try:
        # 在发送响应头之前先占用同一个工作线程，
        # 使并发提问与修改操作继续遵守既有的忙碌策略。
        worker_future = runtime.submit_ask_session(
            session_id, request.question,
            skill_id=request.skill_id, skill_mode=request.skill_mode,
            progress_observer=channel.publish,
        )
    except BaseException:
        channel.detach()
        raise
    wrapped = asyncio.wrap_future(worker_future)
    # 下面的 asyncio.wait 不会取消该 Future。即使浏览器离开，
    # 完成时仍会取回异常，且持久化由工作线程负责。
    wrapped.add_done_callback(ServiceRuntime._consume_future_exception)

    async def stream_events():
        pending_progress: asyncio.Task[dict[str, Any]] | None = None
        try:
            while True:
                while not channel.queue.empty():
                    yield _sse_event("progress", channel.queue.get_nowait())
                if wrapped.done():
                    break
                pending_progress = asyncio.create_task(channel.queue.get())
                completed, _ = await asyncio.wait(
                    (wrapped, pending_progress), timeout=15,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if pending_progress in completed:
                    yield _sse_event("progress", pending_progress.result())
                else:
                    pending_progress.cancel()
                    with suppress(asyncio.CancelledError):
                        await pending_progress
                pending_progress = None
                if not completed:
                    yield ": keep-alive\n\n"

            result = wrapped.result()
            if result is None:
                raise HTTPException(status_code=404, detail="Session not found")
            response = _answer_response(session_id, result)
            if result.status == "service_error":
                error_response = _answer_service_error(response)
                yield _sse_event("error", {
                    **json.loads(error_response.body), "http_status": error_response.status_code,
                })
            else:
                yield _sse_event("result", response.model_dump(mode="json"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if isinstance(exc, KeyError):
                exc = HTTPException(
                    status_code=422, detail=f"Unknown skill: {request.skill_id}",
                )
            handler = next(
                (
                    http_request.app.exception_handlers[exception_type]
                    for exception_type in type(exc).__mro__
                    if exception_type in http_request.app.exception_handlers
                ),
                internal_error_handler,
            )
            error_response = await handler(http_request, exc)
            yield _sse_event("error", {
                **json.loads(error_response.body), "http_status": error_response.status_code,
            })
        finally:
            channel.detach()
            if pending_progress is not None and not pending_progress.done():
                pending_progress.cancel()
                with suppress(asyncio.CancelledError):
                    await pending_progress

    return StreamingResponse(
        stream_events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
