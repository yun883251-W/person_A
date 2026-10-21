"""FastAPI 应用组装：生命周期、跨域配置和路由注册。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import PROJECT_ROOT
from .errors import register_error_handlers
from .routes import chat, knowledge, sessions, system, profile
from .runtime import RuntimeFactory, ServiceRuntime

LOCAL_API_HOST = "127.0.0.1"
LOCAL_API_PORT = 8000
LOCAL_API_WORKERS = 1

def _validated_cors_origins(origins: tuple[str, ...]) -> tuple[str, ...]:
    """返回去重后的显式浏览器来源，供本机网页客户端使用。"""
    normalized: list[str] = []
    for raw_origin in origins:
        origin = raw_origin.strip().rstrip("/")
        parsed = urlsplit(origin)
        if (
            not origin
            or origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "cors_origins must contain explicit HTTP(S) origins, for example "
                "http://127.0.0.1:5173"
            )
        if origin not in normalized:
            normalized.append(origin)
    return tuple(normalized)


def create_app(
    runtime_factory: RuntimeFactory | None = None,
    *,
    cors_origins: tuple[str, ...] = (),
) -> FastAPI:
    """提供本机网页与 API 服务；额外的浏览器来源需显式开启。"""

    allowed_origins = _validated_cors_origins(cors_origins)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        factory = application.state.runtime_factory or ServiceRuntime
        runtime = factory()
        application.state.runtime = runtime
        try:
            runtime.initialize()
            yield
        finally:
            try:
                runtime.close()
            finally:
                application.state.runtime = None

    application = FastAPI(
        title="Person A Knowledge Agent API",
        version="1.0.0",
        description=(
            "本机文档知识库问答服务。问答接口在处理完成后一次性返回完整结果，"
            "同时提供 SSE 阶段进度。会话和已完成问答持久化在 SQLite 中；停止服务不会"
            "删除已提交记录。请求进入有界并发队列，队列满或等待超时返回 503；模型"
            "超时返回 504。客户端断开只会停止等待，不保证取消已经提交到后台的任务，"
            "稍后可通过历史接口确认结果是否已保存。"
        ),
        lifespan=lifespan,
    )
    if allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type"],
        )

    application.state.runtime_factory = runtime_factory
    application.state.runtime = None
    register_error_handlers(application)
    system.register_routes(application, web_directory=PROJECT_ROOT / "web")
    knowledge.register_routes(application)
    sessions.register_routes(application)
    chat.register_routes(application)
    profile.register_routes(application)
    return application
