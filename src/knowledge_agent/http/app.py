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
        version="0.1.0",
        description=(
            "本机文档知识库问答服务。问答接口在处理完成后一次性返回完整结果，"
            "同时提供 SSE 阶段进度。会话和已完成问答持久化在 SQLite 中；停止服务不会"
            "删除已提交记录。若同一工作线程正处理其他业务，请求可能返回 503；模型"
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


# 验证项：以下 TODO 尚未统一执行。

# TODO 17：集中验证 HTTP 参数、状态码与响应格式。
# 使用 FastAPI TestClient 和假运行时，覆盖健康检查、会话 CRUD、分页、空问题及非法参数。
# 验证 201/200/204/404/409/422/500/503/504 与约定相符，204 无响应正文。
# 检查业务状态与 HTTP 状态分别表达，资料不足或调用上限不能被显示成答案质量通过。
# 检查公开结果无内部工具消息、密钥和原始异常，/docs 和健康检查不调用真实模型。

# TODO 18：集中验证线程归属、会话隔离和生命周期。
# 用临时 SQLite 数据库、假模型及记录线程 ID 的运行时，核对连接创建/操作/关闭在同一线程。
# 交替对两个会话提问，验证历史互不混用；验证问答后活动会话释放且数据持久保留。
# 模拟问答进行时的第二次问答、清空和删除，确认忙碌返回且数据库不被并发改写。
# 模拟初始化失败、模型异常、保存失败与服务关闭，检查资源释放及旧记录保留。

# TODO 19：集中验证取消、超时和失联后的行为。
# 用可控的阻塞假任务模拟 HTTP 等待取消，确认后台任务不会被误认为已经停止。
# 检查 busy 状态直到实际任务结束才释放，任务自行完成 finally，会话不会遗留为活动状态。
# 验证保存后响应丢失的场景，客户端可查到历史，不发生服务端自动重试导致的重复问答。
# 关闭时等待在途任务并在工作线程清理，模型长请求不会阻塞 /health 所在事件循环。

# TODO 20：完成真实 API 与网页联调。
# 先确认已发布索引及模型配置可用，显式启动本机服务，通过 /docs 完成一次新建、提问和查历史。
# 重启服务后使用同一会话继续追问，核对 SQLite 历史、回答引用和知识库版本。
# 使用临时会话验证清空与删除，记录各接口实际结果；不拿用户重要聊天记录做破坏性验证。
# 检查 CLI 与 API 对同一数据库的基本使用约定，数据库写锁等待失败能够被正确提示。
# 全部开发和验证完成前不标记接口可用；效果评估仍可使用 evaluation 模块后续单独补齐。
