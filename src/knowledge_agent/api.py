"""FastAPI HTTP 服务模块。

模块功能：接收网页或其他客户端的 HTTP 请求，校验输入，调用现有 Agent
和会话存储接口，将回答、引用、会话信息与错误转换为约定的 JSON 响应。
检索、模型调用、来源校验和 SQLite 持久化继续复用已有模块。

第一版范围：本机 127.0.0.1、单服务进程、同步完整回答、持久化会话。
使用贯穿服务生命周期的固定工作线程串行执行 Agent 和数据库操作。
HTTP 接口异步等待任务，SQLite 连接在同一个工作线程创建、使用和关闭。

网页入口与接口：
GET    /                                本地知识库问答页面
GET    /assets/{path}                   页面静态资源
GET    /health                          服务存活检查
POST   /sessions                        创建会话
GET    /sessions                        分页列出会话
GET    /sessions/{session_id}            查询会话信息
POST   /sessions/{session_id}/messages   在已有会话中提问
GET    /sessions/{session_id}/messages    分页读取已完成问答轮
DELETE /sessions/{session_id}/messages   清空历史，保留会话
DELETE /sessions/{session_id}            删除会话及其历史

开发参考：
https://fastapi.tiangolo.com/advanced/events/
https://fastapi.tiangolo.com/async/
https://docs.python.org/3.10/library/concurrent.futures.html

安装 API 依赖：
python -m pip install -e ".[api]"

启动本机单进程服务：
python -m uvicorn knowledge_agent.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Lock, get_ident
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from .agent.runner import ApplicationBusyError, TurnPersistenceError
from .config import PROJECT_ROOT
from .models import (
    ChatModelAuthenticationError,
    ChatModelInvocationError,
    ChatModelTimeoutError,
)
from .knowledge.manager import (
    DocumentAlreadyExistsError,
    DocumentReferenceMismatchError,
    DocumentSummary,
    DocumentTooLargeError,
    IncompleteIndexError,
    IndexOperationResult,
    IndexStatus,
    InvalidDocumentPathError,
    InvalidIndexManifestError,
    InvalidUploadEncodingError,
    InvalidUploadFilenameError,
    KnowledgeBaseService,
    UploadTooLargeError,
    load_knowledge_base_config,
)
from .schemas import AnswerResult, AnswerSource, AnswerStatus
from .session_store import (
    DuplicateSessionError,
    DuplicateTurnError,
    SessionRecord,
    SessionNotFoundError,
    SessionStoreError,
    SessionStoreLockedError,
    StoredTurn,
)
from .skill_loader import discover_skills
from .agent.trace import (
    TraceDecisionSource,
    TraceInitiator,
    TraceIntentKind,
    TraceOutcome,
    TraceStage,
    TraceStatus,
    TurnTrace,
    trace_to_dict,
)

# 第一阶段：应用生命周期与接口开发。

MAX_TITLE_LENGTH = 200
MAX_QUESTION_LENGTH = 10_000
MAX_PAGE_SIZE = 100
MAX_PREVIEW_LINES = 1_000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LOCAL_API_HOST = "127.0.0.1"
LOCAL_API_PORT = 8000
LOCAL_API_WORKERS = 1
_LOGGER = logging.getLogger(__name__)

SessionId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
    Field(
        description="会话 ID，只能包含字母、数字、点、下划线和连字符",
        examples=["demo-session-001"],
    ),
]
PageLimit = Annotated[
    int,
    Field(ge=1, le=MAX_PAGE_SIZE, description="本页最多返回的记录数"),
]
PageOffset = Annotated[
    int,
    Field(ge=0, description="按既定顺序跳过的记录数"),
]


class _PublicModel(BaseModel):
    """Base for the deliberately small, stable HTTP JSON contract."""

    model_config = ConfigDict(extra="forbid")


class CreateSessionRequest(_PublicModel):
    title: str | None = Field(
        default=None,
        max_length=MAX_TITLE_LENGTH,
        description="可选的会话标题；空白标题会被视为未提供",
        examples=["演示知识库问答"],
    )

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


class AskRequest(_PublicModel):
    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="要在当前会话中提出的问题",
        examples=["RAG 的基本流程是什么？"],
    )
    skill_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="本轮显式选择的技能 ID；不传时使用普通问答流程",
        examples=["concept-explanation"],
    )

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class SessionResponse(_PublicModel):
    session_id: SessionId
    title: str | None = None
    created_at: str
    updated_at: str


class SessionListResponse(_PublicModel):
    items: list[SessionResponse]
    limit: PageLimit
    offset: PageOffset


class SkillResponse(_PublicModel):
    skill_id: str
    name: str
    description: str


class SkillListResponse(_PublicModel):
    items: list[SkillResponse]


class AnswerSourceResponse(_PublicModel):
    chunk_id: str
    source_path: str
    heading_path: list[str]
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content_hash: str
    index_version: str
    retrieval_score: float | None = None


class TraceEventResponse(_PublicModel):
    """One observable execution stage, without messages or tool payloads."""

    sequence: int = Field(ge=0, strict=True)
    stage: TraceStage
    initiator: TraceInitiator
    status: TraceStatus
    started_offset_ms: int | None = Field(default=None, ge=0, strict=True)
    duration_ms: int | None = Field(default=None, ge=0, strict=True)
    tool_name: str | None = None
    tool_call_id: str | None = None
    result_count: int | None = Field(default=None, ge=0, strict=True)
    index_version: str | None = None
    error_code: str | None = None
    decision_source: TraceDecisionSource | None = None
    intent_kind: TraceIntentKind | None = None
    route_kind: Literal["knowledge", "mcp", "direct"] | None = None
    reason_code: str | None = None


class TurnTraceResponse(_PublicModel):
    """Completed turn summary shared by answers and persisted history."""

    schema_version: Literal[1] = 1
    trace_id: str
    started_at: str = Field(description="带 UTC 偏移的 ISO 8601 开始时间")
    duration_ms: int | None = Field(
        default=None,
        ge=0,
        strict=True,
        description="Agent 处理耗时，不含排队、会话提交与 HTTP 传输；缺失时为 null",
    )
    events: list[TraceEventResponse] = Field(default_factory=list)
    outcome: TraceOutcome
    requested_skill_id: str | None = None
    prepared_skill_id: str | None = None
    applied_skill_id: str | None = Field(
        default=None,
        description="已携带该技能指令开始模型调用；不代表调用成功或模型遵循全部指令",
    )
    error_code: str | None = None
    events_truncated: bool = False


class RouteSummaryResponse(_PublicModel):
    route: Literal["knowledge", "mcp", "direct"]
    intent: TraceIntentKind
    decision_source: TraceDecisionSource
    reason_code: str


class AssessmentQuestionResponse(_PublicModel):
    """The current question only; the stored questionnaire stays private."""

    id: str | int
    number: int = Field(ge=1, strict=True)
    text: str


class AssessmentOptionResponse(_PublicModel):
    value: str | int
    label: str


class AssessmentResponse(_PublicModel):
    """Public progress and result, without submitted answers or cached pages."""

    test_id: str
    status: Literal["active", "paused", "ready", "completed", "cancelled"]
    answered_count: int = Field(ge=0, strict=True)
    total: int = Field(ge=1, strict=True)
    current_question: AssessmentQuestionResponse | None = None
    options: list[AssessmentOptionResponse] = Field(default_factory=list)
    result: dict[str, Any] | None = None


class AnswerResponse(_PublicModel):
    session_id: SessionId
    question: str
    answer: str
    status: AnswerStatus
    sources: list[AnswerSourceResponse] = Field(default_factory=list)
    index_version: str | None = None
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    stop_reason: str | None = None
    trace: TurnTraceResponse | None = Field(
        default=None,
        description="本轮结束后的执行摘要；没有可用记录时为 null，不是实时进度",
    )
    routing: RouteSummaryResponse | None = None
    assessment: AssessmentResponse | None = Field(
        default=None,
        description="本轮测试进度、当前题目和结果；不包含已提交答案或内部问卷缓存",
    )


class ErrorResponse(_PublicModel):
    code: str
    message: str
    result: AnswerResponse | None = None
    persisted: bool | None = None
    details: list[dict[str, Any]] | None = None


class TurnResponse(_PublicModel):
    turn_id: str
    turn_no: int = Field(ge=1)
    question: str
    answer: str
    status: AnswerStatus
    sources: list[AnswerSourceResponse] = Field(default_factory=list)
    index_version: str | None = None
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    stop_reason: str | None = None
    trace: TurnTraceResponse | None = Field(
        default=None,
        description="随本轮回答保存的执行摘要；旧记录或不可用摘要为 null",
    )
    routing: RouteSummaryResponse | None = None
    assessment: AssessmentResponse | None = Field(
        default=None,
        description="该历史轮次保存的测试进度；旧记录或非测试轮次为 null",
    )


class TurnListResponse(_PublicModel):
    items: list[TurnResponse]
    limit: PageLimit
    offset: PageOffset


class ValidationIssueResponse(_PublicModel):
    loc: list[str | int]
    msg: str
    type: str


class ValidationErrorResponse(_PublicModel):
    detail: list[ValidationIssueResponse]


class HealthResponse(_PublicModel):
    service: str
    version: str
    status: Literal["ok"] = "ok"


class DocumentResponse(_PublicModel):
    path: str
    exists: bool
    size_bytes: int | None
    modified_at: str | None
    sync_status: str
    content_hash: str | None = None
    error: str | None = None


class DocumentListResponse(_PublicModel):
    items: list[DocumentResponse]
    limit: PageLimit
    offset: PageOffset


class DocumentPreviewResponse(_PublicModel):
    path: str
    content: str
    start_line: int = Field(ge=0)
    end_line: int = Field(ge=0)
    total_lines: int = Field(ge=0)
    content_hash: str
    truncated: bool


class DocumentChangesResponse(_PublicModel):
    added: int | None = Field(default=None, ge=0)
    modified: int | None = Field(default=None, ge=0)
    removed: int | None = Field(default=None, ge=0)
    unreadable: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)


class IndexStatusResponse(_PublicModel):
    disk_status: str
    published_version: str | None
    built_at: str | None
    disk_document_count: int | None = Field(default=None, ge=0)
    indexed_document_count: int | None = Field(default=None, ge=0)
    chunk_count: int | None = Field(default=None, ge=0)
    pending_changes: DocumentChangesResponse
    active_version: str | None = None
    error: str | None = None


class IndexOperationResponse(_PublicModel):
    operation: Literal["build", "update"]
    succeeded: bool
    published: bool | None
    refreshed: bool | None
    published_version: str | None = None
    active_version: str | None = None
    discovered_files: int | None = Field(default=None, ge=0)
    read_files: int | None = Field(default=None, ge=0)
    chunk_count: int | None = Field(default=None, ge=0)
    changed_file_count: int | None = Field(default=None, ge=0)
    full_rebuild: bool | None = None
    reused_chunks: int | None = Field(default=None, ge=0)
    embedded_chunks: int | None = Field(default=None, ge=0)
    removed_chunks: int | None = Field(default=None, ge=0)
    error_category: str | None = None


def _error_content(
    code: str,
    message: str,
    *,
    result: AnswerResponse | None = None,
    persisted: bool | None = None,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return ErrorResponse(
        code=code,
        message=message,
        result=result,
        persisted=persisted,
        details=details,
    ).model_dump(mode="json", exclude_none=True)


RuntimeFactory = Callable[[], Any]
StoreFactory = Callable[[], Any]
ApplicationFactory = Callable[[Any], Any]


def _validated_cors_origins(origins: tuple[str, ...]) -> tuple[str, ...]:
    """Return unique, explicit browser origins suitable for local web clients."""
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


class ServiceBusyError(RuntimeError):
    """Raised when a business operation is already running."""

    code = "service_busy"
    status_code = 503

    def __init__(self) -> None:
        super().__init__("The service is busy; try again after the current request finishes")


def create_app(
    runtime_factory: RuntimeFactory | None = None,
    *,
    cors_origins: tuple[str, ...] = (),
) -> FastAPI:
    """Serve the local UI and API; additional browser origins are opt-in."""

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
            "当前不提供流式输出。会话和已完成问答持久化在 SQLite 中；停止服务不会"
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

    # 只公开网页资源目录；仓库、配置、文档和数据库不通过静态路由暴露。
    web_directory = PROJECT_ROOT / "web"
    application.mount(
        "/assets",
        StaticFiles(directory=web_directory / "assets", check_dir=False),
        name="assets",
    )

    @application.get("/", response_class=FileResponse, include_in_schema=False)
    async def web_index() -> FileResponse:
        page = web_directory / "index.html"
        if not page.is_file():
            # 网页缺失时保留普通 404，不使用会话专属错误码或覆盖 API 路由。
            raise StarletteHTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return FileResponse(page, media_type="text/html")

    @application.get(
        "/skills",
        response_model=SkillListResponse,
        summary="列出可手动选择的技能",
    )
    async def list_skills() -> SkillListResponse:
        return SkillListResponse(
            items=[
                SkillResponse(
                    skill_id=skill.skill_id,
                    name=skill.name,
                    description=skill.description,
                )
                for skill in discover_skills()
            ]
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "loc": list(issue.get("loc", ())),
                "msg": str(issue.get("msg", "Invalid value")),
                "type": str(issue.get("type", "value_error")),
            }
            for issue in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_content(
                "validation_error",
                "The request parameters are invalid",
                details=details,
            ),
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        if (
            exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            and isinstance(exc.detail, str)
            and exc.detail.startswith("Unknown skill:")
        ):
            return JSONResponse(
                status_code=exc.status_code,
                content=_error_content("unknown_skill", exc.detail),
                headers=exc.headers,
            )
        known_errors = {
            status.HTTP_404_NOT_FOUND: ("session_not_found", "Session not found"),
            status.HTTP_409_CONFLICT: ("conflict", "The request conflicts with current state"),
            status.HTTP_422_UNPROCESSABLE_ENTITY: (
                "validation_error",
                "The request parameters are invalid",
            ),
        }
        code, message = known_errors.get(
            exc.status_code, ("http_error", "The request could not be completed")
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_content(code, message),
            headers=exc.headers,
        )

    @application.exception_handler(TurnPersistenceError)
    async def turn_persistence_error_handler(
        request: Request, exc: TurnPersistenceError
    ) -> JSONResponse:
        locked = isinstance(exc.__cause__, SessionStoreLockedError)
        return JSONResponse(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
                if locked
                else status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            content=_error_content(
                "persistence_unavailable" if locked else "persistence_failed",
                "The answer was generated, but the conversation turn was not saved",
                persisted=False,
                result=_answer_response(
                    str(request.path_params.get("session_id", "unknown")),
                    exc.result,
                ),
            ),
        )

    async def conflict_error_handler(
        _request: Request, _exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=_error_content(
                "conflict", "The session or conversation turn conflicts with current state"
            ),
        )

    for exception_type in (
        ApplicationBusyError,
        DuplicateSessionError,
        DuplicateTurnError,
    ):
        application.add_exception_handler(exception_type, conflict_error_handler)

    @application.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(
        _request: Request, _exc: SessionNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=_error_content("session_not_found", "Session not found"),
        )

    async def unavailable_error_handler(
        _request: Request, _exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_content(
                "service_unavailable", "The service is temporarily unavailable"
            ),
        )

    for exception_type in (ServiceBusyError, SessionStoreLockedError):
        application.add_exception_handler(exception_type, unavailable_error_handler)

    def knowledge_error(
        status_code: int, code: str, message: str
    ) -> Callable[[Request, Exception], Any]:
        async def handler(_request: Request, _exc: Exception) -> JSONResponse:
            return JSONResponse(
                status_code=status_code,
                content=_error_content(code, message),
            )

        return handler

    knowledge_error_mappings = (
        (FileNotFoundError, 404, "document_not_found", "Document not found"),
        (InvalidDocumentPathError, 422, "invalid_document_path", "The document path is invalid"),
        (DocumentReferenceMismatchError, 409, "document_changed", "The document no longer matches the referenced content"),
        (DocumentAlreadyExistsError, 409, "document_exists", "A document with that name already exists"),
        (InvalidUploadFilenameError, 422, "invalid_filename", "The upload filename is invalid"),
        (InvalidUploadEncodingError, 422, "invalid_encoding", "The document must use UTF-8 encoding"),
        (UploadTooLargeError, 413, "upload_too_large", "The uploaded document is too large"),
        (DocumentTooLargeError, 413, "document_too_large", "The document is too large to preview"),
        (IncompleteIndexError, 500, "index_incomplete", "The published index is incomplete"),
        (InvalidIndexManifestError, 500, "index_corrupt", "The published index metadata is invalid"),
        (PermissionError, 500, "knowledge_storage_error", "Knowledge-base storage is unavailable"),
        (NotADirectoryError, 500, "knowledge_storage_error", "Knowledge-base storage is unavailable"),
    )
    for exception_type, status_code, code, message in knowledge_error_mappings:
        application.add_exception_handler(
            exception_type, knowledge_error(status_code, code, message)
        )

    @application.exception_handler(ChatModelTimeoutError)
    async def model_timeout_handler(
        _request: Request, _exc: ChatModelTimeoutError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=_error_content("model_timeout", "The model response timed out"),
        )

    @application.exception_handler(ChatModelAuthenticationError)
    async def model_authentication_error_handler(
        _request: Request, _exc: ChatModelAuthenticationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_content(
                "service_configuration_error",
                "The model service is not configured correctly",
            ),
        )

    @application.exception_handler(ChatModelInvocationError)
    async def model_invocation_error_handler(
        _request: Request, _exc: ChatModelInvocationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_content("service_error", "The model service is unavailable"),
        )

    @application.exception_handler(SessionStoreError)
    async def storage_error_handler(
        _request: Request, _exc: SessionStoreError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_content("storage_error", "Session storage failed"),
        )

    @application.exception_handler(Exception)
    async def internal_error_handler(
        _request: Request, _exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_content("internal_error", "An internal service error occurred"),
        )

    @application.get(
        "/health",
        response_model=HealthResponse,
        summary="检查服务状态",
        description="仅返回服务名称、版本和存活状态，不调用模型或知识库。",
    )
    async def health() -> HealthResponse:
        return HealthResponse(
            service=application.title,
            version=application.version,
        )

    @application.get(
        "/knowledge/documents",
        response_model=DocumentListResponse,
        summary="分页列出知识库文档",
    )
    async def list_knowledge_documents(
        limit: PageLimit = 50,
        offset: PageOffset = 0,
    ) -> DocumentListResponse:
        runtime: ServiceRuntime = application.state.runtime
        documents = await runtime.run_knowledge(
            lambda service: service.list_documents(offset=offset, limit=limit)
        )
        return DocumentListResponse(
            items=[_document_response(item) for item in documents],
            limit=limit,
            offset=offset,
        )

    @application.get(
        "/knowledge/documents/content",
        response_model=DocumentPreviewResponse,
        summary="预览受控知识库原文",
    )
    async def read_knowledge_document(
        path: Annotated[str, Query(min_length=1)],
        start_line: Annotated[int, Query(ge=1)] = 1,
        limit: Annotated[int, Query(ge=1, le=MAX_PREVIEW_LINES)] = 200,
        content_hash: Annotated[str | None, Query(min_length=1)] = None,
    ) -> DocumentPreviewResponse:
        runtime: ServiceRuntime = application.state.runtime
        preview = await runtime.run_knowledge(
            lambda service: service.read_document(
                path,
                start_line=start_line,
                limit=limit,
                content_hash=content_hash,
            )
        )
        return DocumentPreviewResponse.model_validate(preview, from_attributes=True)

    @application.get(
        "/knowledge/status",
        response_model=IndexStatusResponse,
        summary="读取知识库索引状态",
    )
    async def knowledge_status() -> IndexStatusResponse:
        runtime: ServiceRuntime = application.state.runtime
        return _index_status_response(await runtime.get_knowledge_status())

    @application.post(
        "/knowledge/documents",
        response_model=DocumentResponse,
        status_code=status.HTTP_201_CREATED,
        summary="上传新的 Markdown 文档",
    )
    async def upload_knowledge_document(
        filename: Annotated[str, Query(min_length=1, max_length=255)],
        content: Annotated[bytes, Body(media_type="text/markdown")],
    ) -> DocumentResponse:
        if len(content) > MAX_UPLOAD_BYTES:
            raise UploadTooLargeError("upload exceeds the HTTP body limit")
        runtime: ServiceRuntime = application.state.runtime
        document = await runtime.run_knowledge(
            lambda service: service.upload_document(filename, content)
        )
        return _document_response(document)

    async def run_index_operation(operation: Literal["build", "update"]) -> Any:
        runtime: ServiceRuntime = application.state.runtime
        result = await runtime.run_knowledge(
            lambda service: (
                service.build_index() if operation == "build" else service.update_index()
            )
        )
        response = _index_operation_response(result)
        if result.succeeded:
            return response
        refresh_failed = result.error_category == "refresh"
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                **_error_content(
                    "index_refresh_failed" if refresh_failed else "index_ingest_failed",
                    (
                        "The index was published, but the running service could not reload it"
                        if refresh_failed
                        else "The index operation failed"
                    ),
                ),
                "operation": response.model_dump(mode="json", exclude_none=True),
            },
        )

    @application.post(
        "/knowledge/index/build",
        response_model=IndexOperationResponse,
        summary="完整重建知识库索引",
    )
    async def build_knowledge_index() -> Any:
        return await run_index_operation("build")

    @application.post(
        "/knowledge/index/update",
        response_model=IndexOperationResponse,
        summary="增量更新知识库索引",
    )
    async def update_knowledge_index() -> Any:
        return await run_index_operation("update")

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
            if request.skill_id is None:
                result = await runtime.ask_session(session_id, request.question)
            else:
                result = await runtime.ask_session(
                    session_id, request.question, skill_id=request.skill_id
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

        if result.stop_reason == "index_not_built":
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            code = "index_not_built"
            message = (
                "The knowledge index has not been built; run `person-a build` "
                "and restart the API service"
            )
        elif result.stop_reason == "model_timeout":
            status_code = status.HTTP_504_GATEWAY_TIMEOUT
            code = "model_timeout"
            message = "The model response timed out"
        elif result.stop_reason == "model_authentication_failed":
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            code = "service_configuration_error"
            message = "The model service is not configured correctly"
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            code = "service_error"
            message = "The model service is unavailable"
        return JSONResponse(
            status_code=status_code,
            content=_error_content(code, message, result=response),
        )

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

    application.state.runtime_factory = runtime_factory
    application.state.runtime = None
    return application

class ServiceRuntime:
    """Run all stateful service work on one long-lived worker thread."""

    def __init__(
        self,
        store_factory: StoreFactory | None = None,
        application_factory: ApplicationFactory | None = None,
    ) -> None:
        self._store_factory = store_factory or self._default_store_factory
        self._application_factory = (
            application_factory or self._default_application_factory
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="knowledge-agent-service",
        )
        self._state_lock = Lock()
        self._worker_thread_id: int | None = None
        self._store: Any = None
        self._application: Any = None
        self._knowledge_service: KnowledgeBaseService | None = None
        self._initialized = False
        self._busy = False
        self._closing = False
        self._closed = False

    @staticmethod
    def _default_store_factory() -> Any:
        from .config import get_session_store_settings
        from .session_store import SQLiteSessionStore

        settings = get_session_store_settings()
        return SQLiteSessionStore(
            settings.db_path,
            lock_timeout=settings.lock_timeout,
        )

    @staticmethod
    def _default_application_factory(store: Any) -> Any:
        from .mcp.bootstrap import create_configured_application

        return create_configured_application(session_store=store)

    def initialize(self) -> None:
        """Construct and initialize storage in the dedicated worker thread."""
        with self._state_lock:
            if self._closed or self._closing:
                raise RuntimeError("service runtime is closed")
            if self._initialized:
                return

        try:
            self._executor.submit(self._initialize_on_worker).result()
        except BaseException:
            with self._state_lock:
                self._closing = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            with self._state_lock:
                self._closed = True
                self._closing = False
            raise

        with self._state_lock:
            self._initialized = True

    def _initialize_on_worker(self) -> None:
        self._worker_thread_id = get_ident()
        store = self._store_factory()
        try:
            store.initialize()
        except BaseException:
            store.close()
            raise
        self._store = store

    def _require_worker_thread(self) -> None:
        if get_ident() != self._worker_thread_id:
            raise RuntimeError("runtime resources may only be used by the worker thread")

    @property
    def store(self) -> Any:
        """Return worker-owned storage while executing a submitted task."""
        self._require_worker_thread()
        if self._store is None:
            raise RuntimeError("service runtime is not initialized")
        return self._store

    @property
    def application(self) -> Any:
        """Lazily create the agent application on its owning worker thread."""
        self._require_worker_thread()
        if self._store is None:
            raise RuntimeError("service runtime is not initialized")
        if self._application is None:
            self._application = self._application_factory(self._store)
        return self._application

    @property
    def knowledge_service(self) -> KnowledgeBaseService:
        """Lazily construct the filesystem service on its owning worker."""
        self._require_worker_thread()
        if self._knowledge_service is None:
            self._knowledge_service = KnowledgeBaseService(
                load_knowledge_base_config(),
                refresh_callback=self._refresh_index_on_worker,
            )
        return self._knowledge_service

    def _refresh_index_on_worker(self) -> Any:
        self._require_worker_thread()
        return self.application.refresh_index()

    def submit(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        """Submit one complete business operation to the dedicated worker."""
        with self._state_lock:
            if not self._initialized:
                raise RuntimeError("service runtime is not initialized")
            if self._closing or self._closed:
                raise RuntimeError("service runtime is closed")
            if self._busy:
                raise ServiceBusyError()
            self._busy = True

            try:
                future = self._executor.submit(function, *args, **kwargs)
            except BaseException:
                self._busy = False
                raise

        # A very short operation may already be complete here. Register outside
        # the state lock because Future then invokes the callback immediately.
        future.add_done_callback(self._operation_finished)
        return future

    def _operation_finished(self, _future: Future[Any]) -> None:
        with self._state_lock:
            self._busy = False

    @property
    def busy(self) -> bool:
        """Return whether a submitted business operation is still running."""
        with self._state_lock:
            return self._busy

    @property
    def closing(self) -> bool:
        """Return whether runtime shutdown has started."""
        with self._state_lock:
            return self._closing

    async def run(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Submit and asynchronously wait for one complete business operation."""
        future = self.submit(function, *args, **kwargs)
        return await self._wait_for_future(future)

    @staticmethod
    async def _wait_for_future(future: Future[Any]) -> Any:
        wrapped = asyncio.wrap_future(future)
        try:
            # Cancelling an HTTP waiter must not cancel work already handed to the
            # dedicated worker. Its completion callback owns the busy transition.
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            # Retrieve a later exception when there is no longer an HTTP waiter.
            wrapped.add_done_callback(ServiceRuntime._consume_future_exception)
            raise

    @staticmethod
    def _consume_future_exception(future: asyncio.Future[Any]) -> None:
        if not future.cancelled():
            future.exception()

    def submit_store(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        """Submit an operation whose first argument is the session store."""
        return self.submit(self._call_with_store, function, args, kwargs)

    async def run_store(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Run and asynchronously wait for a session-store operation."""
        return await self._wait_for_future(
            self.submit_store(function, *args, **kwargs)
        )

    def _call_with_store(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return function(self.store, *args, **kwargs)

    def submit_application(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        """Submit an operation and lazily supply the agent application."""
        return self.submit(self._call_with_application, function, args, kwargs)

    async def run_application(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Run and asynchronously wait for an agent-application operation."""
        return await self._wait_for_future(
            self.submit_application(function, *args, **kwargs)
        )

    def _call_with_application(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return function(self.application, *args, **kwargs)

    async def run_knowledge(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Run one knowledge-base operation on the shared service worker."""
        return await self.run(self._call_with_knowledge, function, args, kwargs)

    def _call_with_knowledge(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return function(self.knowledge_service, *args, **kwargs)

    async def get_knowledge_status(self) -> IndexStatus:
        return await self.run(self._get_knowledge_status_on_worker)

    def _get_knowledge_status_on_worker(self) -> IndexStatus:
        from dataclasses import replace

        result = self.knowledge_service.get_index_status()
        application = self._application
        if application is None:
            return result
        try:
            active_version = application.agent.retriever.index_version
        except Exception:
            return result
        return replace(result, active_version=active_version)

    async def ask_session(
        self,
        session_id: str,
        question: str,
        *,
        skill_id: str | None = None,
    ) -> Any | None:
        """Ask one question in an existing persisted session."""
        return await self.run(
            self._ask_session_on_worker, session_id, question, skill_id
        )

    def _ask_session_on_worker(
        self, session_id: str, question: str, skill_id: str | None = None
    ) -> Any | None:
        self._require_worker_thread()
        if self.store.get_session(session_id) is None:
            return None

        application = self.application
        application.resume_session(session_id)
        try:
            return application.ask(question, skill_id=skill_id)
        finally:
            application.end_session()

    async def load_turns(
        self, session_id: str, *, limit: int, offset: int
    ) -> Any | None:
        """Load one chronological page, or return ``None`` if absent."""
        return await self.run(
            self._load_turns_on_worker,
            session_id,
            limit,
            offset,
        )

    def _load_turns_on_worker(
        self, session_id: str, limit: int, offset: int
    ) -> Any | None:
        self._require_worker_thread()
        if self.store.get_session(session_id) is None:
            return None
        return self.store.load_turns(session_id, limit=limit, offset=offset)

    async def clear_session(self, session_id: str) -> bool:
        """Clear persisted turns and release matching in-memory history."""
        return await self.run(self._clear_session_on_worker, session_id)

    def _clear_session_on_worker(self, session_id: str) -> bool:
        self._require_worker_thread()
        cleared = self.store.clear_session(session_id)
        application = self._application
        if cleared and application is not None and application.session_id == session_id:
            application.end_session()
        return cleared

    async def delete_session(self, session_id: str) -> bool:
        """Delete a persisted session and release matching in-memory history."""
        return await self.run(self._delete_session_on_worker, session_id)

    def _delete_session_on_worker(self, session_id: str) -> bool:
        self._require_worker_thread()
        deleted = self.store.delete_session(session_id)
        application = self._application
        if deleted and application is not None and application.session_id == session_id:
            application.end_session()
        return deleted

    def close(self) -> None:
        """Drain pending work, close owned resources, and stop the worker."""
        with self._state_lock:
            if self._closed or self._closing:
                return
            self._closing = True
            initialized = self._initialized

        try:
            if initialized:
                self._executor.submit(self._close_on_worker).result()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
            with self._state_lock:
                self._initialized = False
                self._closed = True
                self._closing = False

    def _close_on_worker(self) -> None:
        self._require_worker_thread()
        application = self._application
        store = self._store
        self._application = None
        self._knowledge_service = None
        self._store = None
        if application is not None:
            # The application owns the injected store and closes it itself.
            application.close()
        elif store is not None:
            store.close()

    def __enter__(self) -> ServiceRuntime:
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _source_response(source: AnswerSource) -> AnswerSourceResponse:
    """Copy only the source metadata that belongs to the public API."""
    return AnswerSourceResponse(
        chunk_id=source.chunk_id,
        source_path=source.source_path,
        heading_path=list(source.heading_path),
        start_line=source.start_line,
        end_line=source.end_line,
        content_hash=source.content_hash,
        index_version=source.index_version,
        retrieval_score=source.retrieval_score,
    )


def _session_response(record: SessionRecord) -> SessionResponse:
    """Convert persisted session metadata to its stable public shape."""
    return SessionResponse(
        session_id=record.session_id,
        title=record.title,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _trace_response(trace: TurnTrace | None) -> TurnTraceResponse | None:
    """Use the trace allow-list; a malformed optional summary cannot hide an answer."""
    if trace is None:
        return None
    try:
        return TurnTraceResponse.model_validate(trace_to_dict(trace))
    except (TypeError, ValueError, AttributeError):
        # This conversion has no business side effects. Do not log trace content,
        # retry a turn, or turn a usable answer into an HTTP failure.
        _LOGGER.warning("trace_response_unavailable")
        return None


def _assessment_response(
    snapshot: dict[str, Any] | None,
) -> AssessmentResponse | None:
    """Expose only the assessment allow-list, never the stored answer payload."""
    if snapshot is None:
        return None
    from .agent.assessment import public_assessment

    public = public_assessment(snapshot)
    if public is None:
        return None
    return AssessmentResponse.model_validate(public)


def _answer_response(session_id: str, result: AnswerResult) -> AnswerResponse:
    """Convert an answer without serializing any Agent or model state."""
    return AnswerResponse(
        session_id=session_id,
        question=result.question,
        answer=result.answer,
        status=result.status,
        sources=[_source_response(source) for source in result.sources],
        index_version=result.index_version,
        model_calls=result.model_calls,
        tool_calls=result.tool_calls,
        stop_reason=result.stop_reason,
        trace=_trace_response(result.trace),
        routing=(
            RouteSummaryResponse.model_validate(result.routing, from_attributes=True)
            if result.routing is not None
            else None
        ),
        assessment=_assessment_response(result.assessment_state),
    )


def _turn_response(turn: StoredTurn) -> TurnResponse:
    """Expose a stored result as-is while omitting its internal messages."""
    result = turn.result
    return TurnResponse(
        turn_id=turn.turn_id,
        turn_no=turn.turn_no,
        question=turn.question,
        answer=result.answer,
        status=result.status,
        sources=[_source_response(source) for source in result.sources],
        index_version=result.index_version,
        model_calls=result.model_calls,
        tool_calls=result.tool_calls,
        stop_reason=result.stop_reason,
        trace=_trace_response(result.trace),
        routing=(
            RouteSummaryResponse.model_validate(result.routing, from_attributes=True)
            if result.routing is not None
            else None
        ),
        assessment=_assessment_response(result.assessment_state),
    )


def _document_response(document: DocumentSummary) -> DocumentResponse:
    response = DocumentResponse.model_validate(document, from_attributes=True)
    if response.error is None:
        return response
    return response.model_copy(update={"error": "The document could not be read"})


def _index_status_response(result: IndexStatus) -> IndexStatusResponse:
    changes = result.pending_changes
    return IndexStatusResponse(
        disk_status=result.disk_status,
        published_version=result.published_version,
        built_at=result.built_at,
        disk_document_count=result.disk_document_count,
        indexed_document_count=result.indexed_document_count,
        chunk_count=result.chunk_count,
        pending_changes=DocumentChangesResponse(
            added=changes.added,
            modified=changes.modified,
            removed=changes.removed,
            unreadable=changes.unreadable,
            total=changes.total,
        ),
        active_version=result.active_version,
        error=(
            None
            if result.error is None
            else {
                "not_built": "The knowledge index has not been built",
                "incomplete": "The published index is incomplete",
                "corrupt": "The published index metadata is invalid",
            }.get(result.disk_status, "The index status could not be determined")
        ),
    )


def _index_operation_response(result: IndexOperationResult) -> IndexOperationResponse:
    ingest = result.ingest_result
    return IndexOperationResponse(
        operation=result.operation,
        succeeded=result.succeeded,
        published=result.published,
        refreshed=result.refreshed,
        published_version=result.published_version,
        active_version=result.active_version,
        discovered_files=getattr(ingest, "discovered_files", None),
        read_files=getattr(ingest, "read_files", None),
        chunk_count=getattr(ingest, "chunk_count", None),
        changed_file_count=getattr(ingest, "changed_file_count", None),
        full_rebuild=getattr(ingest, "full_rebuild", None),
        reused_chunks=getattr(ingest, "reused_chunks", None),
        embedded_chunks=getattr(ingest, "embedded_chunks", None),
        removed_chunks=getattr(ingest, "removed_chunks", None),
        error_category=result.error_category,
    )

# 第二阶段：开发完成后集中验证；本阶段不执行。

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
