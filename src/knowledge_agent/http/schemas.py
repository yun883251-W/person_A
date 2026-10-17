"""HTTP 请求、响应与参数约束，不依赖路由或应用生命周期。"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from ..agent.skill_selector import normalize_skill_request
from ..agent.trace import (
    TraceDecisionSource,
    TraceInitiator,
    TraceIntentKind,
    TraceOutcome,
    TraceSkillMode,
    TraceStage,
    TraceStatus,
)
from ..schemas import AnswerStatus, AnswerValidationStatus, DeclaredAnswerStatus

MAX_TITLE_LENGTH = 200
MAX_QUESTION_LENGTH = 10_000
MAX_PAGE_SIZE = 100
MAX_PREVIEW_LINES = 1_000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

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
        max_length=256,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="手动模式的技能 ID；省略 skill_mode 时提供此字段仍按旧版手动模式处理",
        examples=["concept-explanation"],
    )
    skill_mode: TraceSkillMode | None = Field(
        default=None,
        description="off 普通问答、auto 自动选择、manual 手动选择；省略时按 skill_id 兼容旧行为",
    )

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_skill_request(self) -> AskRequest:
        # Validate the same contract as CLI/Agent without loading skill files or
        # turning an omitted mode into an implicit request for automatic routing.
        normalize_skill_request(self.skill_mode, self.skill_id)
        return self


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
    skill_mode: TraceSkillMode | None = None
    selected_skill_id: str | None = None
    strategy_id: str | None = None
    step_id: str | None = None
    declared_status: DeclaredAnswerStatus | None = None
    validation_status: AnswerValidationStatus | None = None
    validation_reason: str | None = None
    supplied_source_count: int | None = Field(default=None, ge=0, strict=True)
    dialogue_action: Literal[
        "new_business", "continue_business", "cancel_business", "pause_business",
        "resume_business", "chitchat", "unclear", "unsupported",
    ] | None = None
    business_type: Literal["knowledge", "bilibili_search", "mbti", "mcp_tool"] | None = None
    execution_mode: Literal["single", "plan"] | None = None


class AnswerDiagnosticsResponse(_PublicModel):
    """Public status/validation facts only; private model text has no API field."""

    declared_status: DeclaredAnswerStatus | None = None
    validation_status: AnswerValidationStatus
    validation_reason: str | None = None
    supplied_source_count: int = Field(ge=0, strict=True)
    accepted_source_count: int = Field(ge=0, strict=True)


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


class DialogueSummaryResponse(_PublicModel):
    action: Literal[
        "new_business", "continue_business", "cancel_business", "pause_business",
        "resume_business", "chitchat", "unclear", "unsupported",
    ]
    business_type: Literal["knowledge", "bilibili_search", "mbti", "mcp_tool"] | None = None
    execution_mode: Literal["single", "plan"]
    reason_code: str


class BusinessStateResponse(_PublicModel):
    business_type: Literal["knowledge", "bilibili_search", "mbti", "mcp_tool"] | None = None
    phase: Literal[
        "idle", "awaiting_input", "ready", "active", "paused", "dispatching",
        "completed", "cancelled", "execution_unknown",
    ]
    waiting_for_input: bool
    missing_slots: list[str] = Field(default_factory=list)
    clarification_question: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    allowed_actions: list[str] = Field(default_factory=list)


class SessionBusinessStateResponse(_PublicModel):
    session_id: SessionId
    state: BusinessStateResponse


class AssessmentQuestionResponse(_PublicModel):
    """The current question only; the stored questionnaire stays private."""

    id: int = Field(ge=0, le=47, strict=True)
    number: int = Field(ge=1, strict=True)
    text: str


class AssessmentOptionResponse(_PublicModel):
    value: int = Field(ge=1, le=5, strict=True)
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


class TopicBindingResponse(_PublicModel):
    """Same-request user-topic reference, distinct from an answer dependency."""

    topic: str = Field(min_length=1, max_length=200)
    source_step_id: Literal["P1", "P2"]


class PlanStepResponse(_PublicModel):
    """Observable work and outcome for one bounded task step."""

    step_id: str
    goal: str
    kind: Literal["knowledge", "mcp"]
    depends_on: list[str]
    status: Literal[
        "pending", "success", "insufficient_evidence", "needs_clarification",
        "service_error", "limit_reached", "out_of_scope", "skipped",
    ]
    model_calls: int = Field(ge=0, strict=True)
    tool_calls: int = Field(ge=0, strict=True)
    stop_reason: str | None = None
    answer_diagnostics: AnswerDiagnosticsResponse | None = None
    topic_binding: TopicBindingResponse | None = None


class PlanSummaryResponse(_PublicModel):
    """Final task summary; excludes internal prompts and model reasoning."""

    version: Literal[1]
    status: Literal["completed", "partial", "failed", "needs_clarification"]
    steps: list[PlanStepResponse]


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
    answer_diagnostics: AnswerDiagnosticsResponse | None = None
    trace: TurnTraceResponse | None = Field(
        default=None,
        description="本轮结束后的执行摘要；没有可用记录时为 null，不是实时进度",
    )
    routing: RouteSummaryResponse | None = None
    dialogue: DialogueSummaryResponse | None = None
    business_state: BusinessStateResponse | None = None
    assessment: AssessmentResponse | None = Field(
        default=None,
        description="本轮测试进度、当前题目和结果；不包含已提交答案或内部问卷缓存",
    )
    plan: PlanSummaryResponse | None = Field(
        default=None,
        description="本轮多步骤任务的执行摘要；只包含目标、依赖、状态和调用统计",
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
    answer_diagnostics: AnswerDiagnosticsResponse | None = None
    trace: TurnTraceResponse | None = Field(
        default=None,
        description="随本轮回答保存的执行摘要；旧记录或不可用摘要为 null",
    )
    routing: RouteSummaryResponse | None = None
    dialogue: DialogueSummaryResponse | None = None
    business_state: BusinessStateResponse | None = None
    assessment: AssessmentResponse | None = Field(
        default=None,
        description="该历史轮次保存的测试进度；旧记录或非测试轮次为 null",
    )
    plan: PlanSummaryResponse | None = Field(
        default=None,
        description="随回答保存的多步骤任务摘要；旧记录或普通单步问答为 null",
    )


class TurnListResponse(_PublicModel):
    items: list[TurnResponse]
    limit: PageLimit
    offset: PageOffset


class SessionSummaryResponse(_PublicModel):
    session_id: str
    summary: str = ""
    covered_turn_no: int = Field(default=0, ge=0)
    updated_at: str | None = None


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
