"""共享数据结构。

文档片段：片段 ID、原文、完整相对文件名、标题层级、
原文起止行号、内容指纹及内容类型。
检索结果：片段、检索渠道、排名和相似度等信息。
回答结果：回答正文、实际使用的来源、资料不足说明。
检索分数不得作为经过校准的答案正确概率。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from knowledge_agent.observability.trace import TurnTrace


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """保留原文的 Markdown 片段，附带可追溯的元数据。

    ``text`` 始终是逐字的原文切片。标题信息单独保存，
    因此把它加入 Embedding/检索输入不会改变引用内容。
    ``chunk_id`` 由稳定 ID 阶段填充。
    """

    text: str
    source_path: str
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    content_hash: str
    content_type: Literal["markdown"] = "markdown"
    chunk_id: str | None = None

    @property
    def retrieval_text(self) -> str:
        """返回用于检索的文本，同时保持被引用的原文不变。"""
        if not self.heading_path:
            return self.text
        return f"{' > '.join(self.heading_path)}\n\n{self.text}"


RetrievalChannel = Literal["vector", "keyword"]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """索引快照中一条已排序、保留原文的检索结果。"""

    chunk_id: str
    chunk: DocumentChunk
    channels: tuple[RetrievalChannel, ...]
    fused_score: float
    index_version: str
    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_score: float | None = None
    keyword_score: float | None = None


AnswerStatus = Literal[
    "success",
    "insufficient_evidence",
    "needs_clarification",
    "out_of_scope",
    "service_error",
    "limit_reached",
]

RouteName = Literal["knowledge", "mcp", "direct", "chat"]
IntentName = Literal[
    "chat",
    "knowledge",
    "mcp_tool",
    "greeting",
    "scope",
    "needs_clarification",
    "out_of_scope",
]
DecisionSource = Literal["rules", "model", "fallback"]


@dataclass(frozen=True, slots=True)
class DialogueSummary:
    """对本轮单次对话决策的展示安全描述。"""

    action: str
    business_type: str | None
    execution_mode: str
    reason_code: str


@dataclass(frozen=True, slots=True)
class BusinessStateSummary:
    """对最新已提交业务状态的展示安全投影。"""

    business_type: str | None
    phase: str
    waiting_for_input: bool
    missing_slots: tuple[str, ...] = ()
    clarification_question: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    allowed_actions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RouteSummary:
    """一个已完成轮次的稳定公开路由元数据。"""

    route: RouteName
    intent: IntentName
    decision_source: DecisionSource
    reason_code: str


@dataclass(frozen=True, slots=True)
class AnswerSource:
    """实际提供给回答并被引用的来源。"""

    chunk_id: str
    source_path: str
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    content_hash: str
    index_version: str
    retrieval_score: float | None = None

    @classmethod
    def from_retrieval_result(cls, result: RetrievalResult) -> "AnswerSource":
        chunk = result.chunk
        return cls(
            chunk_id=result.chunk_id,
            source_path=chunk.source_path,
            heading_path=chunk.heading_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            content_hash=chunk.content_hash,
            index_version=result.index_version,
            retrieval_score=result.fused_score,
        )


DeclaredAnswerStatus = Literal[
    "supported", "partial", "insufficient_evidence", "needs_clarification", "out_of_scope",
]
AnswerValidationStatus = Literal["passed", "warning", "rejected"]


@dataclass(frozen=True, slots=True)
class AnswerDiagnostics:
    """将模型意图与校验结果、回答的业务状态分开。

    有界的最终文本摘录属于私有持久化数据，绝不会进入模型历史、
    公开 JSON 或 trace 进度。模型推理字段一律排除。
    """

    declared_status: DeclaredAnswerStatus | None
    validation_status: AnswerValidationStatus
    validation_reason: str | None
    supplied_source_count: int
    accepted_source_count: int
    model_final_excerpt: str | None = field(default=None, repr=False)
    excerpt_truncated: bool = False


@dataclass(frozen=True, slots=True)
class StepAnswerDiagnostics:
    """单个叶节点的私有诊断；公开计划数据使用白名单投影。"""

    step_id: str
    diagnostics: AnswerDiagnostics


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """结构化回答，可附带一份私有的、已持久化的测评快照。

    HTTP 响应必须用 ``public_assessment`` 转换 ``assessment_state``；
    该快照包含已提交的答案，绝不能作为公开响应体。
    """

    question: str
    answer: str
    status: AnswerStatus
    sources: tuple[AnswerSource, ...] = ()
    index_version: str | None = None
    model_calls: int = 0
    tool_calls: int = 0
    stop_reason: str | None = None
    trace: TurnTrace | None = None
    routing: RouteSummary | None = None
    assessment_state: dict[str, Any] | None = None
    # 仅公开执行摘要：目标、依赖关系、结果和计数。
    # 该字段绝不存储模型推理或原始工具响应。
    plan: dict[str, Any] | None = None
    answer_diagnostics: AnswerDiagnostics | None = None
    step_answer_diagnostics: tuple[StepAnswerDiagnostics, ...] = field(default=(), repr=False)
    dialogue: DialogueSummary | None = None
    business_state: BusinessStateSummary | None = None
