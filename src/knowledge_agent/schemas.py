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
    """A source-preserving Markdown chunk with traceable metadata.

    ``text`` is always the verbatim source slice.  Heading information remains
    separate so adding it to embedding/search input cannot change citations.
    ``chunk_id`` is populated by the stable-ID stage.
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
        """Return search text while keeping the cited source text untouched."""
        if not self.heading_path:
            return self.text
        return f"{' > '.join(self.heading_path)}\n\n{self.text}"


RetrievalChannel = Literal["vector", "keyword"]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked, source-preserving result from an index snapshot."""

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

RouteName = Literal["knowledge", "mcp", "direct"]
IntentName = Literal[
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
    """Display-safe description of the single dialogue decision for this turn."""

    action: str
    business_type: str | None
    execution_mode: str
    reason_code: str


@dataclass(frozen=True, slots=True)
class BusinessStateSummary:
    """Display-safe projection of the latest committed business state."""

    business_type: str | None
    phase: str
    waiting_for_input: bool
    missing_slots: tuple[str, ...] = ()
    clarification_question: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    allowed_actions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RouteSummary:
    """Stable public routing metadata for one completed turn."""

    route: RouteName
    intent: IntentName
    decision_source: DecisionSource
    reason_code: str


@dataclass(frozen=True, slots=True)
class AnswerSource:
    """A source that was actually supplied to and cited by the answer."""

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
    """Keep model intent separate from validation and the answer's business status.

    The bounded final-text excerpt is private persistence data, never model
    history, public JSON, or trace progress. Model reasoning fields are excluded.
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
    """Private diagnostics for one leaf; public plan data uses an allow-list."""

    step_id: str
    diagnostics: AnswerDiagnostics


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """Structured answer with an optional private, persisted assessment snapshot.

    HTTP responses must convert ``assessment_state`` with ``public_assessment``;
    the snapshot contains submitted answers and is never a public response body.
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
    # Public execution summary only: goals, dependencies, outcomes and counts.
    # Never store model reasoning or raw tool responses in this field.
    plan: dict[str, Any] | None = None
    answer_diagnostics: AnswerDiagnostics | None = None
    step_answer_diagnostics: tuple[StepAnswerDiagnostics, ...] = field(default=(), repr=False)
    dialogue: DialogueSummary | None = None
    business_state: BusinessStateSummary | None = None
