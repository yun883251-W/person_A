"""共享数据结构。

文档片段：片段 ID、原文、完整相对文件名、标题层级、
原文起止行号、内容指纹及内容类型。
检索结果：片段、检索渠道、排名和相似度等信息。
回答结果：回答正文、实际使用的来源、资料不足说明。
检索分数不得作为经过校准的答案正确概率。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .agent.trace import TurnTrace


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


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """Structured public result of one Agent answer."""

    question: str
    answer: str
    status: AnswerStatus
    sources: tuple[AnswerSource, ...] = ()
    index_version: str | None = None
    model_calls: int = 0
    tool_calls: int = 0
    stop_reason: str | None = None
    trace: TurnTrace | None = None
