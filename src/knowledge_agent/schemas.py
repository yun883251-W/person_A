"""共享数据结构。

文档片段：片段 ID、原文、完整相对文件名、标题层级、
原文起止行号、内容指纹及内容类型。
检索结果：片段、检索渠道、排名和相似度等信息。
回答结果：回答正文、实际使用的来源、资料不足说明。
检索分数不得作为经过校准的答案正确概率。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
