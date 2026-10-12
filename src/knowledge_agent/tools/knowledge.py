"""知识库工具封装模块（待开发）。

模块功能：集中定义 search_knowledge 与 read_knowledge_section 两个已有工具。
输入：KnowledgeRetriever，以及当前问答轮所需的检索和来源上下文。
输出：供注册器使用的工具定义与执行函数，以及可追溯的检索结果。

开发以迁移 agent/runner.py 中现有实现为主，不重新实现向量化、检索或索引构建。
当前文件只有 TODO，原有工具仍在 agent/runner.py 中正常运行。
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from ..knowledge.retrieval import IndexNotBuiltError
from ..schemas import DocumentChunk, RetrievalResult
from .registry import InMemoryToolRegistry, ToolRegistration


# 第一阶段：迁移现有工具。

RetrievalStatus = Literal["not_started", "empty", "results_found", "failed"]
KnowledgeToolResult = dict[str, Any]


class SectionReadResult(Protocol):
    """Source-preserving section data required by the section reader."""

    source_path: str
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    text: str
    chunks: tuple[DocumentChunk, ...]
    complete_section: bool


class KnowledgeRetrieverPort(Protocol):
    """Retrieval operations consumed by the two knowledge tools."""

    @property
    def index_version(self) -> str: ...

    def retrieve(self, question: str) -> Sequence[RetrievalResult]: ...

    def read_related_section(
        self,
        chunk_id: str,
        *,
        max_chunks: int,
        max_chars: int,
    ) -> SectionReadResult: ...


class KnowledgeTurnState(Protocol):
    """The minimal mutable per-turn state shared by the knowledge tools."""

    evidence: MutableMapping[str, RetrievalResult]
    retrieval_status: RetrievalStatus
    retrieval_error_code: str | None


@dataclass(frozen=True, slots=True)
class KnowledgeToolDependencies:
    """Explicit dependencies passed to executors without importing the Agent."""

    retriever: KnowledgeRetrieverPort
    state: KnowledgeTurnState


class SearchKnowledgeExecutor(Protocol):
    """Input/output contract for ``search_knowledge``."""

    def __call__(self, query: str) -> KnowledgeToolResult: ...


class ReadKnowledgeSectionExecutor(Protocol):
    """Input/output contract for ``read_knowledge_section``."""

    def __call__(self, chunk_id: str) -> KnowledgeToolResult: ...


class SearchKnowledgeInput(BaseModel):
    """Arguments accepted by the knowledge search tool."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(
        min_length=1,
        description="A non-empty question or search phrase to find relevant knowledge chunks.",
    )


class ReadKnowledgeSectionInput(BaseModel):
    """Arguments accepted by the section-reading tool."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk_id: str = Field(
        min_length=1,
        description=(
            "The ID of a chunk returned and registered by search_knowledge during "
            "the current turn."
        ),
    )


SEARCH_KNOWLEDGE_DESCRIPTION = (
    "Search the read-only knowledge base for chunks relevant to a question. "
    "Use this before requesting more context for a result."
)

READ_KNOWLEDGE_SECTION_DESCRIPTION = (
    "Read the surrounding section for a chunk already returned by search_knowledge "
    "in this turn. Do not use arbitrary chunk IDs, file paths, or commands."
)

def create_search_executor(
    retriever: KnowledgeRetrieverPort,
    state: KnowledgeTurnState,
) -> SearchKnowledgeExecutor:
    """Wrap hybrid retrieval as a structured, turn-scoped tool executor."""

    def search_knowledge(query: str) -> KnowledgeToolResult:
        try:
            results = retriever.retrieve(query)
        except IndexNotBuiltError:
            state.retrieval_status = "failed"
            state.retrieval_error_code = "index_not_built"
            return {
                "status": "error",
                "query": query,
                "results": [],
                "error": {
                    "code": "index_not_built",
                    "message": "The knowledge index is not available.",
                },
            }
        except Exception as exc:
            state.retrieval_status = "failed"
            state.retrieval_error_code = "retrieval_failed"
            return {
                "status": "error",
                "query": query,
                "results": [],
                "error": {
                    "code": "retrieval_failed",
                    "type": type(exc).__name__,
                    "message": "Knowledge retrieval failed.",
                },
            }

        if not results:
            state.retrieval_status = "empty"
            return {
                "status": "empty",
                "query": query,
                "index_version": retriever.index_version,
                "results": [],
            }

        state.retrieval_status = "results_found"
        serialized_results: list[dict[str, Any]] = []
        for rank, result in enumerate(results, start=1):
            state.evidence[result.chunk_id] = result
            chunk = result.chunk
            serialized_results.append(
                {
                    "chunk_id": result.chunk_id,
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                    "heading_path": list(chunk.heading_path),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content_hash": chunk.content_hash,
                    "index_version": result.index_version,
                    "rank": rank,
                    "channels": list(result.channels),
                    "fused_score": result.fused_score,
                    "vector_rank": result.vector_rank,
                    "keyword_rank": result.keyword_rank,
                    "vector_score": result.vector_score,
                    "keyword_score": result.keyword_score,
                }
            )

        return {
            "status": "success",
            "query": query,
            "index_version": results[0].index_version,
            "results": serialized_results,
        }

    return search_knowledge


def create_section_executor(
    retriever: KnowledgeRetrieverPort,
    state: KnowledgeTurnState,
    *,
    max_chunks: int = 8,
    max_chars: int = 8_000,
) -> ReadKnowledgeSectionExecutor:
    """Wrap bounded section reads with turn-scoped ID and version checks."""

    def read_knowledge_section(chunk_id: str) -> KnowledgeToolResult:
        anchor = state.evidence.get(chunk_id)
        if anchor is None:
            return {
                "status": "error",
                "chunk_id": chunk_id,
                "error": {
                    "code": "unknown_turn_chunk",
                    "message": "The chunk was not returned during the current turn.",
                },
            }

        expected_version = anchor.index_version
        evidence_versions = {result.index_version for result in state.evidence.values()}
        if evidence_versions != {expected_version}:
            return {
                "status": "error",
                "chunk_id": chunk_id,
                "index_version": expected_version,
                "error": {
                    "code": "mixed_evidence_versions",
                    "message": "Current-turn evidence contains more than one index version.",
                },
            }

        try:
            if retriever.index_version != expected_version:
                return {
                    "status": "error",
                    "chunk_id": chunk_id,
                    "index_version": expected_version,
                    "error": {
                        "code": "index_version_changed",
                        "message": "The index changed after this chunk was retrieved.",
                    },
                }
            section = retriever.read_related_section(
                chunk_id,
                max_chunks=max_chunks,
                max_chars=max_chars,
            )
            if retriever.index_version != expected_version:
                return {
                    "status": "error",
                    "chunk_id": chunk_id,
                    "index_version": expected_version,
                    "error": {
                        "code": "index_version_changed",
                        "message": "The index changed while the section was being read.",
                    },
                }
        except Exception as exc:
            return {
                "status": "error",
                "chunk_id": chunk_id,
                "index_version": expected_version,
                "error": {
                    "code": "section_read_failed",
                    "type": type(exc).__name__,
                    "message": "The related section could not be read.",
                },
            }

        serialized_chunks: list[dict[str, Any]] = []
        for chunk in section.chunks:
            if chunk.chunk_id is None:
                continue
            if chunk.chunk_id not in state.evidence:
                state.evidence[chunk.chunk_id] = RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    chunk=chunk,
                    channels=anchor.channels,
                    fused_score=anchor.fused_score,
                    index_version=expected_version,
                    vector_rank=anchor.vector_rank,
                    keyword_rank=anchor.keyword_rank,
                    vector_score=anchor.vector_score,
                    keyword_score=anchor.keyword_score,
                )
            serialized_chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                    "heading_path": list(chunk.heading_path),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content_hash": chunk.content_hash,
                    "index_version": expected_version,
                }
            )

        complete = section.complete_section
        return {
            "status": "success" if complete else "limited",
            "chunk_id": chunk_id,
            "index_version": expected_version,
            "source_path": section.source_path,
            "heading_path": list(section.heading_path),
            "start_line": section.start_line,
            "end_line": section.end_line,
            "text": section.text,
            "chunks": serialized_chunks,
            "complete_section": complete,
            "scope": (
                "complete_section"
                if complete
                else "partial_section_limited_by_chunk_or_text_budget"
            ),
            "limits": {"max_chunks": max_chunks, "max_chars": max_chars},
        }

    return read_knowledge_section

def create_knowledge_tool_registry(
    *,
    max_section_chunks: int = 8,
    max_section_chars: int = 8_000,
) -> InMemoryToolRegistry[KnowledgeToolDependencies]:
    """Create the two knowledge-tool registrations without doing any I/O."""

    def search_definition(query: str) -> KnowledgeToolResult:
        raise RuntimeError("knowledge tools must be executed through the registry")

    def section_definition(chunk_id: str) -> KnowledgeToolResult:
        raise RuntimeError("knowledge tools must be executed through the registry")

    search_tool = StructuredTool.from_function(
        func=search_definition,
        name="search_knowledge",
        description=SEARCH_KNOWLEDGE_DESCRIPTION,
        args_schema=SearchKnowledgeInput,
    )
    section_tool = StructuredTool.from_function(
        func=section_definition,
        name="read_knowledge_section",
        description=READ_KNOWLEDGE_SECTION_DESCRIPTION,
        args_schema=ReadKnowledgeSectionInput,
    )

    def execute_search(
        context: KnowledgeToolDependencies,
        /,
        **arguments: Any,
    ) -> KnowledgeToolResult:
        executor = create_search_executor(context.retriever, context.state)
        return executor(**arguments)

    def execute_section(
        context: KnowledgeToolDependencies,
        /,
        **arguments: Any,
    ) -> KnowledgeToolResult:
        executor = create_section_executor(
            context.retriever,
            context.state,
            max_chunks=max_section_chunks,
            max_chars=max_section_chars,
        )
        return executor(**arguments)

    return InMemoryToolRegistry(
        (
            ToolRegistration(
                name="search_knowledge",
                definition=search_tool,
                executor=execute_search,
            ),
            ToolRegistration(
                name="read_knowledge_section",
                definition=section_tool,
                executor=execute_section,
            ),
        )
    )

# 第二阶段：开发完成后统一验证，本阶段不执行。

# TODO 07：验证迁移后的搜索、小节补查和来源一致性。
# 覆盖无命中、无索引、非法 chunk_id、非本轮片段和资料长度限制。
# 确认工具调用次数不重复累加，最终引用仍由真实索引元数据生成。
