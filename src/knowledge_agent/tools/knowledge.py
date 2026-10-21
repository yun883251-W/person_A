"""知识库搜索和小节读取工具。

模块功能：集中定义 search_knowledge 与 read_knowledge_section 两个已有工具。
输入：KnowledgeRetriever，以及当前问答轮所需的检索和来源上下文。
输出：供注册器使用的工具定义与执行函数，以及可追溯的检索结果。

复用已有向量化、检索和索引服务，不重新实现入库流程。
agent/preparation.py 负责当前回合的工具组装与上下文准备。
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from ..knowledge.index_reader import IndexNotBuiltError
from ..schemas import DocumentChunk, RetrievalResult
from .registry import InMemoryToolRegistry, ToolRegistration


# 第一阶段：迁移现有工具。

RetrievalStatus = Literal["not_started", "empty", "results_found", "failed"]
KnowledgeToolResult = dict[str, Any]


class SectionReadResult(Protocol):
    """小节读取器所需的、保留来源信息的片段数据。"""

    source_path: str
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    text: str
    chunks: tuple[DocumentChunk, ...]
    complete_section: bool


class KnowledgeRetrieverPort(Protocol):
    """两个知识工具共同使用的检索操作。"""

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
    """知识工具共享的最小可变轮次内状态。"""

    evidence: MutableMapping[str, RetrievalResult]
    retrieval_status: RetrievalStatus
    retrieval_error_code: str | None


@dataclass(frozen=True, slots=True)
class KnowledgeToolDependencies:
    """显式传入执行器的依赖，避免导入 Agent。"""

    retriever: KnowledgeRetrieverPort
    state: KnowledgeTurnState


class SearchKnowledgeExecutor(Protocol):
    """``search_knowledge`` 的输入/输出契约。"""

    def __call__(self, query: str) -> KnowledgeToolResult: ...


class ReadKnowledgeSectionExecutor(Protocol):
    """``read_knowledge_section`` 的输入/输出契约。"""

    def __call__(self, chunk_id: str) -> KnowledgeToolResult: ...


class SearchKnowledgeInput(BaseModel):
    """知识搜索工具接受的参数。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(
        min_length=1,
        description="A non-empty question or search phrase to find relevant knowledge chunks.",
    )


class ReadKnowledgeSectionInput(BaseModel):
    """小节读取工具接受的参数。"""

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
    """将混合检索包装为结构化、限定于当前轮次的工具执行器。"""

    def search_knowledge(query: str) -> KnowledgeToolResult:
        try:
            detailed_retrieve = getattr(retriever, "retrieve_with_diagnostics", None)
            relevance: dict[str, Any] = {}
            if callable(detailed_retrieve):
                outcome = detailed_retrieve(query)
                results = outcome.results
                relevance = asdict(outcome.diagnostics)
                index_version = outcome.index_version
            else:
                results = retriever.retrieve(query)
                index_version = results[0].index_version if results else retriever.index_version
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
            state.retrieval_error_code = None
            return {
                "status": "empty",
                "query": query,
                "index_version": index_version,
                "reason_code": relevance.get("reason_code", "retrieval_no_candidates"),
                "relevance": relevance,
                "results": [],
            }

        state.retrieval_status = "results_found"
        state.retrieval_error_code = None
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
            "reason_code": relevance.get("reason_code", "retrieval_relevant"),
            "relevance": relevance,
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
    """包装有界小节读取，并执行轮次内 ID 与版本校验。"""

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
    """创建两个知识工具注册项，不执行任何 I/O。"""

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
