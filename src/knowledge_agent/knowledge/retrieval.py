"""知识库检索。

职责：读取 ingest 发布的本地索引，根据问题返回带来源的相关文档片段。
输入：用户问题、索引目录、Embedding 配置及检索参数。
输出：候选片段、来源、检索渠道、排名及必要的诊断信息。

沿用本地 JSON / JSONL 索引，不引入数据库，也不在查询时重新入库。
索引文件读取与完整性校验委托给 index_reader.py。
本模块不生成最终回答，不执行检索到的代码或 Prompt。
相关性闸门先于 RRF/top-k，允许正常返回空结果；模型或索引错误仍抛出异常。
默认余弦阈值是可通过环境变量调整的启发式参数。
"""

from __future__ import annotations

import heapq
import math
import threading
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from ..config import get_ingest_settings, get_retrieval_settings
from .ingest import (
    OllamaEmbeddingError,
    OllamaEmbeddingModel,
    _is_transient_embedding_error,
    check_ollama_embedding_model,
)
from . import index_reader
# 保留原公开类型和片段读取入口，避免调用方迁移时产生类型不一致。
from .index_reader import (
    IndexSnapshot as IndexSnapshot,
    PublishedIndexArtifacts as PublishedIndexArtifacts,
    IndexNotBuiltError as IndexNotBuiltError,
    load_published_chunks as load_published_chunks,
)
from .keywords import extract_keywords
from .relevance import extract_topic_terms, matching_topic_terms, topic_coverage
from knowledge_agent.models.provider import create_embedding_model
from ..schemas import DocumentChunk, RetrievalChannel, RetrievalResult


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """一次检索请求的加载与排序控制参数。"""

    index_dir: Path
    top_k: int
    vector_candidate_count: int
    keyword_candidate_count: int
    rrf_k: int
    request_timeout: float
    min_cosine: float = 0.45
    semantic_min_cosine: float = 0.65
    min_topic_coverage: float = 0.6

    def __post_init__(self) -> None:
        for name, value in (
            ("RETRIEVAL_MIN_COSINE", self.min_cosine),
            ("RETRIEVAL_SEMANTIC_MIN_COSINE", self.semantic_min_cosine),
        ):
            if (
                isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or not -1.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} 必须是 [-1, 1] 内的有限数值")
        if self.semantic_min_cosine < self.min_cosine:
            raise ValueError("RETRIEVAL_SEMANTIC_MIN_COSINE 不能小于 RETRIEVAL_MIN_COSINE")
        coverage = self.min_topic_coverage
        if (
            isinstance(coverage, bool) or not isinstance(coverage, Real)
            or not math.isfinite(coverage) or not 0.0 < coverage <= 1.0
        ):
            raise ValueError("RETRIEVAL_MIN_TOPIC_COVERAGE 必须是 (0, 1] 内的有限数值")


def load_retrieval_config() -> RetrievalConfig:
    """加载检索设置、解析路径并校验排序上限。"""
    settings = get_retrieval_settings()
    index_dir = Path(settings.index_dir).expanduser()
    if not index_dir.is_absolute():
        index_dir = settings.project_root / index_dir
    index_dir = index_dir.resolve()

    positive_values = {
        "RETRIEVAL_TOP_K": settings.top_k,
        "RETRIEVAL_VECTOR_CANDIDATES": settings.vector_candidate_count,
        "RETRIEVAL_KEYWORD_CANDIDATES": settings.keyword_candidate_count,
        "RETRIEVAL_RRF_K": settings.rrf_k,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0")

    if settings.vector_candidate_count < settings.top_k:
        raise ValueError("RETRIEVAL_VECTOR_CANDIDATES 不能小于 RETRIEVAL_TOP_K")
    if settings.keyword_candidate_count < settings.top_k:
        raise ValueError("RETRIEVAL_KEYWORD_CANDIDATES 不能小于 RETRIEVAL_TOP_K")
    if settings.request_timeout <= 0:
        raise ValueError("EMBEDDING_REQUEST_TIMEOUT 必须大于 0")

    return RetrievalConfig(
        index_dir=index_dir,
        top_k=settings.top_k,
        vector_candidate_count=settings.vector_candidate_count,
        keyword_candidate_count=settings.keyword_candidate_count,
        rrf_k=settings.rrf_k,
        request_timeout=settings.request_timeout,
        min_cosine=settings.min_cosine,
        semantic_min_cosine=settings.semantic_min_cosine,
        min_topic_coverage=settings.min_topic_coverage,
    )

def load_published_index_artifacts(
    config: RetrievalConfig | None = None,
) -> PublishedIndexArtifacts:
    """保留检索配置入口，将目录交给索引读取层。"""
    config = config or load_retrieval_config()
    return index_reader.load_published_index_artifacts(config.index_dir)


def load_index_snapshot(config: RetrievalConfig | None = None) -> IndexSnapshot:
    """加载默认或指定检索配置对应的索引快照。"""
    config = config or load_retrieval_config()
    return index_reader.load_index_snapshot(config.index_dir)


def extract_query_keywords(query_text: str) -> tuple[str, ...]:
    """按建库时的规则规范化并去重查询词。"""
    return extract_keywords(query_text)

class EmbeddingCompatibilityError(RuntimeError):
    """当前 Embedding 模型无法查询已存索引时抛出。"""


_TEXT_STRATEGY_PAIRS = {
    (
        "heading_path_then_verbatim_source",
        "retrieval_text-v1",
        "verbatim",
        "adapter-embed_query-v1",
    )
}


def create_compatible_embedding_model(
    snapshot: IndexSnapshot,
    config: RetrievalConfig | None = None,
    *,
    adapter: Any | None = None,
    model_identity_checker: Callable[..., OllamaEmbeddingModel] | None = None,
) -> OllamaEmbeddingModel:
    """创建查询适配器，并校验它与 *snapshot* 完全匹配。"""
    config = config or load_retrieval_config()
    manifest = snapshot.manifest
    if not isinstance(manifest, dict):
        raise EmbeddingCompatibilityError("索引快照缺少 manifest，无法验证 Embedding 模型")

    manifest_embedding = manifest.get("embedding")
    metadata_embedding = snapshot.metadata.get("embedding")
    if not isinstance(manifest_embedding, dict) or not isinstance(metadata_embedding, dict):
        raise EmbeddingCompatibilityError("索引缺少有效的 Embedding 元数据")

    identity_fields = ("provider", "model", "digest", "dimension")
    for field in identity_fields:
        manifest_value = manifest_embedding.get(field)
        metadata_value = metadata_embedding.get(field)
        if manifest_value != metadata_value:
            raise EmbeddingCompatibilityError(
                f"索引 Embedding {field} 在 manifest 与 metadata 中不一致"
            )
    provider, model, digest, dimension = (
        manifest_embedding.get(field) for field in identity_fields
    )
    if not all(isinstance(value, str) and value for value in (provider, model, digest)):
        raise EmbeddingCompatibilityError("索引中的 Embedding 模型身份无效")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise EmbeddingCompatibilityError("索引中的 Embedding 向量维度无效")
    if snapshot.vectors and len(snapshot.vectors[0]) != dimension:
        raise EmbeddingCompatibilityError("索引向量的实际维度与 Embedding 元数据不一致")

    strategies = (
        manifest_embedding.get("document_text"),
        metadata_embedding.get("document_strategy"),
        manifest_embedding.get("query_text"),
        metadata_embedding.get("query_strategy"),
    )
    if strategies not in _TEXT_STRATEGY_PAIRS:
        raise EmbeddingCompatibilityError(
            "索引使用了不兼容的 Embedding 文本处理策略: "
            f"{strategies!r}"
        )

    settings = get_ingest_settings()
    configured_provider = settings.embedding_provider.strip().lower()
    configured_model = settings.embedding_model.strip()
    if configured_provider != provider or configured_model != model:
        raise EmbeddingCompatibilityError(
            "当前 Embedding 配置与索引不兼容: "
            f"索引为 {provider}/{model}，当前为 {configured_provider}/{configured_model}"
        )

    embedding_adapter = adapter
    if embedding_adapter is None:
        embedding_adapter = create_embedding_model(
            provider=configured_provider,
            model=configured_model,
            base_url=settings.embedding_base_url,
            request_timeout=config.request_timeout,
        )
    checker = model_identity_checker or check_ollama_embedding_model
    current = checker(
        config=SimpleNamespace(
            embedding_provider=configured_provider,
            embedding_model=configured_model,
            embedding_base_url=settings.embedding_base_url.strip().rstrip("/"),
            embedding_request_timeout=config.request_timeout,
        ),
        timeout=config.request_timeout,
        adapter=embedding_adapter,
    )
    actual_identity = (current.provider, current.model, current.digest, current.dimension)
    expected_identity = (provider, model, digest, dimension)
    if actual_identity != expected_identity:
        raise EmbeddingCompatibilityError(
            "当前 Ollama Embedding 模型与索引不兼容: "
            f"索引为 {expected_identity!r}，当前为 {actual_identity!r}"
        )
    return current

@dataclass(frozen=True, slots=True)
class PreparedQuery:
    """原始问题、实际 Embedding 输入以及经校验的向量。"""

    original_question: str
    query_text: str
    vector: tuple[float, ...]


def prepare_query(
    question: str,
    snapshot: IndexSnapshot,
    embedding_model: OllamaEmbeddingModel,
    *,
    max_retries: int | None = None,
    retry_delay: float | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> PreparedQuery:
    """使用索引查询策略校验并嵌入一个用户问题。"""
    if not isinstance(question, str):
        raise TypeError("用户问题必须是字符串")
    if not question.strip():
        raise ValueError("用户问题不能为空")
    if not isinstance(snapshot, IndexSnapshot):
        raise TypeError("snapshot 必须是 IndexSnapshot")
    if not isinstance(embedding_model, OllamaEmbeddingModel):
        raise TypeError("embedding_model 必须是 OllamaEmbeddingModel")

    manifest_embedding = (snapshot.manifest or {}).get("embedding")
    metadata_embedding = snapshot.metadata.get("embedding")
    if not isinstance(manifest_embedding, dict) or not isinstance(metadata_embedding, dict):
        raise EmbeddingCompatibilityError("索引缺少有效的 Embedding 文本策略")
    query_strategies = (
        manifest_embedding.get("query_text"),
        metadata_embedding.get("query_strategy"),
    )
    if query_strategies != ("verbatim", "adapter-embed_query-v1"):
        raise EmbeddingCompatibilityError(
            f"索引使用了不兼容的查询文本处理策略: {query_strategies!r}"
        )

    settings = get_ingest_settings()
    retries = settings.embedding_max_retries if max_retries is None else max_retries
    delay = settings.embedding_retry_delay if retry_delay is None else retry_delay
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise ValueError("max_retries 必须是非负整数")
    if (
        not isinstance(delay, Real)
        or isinstance(delay, bool)
        or not math.isfinite(delay)
        or delay < 0
    ):
        raise ValueError("retry_delay 必须是有限的非负数")
    if not callable(sleeper):
        raise TypeError("sleeper 必须可调用")

    query_text = question
    embed_query = getattr(embedding_model.adapter, "embed_query", None)
    if not callable(embed_query):
        raise TypeError("Embedding 适配器缺少可调用的 embed_query")

    for attempt in range(retries + 1):
        try:
            raw_vector = embed_query(query_text)
            break
        except Exception as exc:
            if attempt >= retries or not _is_transient_embedding_error(exc):
                raise OllamaEmbeddingError(f"查询向量化失败: {exc}") from exc
            sleeper(float(delay) * (2**attempt))

    expected_dimension = embedding_model.dimension
    if not isinstance(expected_dimension, int) or isinstance(expected_dimension, bool) or expected_dimension <= 0:
        raise EmbeddingCompatibilityError("Embedding 向量维度必须是正整数")
    if not isinstance(raw_vector, (list, tuple)):
        raise OllamaEmbeddingError("查询向量不是有效序列")
    if len(raw_vector) != expected_dimension:
        raise OllamaEmbeddingError(
            f"查询向量维度错误（期望 {expected_dimension}，实际 {len(raw_vector)}）"
        )
    if not all(
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in raw_vector
    ):
        raise OllamaEmbeddingError("查询向量包含非有限或非数值元素")
    if not any(value != 0 for value in raw_vector):
        raise OllamaEmbeddingError("查询向量为零向量")

    return PreparedQuery(
        original_question=question,
        query_text=query_text,
        vector=tuple(float(value) for value in raw_vector),
    )

@dataclass(frozen=True, slots=True)
class VectorCandidate:
    """一条向量通道候选，位于跨通道融合之前。"""

    chunk_id: str
    score: float
    rank: int


def retrieve_vector_candidates(
    query: PreparedQuery,
    snapshot: IndexSnapshot,
    config: RetrievalConfig,
) -> tuple[VectorCandidate, ...]:
    """从单个索引快照返回余弦相似度最高的向量候选。"""
    if not isinstance(query, PreparedQuery):
        raise TypeError("query 必须是 PreparedQuery")
    if not isinstance(snapshot, IndexSnapshot):
        raise TypeError("snapshot 必须是 IndexSnapshot")
    if not isinstance(config, RetrievalConfig):
        raise TypeError("config 必须是 RetrievalConfig")

    candidate_count = config.vector_candidate_count
    if candidate_count <= 0:
        raise ValueError("vector_candidate_count 必须大于 0")
    if not (
        len(snapshot.vector_chunk_ids)
        == len(snapshot.vectors)
        == len(snapshot.vector_norms)
    ):
        raise ValueError("向量索引的 ID、向量与范数数量不一致")
    if not snapshot.vectors:
        return ()

    query_vector = query.vector
    dimension = len(snapshot.vectors[0])
    if len(query_vector) != dimension:
        raise ValueError(
            f"查询向量维度错误（期望 {dimension}，实际 {len(query_vector)}）"
        )
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    if not math.isfinite(query_norm) or query_norm == 0.0:
        raise ValueError("查询向量必须具有有限的非零范数")

    def scored_candidates() -> Any:
        for chunk_id, vector, vector_norm in zip(
            snapshot.vector_chunk_ids,
            snapshot.vectors,
            snapshot.vector_norms,
            strict=True,
        ):
            score = sum(
                left * right
                for left, right in zip(query_vector, vector, strict=True)
            ) / (query_norm * vector_norm)
            yield chunk_id, score

    limit = min(candidate_count, len(snapshot.vectors))
    ranking_key = lambda item: (-item[1], item[0])
    if limit == len(snapshot.vectors):
        ranked = sorted(scored_candidates(), key=ranking_key)
    else:
        ranked = heapq.nsmallest(limit, scored_candidates(), key=ranking_key)

    return tuple(
        VectorCandidate(chunk_id=chunk_id, score=score, rank=rank)
        for rank, (chunk_id, score) in enumerate(ranked, start=1)
    )

@dataclass(frozen=True, slots=True)
class KeywordCandidate:
    """一条关键词通道候选，位于跨通道融合之前。"""

    chunk_id: str
    score: int
    rank: int
    matched_terms: tuple[str, ...]


def retrieve_keyword_candidates(
    query: PreparedQuery,
    snapshot: IndexSnapshot,
    config: RetrievalConfig,
) -> tuple[KeywordCandidate, ...]:
    """对有意义的主题匹配排序，并复用现有索引做候选查找。

    中文倒排项是单个字符。先求交集发现候选，再要求原文中出现
    连续的原始二元组；单字/疑问词重叠不构成有意义的关键词命中。
    """
    if not isinstance(query, PreparedQuery):
        raise TypeError("query 必须是 PreparedQuery")
    if not isinstance(snapshot, IndexSnapshot):
        raise TypeError("snapshot 必须是 IndexSnapshot")
    if not isinstance(config, RetrievalConfig):
        raise TypeError("config 必须是 RetrievalConfig")

    candidate_count = config.keyword_candidate_count
    if candidate_count <= 0:
        raise ValueError("keyword_candidate_count 必须大于 0")

    matches: dict[str, list[str]] = {}
    postings = snapshot.keyword_postings
    for term in extract_topic_terms(query.original_question):
        index_terms = extract_keywords(term)
        if not index_terms:
            continue
        possible_ids = set(postings.get(index_terms[0], ()))
        for index_term in index_terms[1:]:
            possible_ids.intersection_update(postings.get(index_term, ()))
        for chunk_id in possible_ids:
            chunk = snapshot.chunks[chunk_id]
            if matching_topic_terms((term,), chunk.retrieval_text):
                matches.setdefault(chunk_id, []).append(term)

    if not matches:
        return ()

    ranking_key = lambda item: (-len(item[1]), item[0])
    limit = min(candidate_count, len(matches))
    if limit == len(matches):
        ranked = sorted(matches.items(), key=ranking_key)
    else:
        ranked = heapq.nsmallest(limit, matches.items(), key=ranking_key)

    return tuple(
        KeywordCandidate(
            chunk_id=chunk_id,
            score=len(matched_terms),
            rank=rank,
            matched_terms=tuple(matched_terms),
        )
        for rank, (chunk_id, matched_terms) in enumerate(ranked, start=1)
    )


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    """单次调用的信号，可安全保留，不共享可变请求状态。"""

    reason_code: str
    candidate_count: int
    relevant_candidate_count: int
    topic_term_count: int
    max_cosine: float | None
    max_topic_coverage: float
    min_cosine: float
    semantic_min_cosine: float
    min_topic_coverage: float


def filter_relevant_candidates(
    query: PreparedQuery,
    snapshot: IndexSnapshot,
    vector_candidates: tuple[VectorCandidate, ...],
    keyword_candidates: tuple[KeywordCandidate, ...],
    config: RetrievalConfig,
) -> tuple[tuple[VectorCandidate, ...], tuple[KeywordCandidate, ...], RetrievalDiagnostics]:
    """在融合之前对原始候选做闸门过滤，绝不把 RRF 解释为置信度。

    保留同时满足余弦与主题覆盖的候选、没有词法重叠但余弦较高的改写，
    或关键词通道中命中全部主题提示的精确匹配。这些是相关性启发式，
    并不证明某段文字回答了所有请求细节。回答阶段仍会检查证据充分性。
    全部候选被拒绝属于正常的空检索。
    """
    terms = extract_topic_terms(query.original_question)
    vectors = {candidate.chunk_id: candidate for candidate in vector_candidates}
    keywords = {candidate.chunk_id: candidate for candidate in keyword_candidates}
    candidate_ids = vectors.keys() | keywords.keys()
    accepted: set[str] = set()
    max_coverage = 0.0
    for chunk_id in candidate_ids:
        chunk = snapshot.chunks[chunk_id]
        matched = matching_topic_terms(terms, chunk.retrieval_text)
        coverage = topic_coverage(terms, matched)
        max_coverage = max(max_coverage, coverage)
        vector = vectors.get(chunk_id)
        cosine = vector.score if vector is not None else None
        semantic_support = cosine is not None and cosine >= config.semantic_min_cosine
        combined_support = (
            cosine is not None and cosine >= config.min_cosine
            and coverage >= config.min_topic_coverage
        )
        exact_topic_support = chunk_id in keywords and bool(terms) and coverage == 1.0
        if semantic_support or combined_support or exact_topic_support:
            accepted.add(chunk_id)

    diagnostics = RetrievalDiagnostics(
        reason_code=(
            "retrieval_relevant" if accepted else
            "retrieval_below_relevance" if candidate_ids else "retrieval_no_candidates"
        ),
        candidate_count=len(candidate_ids),
        relevant_candidate_count=len(accepted),
        topic_term_count=len(terms),
        max_cosine=max((candidate.score for candidate in vector_candidates), default=None),
        max_topic_coverage=max_coverage,
        min_cosine=config.min_cosine,
        semantic_min_cosine=config.semantic_min_cosine,
        min_topic_coverage=config.min_topic_coverage,
    )
    return (
        tuple(candidate for candidate in vector_candidates if candidate.chunk_id in accepted),
        tuple(candidate for candidate in keyword_candidates if candidate.chunk_id in accepted),
        diagnostics,
    )

@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """一个跨检索通道去重并按 RRF 排序的片段。"""

    chunk_id: str
    channels: tuple[RetrievalChannel, ...]
    fused_score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_score: float | None = None
    keyword_score: int | None = None


def fuse_candidates(
    vector_candidates: tuple[VectorCandidate, ...],
    keyword_candidates: tuple[KeywordCandidate, ...],
    config: RetrievalConfig,
) -> tuple[FusedCandidate, ...]:
    """用 RRF 融合各排序通道，按片段 ID 去重，并应用 top-k。"""
    if not isinstance(config, RetrievalConfig):
        raise TypeError("config 必须是 RetrievalConfig")
    if config.top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if config.rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")

    # 每个通道对每个片段至多贡献一次。保留最佳条目
    # 也使本函数对外部传入的候选序列同样安全。
    merged: dict[str, list[VectorCandidate | KeywordCandidate | None]] = {}
    for candidate in vector_candidates:
        entry = merged.setdefault(candidate.chunk_id, [None, None])
        previous = entry[0]
        if previous is None or candidate.rank < previous.rank:
            entry[0] = candidate
    for candidate in keyword_candidates:
        entry = merged.setdefault(candidate.chunk_id, [None, None])
        previous = entry[1]
        if previous is None or candidate.rank < previous.rank:
            entry[1] = candidate

    rrf_k = config.rrf_k

    def fused_items() -> Any:
        for chunk_id, (vector, keyword) in merged.items():
            fused_score = 0.0
            channels: tuple[RetrievalChannel, ...]
            if vector is not None:
                fused_score += 1.0 / (rrf_k + vector.rank)
            if keyword is not None:
                fused_score += 1.0 / (rrf_k + keyword.rank)
            if vector is not None and keyword is not None:
                channels = ("vector", "keyword")
            elif vector is not None:
                channels = ("vector",)
            else:
                channels = ("keyword",)

            yield FusedCandidate(
                chunk_id=chunk_id,
                channels=channels,
                fused_score=fused_score,
                vector_rank=vector.rank if vector is not None else None,
                keyword_rank=keyword.rank if keyword is not None else None,
                vector_score=vector.score if vector is not None else None,
                keyword_score=keyword.score if keyword is not None else None,
            )

    limit = min(config.top_k, len(merged))
    ranking_key = lambda candidate: (-candidate.fused_score, candidate.chunk_id)
    if limit == len(merged):
        ranked = sorted(fused_items(), key=ranking_key)
    else:
        ranked = heapq.nsmallest(limit, fused_items(), key=ranking_key)
    return tuple(ranked)

def organize_retrieval_results(
    snapshot: IndexSnapshot,
    candidates: tuple[FusedCandidate, ...],
) -> tuple[RetrievalResult, ...]:
    """将排序候选解析为同一快照中保留原文的片段。"""
    if not isinstance(snapshot, IndexSnapshot):
        raise TypeError("snapshot 必须是 IndexSnapshot")

    chunks = snapshot.chunks
    index_version = snapshot.index_version
    results: list[RetrievalResult] = []
    append_result = results.append
    for candidate in candidates:
        try:
            chunk = chunks[candidate.chunk_id]
        except KeyError as exc:
            raise ValueError(
                f"Retrieval candidate refers to unknown chunk ID: {candidate.chunk_id!r}"
            ) from exc
        append_result(RetrievalResult(
            chunk_id=candidate.chunk_id,
            chunk=chunk,
            channels=candidate.channels,
            fused_score=candidate.fused_score,
            index_version=index_version,
            vector_rank=candidate.vector_rank,
            keyword_rank=candidate.keyword_rank,
            vector_score=candidate.vector_score,
            keyword_score=candidate.keyword_score,
        ))
    return tuple(results)

@dataclass(frozen=True, slots=True)
class ContextReadResult:
    """从单个索引快照组装的、保留原文的上下文。"""

    source_path: str
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    text: str
    chunks: tuple[DocumentChunk, ...]
    complete_section: bool


def _merge_context_text(chunks: tuple[DocumentChunk, ...]) -> str:
    """拼接有序片段，同时移除按行对齐的重叠部分。"""
    if not chunks:
        return ""

    parts = [chunks[0].text]
    previous = chunks[0]
    for chunk in chunks[1:]:
        overlap = previous.end_line - chunk.start_line + 1
        if overlap > 0:
            previous_lines = previous.text.splitlines(keepends=True)
            current_lines = chunk.text.splitlines(keepends=True)
            overlap = min(overlap, len(previous_lines), len(current_lines))
            if previous_lines[-overlap:] == current_lines[:overlap]:
                parts.append("".join(current_lines[overlap:]))
            else:
                # 绝不从外部构建的不一致快照中丢弃文本。
                parts.append(chunk.text)
        else:
            parts.append(chunk.text)
        if chunk.end_line > previous.end_line:
            previous = chunk
    return "".join(parts)


def _starts_heading_section(chunk: DocumentChunk) -> bool:
    """返回片段是否以其标题路径所表示的标题开头。"""
    if not chunk.heading_path:
        return False
    first_line = chunk.text.splitlines()[0].lstrip() if chunk.text else ""
    level = len(first_line) - len(first_line.lstrip("#"))
    if level == 0 or level > 6 or len(first_line) == level:
        return False
    separator = first_line[level]
    if separator not in " \t":
        return False
    title = first_line[level:].strip()
    if " #" in title:
        title = title.rstrip("#").rstrip()
    return title == chunk.heading_path[-1]


def read_context(
    snapshot: IndexSnapshot,
    chunk_id: str,
    *,
    adjacent_before: int = 1,
    adjacent_after: int = 1,
    whole_section: bool = False,
    max_chunks: int | None = None,
    max_chars: int | None = None,
) -> ContextReadResult:
    """从 *snapshot* 读取相邻片段或命中片段所属的相关章节。

    章节是具有相同完整源路径和标题路径的连续片段集合。
    这样可区分重复的同名章节。即使命中片段单独就超出字符预算，
    也始终返回该片段。
    """
    if not isinstance(snapshot, IndexSnapshot):
        raise TypeError("snapshot must be an IndexSnapshot")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("chunk_id must be a non-empty string")
    for name, value in (
        ("adjacent_before", adjacent_before),
        ("adjacent_after", adjacent_after),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if max_chunks is not None and (
        not isinstance(max_chunks, int) or isinstance(max_chunks, bool) or max_chunks <= 0
    ):
        raise ValueError("max_chunks must be a positive integer")
    if max_chars is not None and (
        not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0
    ):
        raise ValueError("max_chars must be a positive integer")

    try:
        hit = snapshot.chunks[chunk_id]
    except KeyError as exc:
        raise ValueError(f"unknown chunk ID: {chunk_id!r}") from exc

    file_chunks = sorted(
        (chunk for chunk in snapshot.chunks.values() if chunk.source_path == hit.source_path),
        key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.chunk_id or ""),
    )
    hit_index = next(
        (index for index, chunk in enumerate(file_chunks) if chunk.chunk_id == chunk_id),
        None,
    )
    if hit_index is None:
        raise ValueError(f"chunk {chunk_id!r} is inconsistent with the index snapshot")

    if whole_section:
        start = hit_index
        while (
            start
            and file_chunks[start - 1].heading_path == hit.heading_path
            and not _starts_heading_section(file_chunks[start])
        ):
            start -= 1
        stop = hit_index + 1
        while (
            stop < len(file_chunks)
            and file_chunks[stop].heading_path == hit.heading_path
            and not _starts_heading_section(file_chunks[stop])
        ):
            stop += 1
    else:
        start = max(0, hit_index - adjacent_before)
        stop = min(len(file_chunks), hit_index + adjacent_after + 1)
    candidates = file_chunks[start:stop]

    # 最近优先的选择方式，能在紧张预算下保留命中片段及其最近的上下文。
    ordered_indexes = sorted(
        range(len(candidates)),
        key=lambda index: (abs(start + index - hit_index), start + index),
    )
    selected: list[DocumentChunk] = []
    for index in ordered_indexes:
        candidate = candidates[index]
        if max_chunks is not None and len(selected) >= max_chunks:
            break
        trial = tuple(sorted(
            (*selected, candidate),
            key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.chunk_id or ""),
        ))
        if max_chars is not None and selected and len(_merge_context_text(trial)) > max_chars:
            continue
        selected.append(candidate)

    chunks = tuple(sorted(
        selected,
        key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.chunk_id or ""),
    ))
    return ContextReadResult(
        source_path=hit.source_path,
        heading_path=hit.heading_path,
        start_line=min(chunk.start_line for chunk in chunks),
        end_line=max(chunk.end_line for chunk in chunks),
        text=_merge_context_text(chunks),
        chunks=chunks,
        complete_section=whole_section and len(chunks) == len(candidates),
    )


def read_adjacent_chunks(
    snapshot: IndexSnapshot,
    chunk_id: str,
    *,
    before: int = 1,
    after: int = 1,
    max_chunks: int | None = None,
    max_chars: int | None = None,
) -> ContextReadResult:
    """读取某个已索引片段周围有界、按源顺序排列的窗口。"""
    return read_context(
        snapshot,
        chunk_id,
        adjacent_before=before,
        adjacent_after=after,
        max_chunks=max_chunks,
        max_chars=max_chars,
    )


def read_related_section(
    snapshot: IndexSnapshot,
    chunk_id: str,
    *,
    max_chunks: int | None = None,
    max_chars: int | None = None,
) -> ContextReadResult:
    """读取命中片段精确所属章节内可用的已索引片段。"""
    return read_context(
        snapshot,
        chunk_id,
        whole_section=True,
        max_chunks=max_chunks,
        max_chars=max_chars,
    )

class IndexRefreshError(RuntimeError):
    """刷新失败且仍保留旧快照处于活动状态时抛出。"""

    def __init__(self, retained_index_version: str | None, cause: BaseException) -> None:
        self.retained_index_version = retained_index_version
        retained = retained_index_version or "none"
        super().__init__(
            f"Index refresh failed; retained index version: {retained}: {cause}"
        )


@dataclass(frozen=True, slots=True)
class _RetrievalRuntime:
    snapshot: IndexSnapshot
    embedding_model: OllamaEmbeddingModel


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """从同一索引/请求捕获的结果与相关性诊断。"""

    results: tuple[RetrievalResult, ...]
    diagnostics: RetrievalDiagnostics
    index_version: str


class KnowledgeRetriever:
    """线程安全、可复用的索引检索与上下文读取入口。"""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        *,
        snapshot_loader: Callable[[RetrievalConfig], IndexSnapshot] = load_index_snapshot,
        embedding_model_factory: Callable[
            [IndexSnapshot, RetrievalConfig], OllamaEmbeddingModel
        ] = create_compatible_embedding_model,
    ) -> None:
        self.config = config or load_retrieval_config()
        self._snapshot_loader = snapshot_loader
        self._embedding_model_factory = embedding_model_factory
        self._runtime: _RetrievalRuntime | None = None
        self._state_lock = threading.Lock()
        self._refresh_lock = threading.Lock()

    def _build_runtime(self) -> _RetrievalRuntime:
        snapshot = self._snapshot_loader(self.config)
        embedding_model = self._embedding_model_factory(snapshot, self.config)
        return _RetrievalRuntime(snapshot, embedding_model)

    def load(self) -> IndexSnapshot:
        """只加载一次活动索引；后续调用复用同一快照。"""
        runtime = self._runtime
        if runtime is not None:
            return runtime.snapshot

        with self._refresh_lock:
            runtime = self._runtime
            if runtime is None:
                runtime = self._build_runtime()
                with self._state_lock:
                    self._runtime = runtime
            return runtime.snapshot

    def _current_runtime(self) -> _RetrievalRuntime:
        self.load()
        runtime = self._runtime
        if runtime is None:  # 防御性检查：load 要么发布运行时，要么抛出异常。
            raise RuntimeError("Retrieval runtime was not loaded")
        return runtime

    @property
    def index_version(self) -> str:
        """返回当前提供查询服务的快照版本。"""
        return self._current_runtime().snapshot.index_version

    def retrieve(self, question: str) -> tuple[RetrievalResult, ...]:
        """返回相关的、保留原文的结果；空元组是合法结果。"""
        return self.retrieve_with_diagnostics(question).results

    def retrieve_with_diagnostics(self, question: str) -> RetrievalOutcome:
        """针对同一份捕获的快照运行两个通道并进行相关性过滤。

        不保存可变的 ``last_stats``，因此并发调用方不会读到彼此的诊断。
        技术性故障仍会作为错误向上传播。
        """
        runtime = self._current_runtime()
        snapshot = runtime.snapshot
        query = prepare_query(question, snapshot, runtime.embedding_model)
        vectors = retrieve_vector_candidates(query, snapshot, self.config)
        keywords = retrieve_keyword_candidates(query, snapshot, self.config)
        vectors, keywords, diagnostics = filter_relevant_candidates(
            query, snapshot, vectors, keywords, self.config,
        )
        fused = fuse_candidates(vectors, keywords, self.config)
        return RetrievalOutcome(
            organize_retrieval_results(snapshot, fused), diagnostics, snapshot.index_version,
        )

    def read_context(self, chunk_id: str, **kwargs: Any) -> ContextReadResult:
        """从本次调用开始时活动的快照读取上下文。"""
        snapshot = self._current_runtime().snapshot
        return read_context(snapshot, chunk_id, **kwargs)

    def read_adjacent_chunks(self, chunk_id: str, **kwargs: Any) -> ContextReadResult:
        """从本次调用开始时活动的快照读取相邻片段。"""
        snapshot = self._current_runtime().snapshot
        return read_adjacent_chunks(snapshot, chunk_id, **kwargs)

    def read_related_section(self, chunk_id: str, **kwargs: Any) -> ContextReadResult:
        """从本次调用开始时活动的快照读取一个章节。"""
        snapshot = self._current_runtime().snapshot
        return read_related_section(snapshot, chunk_id, **kwargs)

    def refresh(self, _published: Any = None) -> IndexSnapshot:
        """原子地发布新加载的运行时；失败时保留并报告旧运行时。

        可选参数使本方法可直接用作 ingest 的 ``refresh_callback``。
        """
        with self._refresh_lock:
            previous = self._runtime
            try:
                replacement = self._build_runtime()
            except Exception as exc:
                version = previous.snapshot.index_version if previous is not None else None
                raise IndexRefreshError(version, exc) from exc
            with self._state_lock:
                self._runtime = replacement
            return replacement.snapshot


RetrievalService = KnowledgeRetriever
