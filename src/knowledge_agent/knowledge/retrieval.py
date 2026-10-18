"""知识库检索。

职责：读取 ingest 发布的本地索引，根据问题返回带来源的相关文档片段。
输入：用户问题、索引目录、Embedding 配置及检索参数。
输出：候选片段、来源、检索渠道、排名及必要的诊断信息。

沿用本地 JSON / JSONL 索引，不引入数据库，也不在查询时重新入库。
本模块不生成最终回答，不执行检索到的代码或 Prompt。
相关性闸门先于 RRF/top-k，允许正常返回空结果；模型或索引错误仍抛出异常。
默认余弦阈值属于未校准的启发式参数，开发后的测试与效果评估仍待执行。
"""


# 第一阶段：完成全部检索开发。此阶段不逐项执行测试或真实模型联调。

from __future__ import annotations

import json
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
from .keywords import extract_keywords
from .relevance import extract_topic_terms, matching_topic_terms, topic_coverage
from knowledge_agent.models.provider import create_embedding_model
from ..schemas import DocumentChunk, RetrievalChannel, RetrievalResult


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Validated controls for loading and ranking one retrieval request."""

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
    """Load retrieval settings, resolve paths, and validate ranking limits."""
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

@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    """A complete, single-version set of in-memory retrieval artifacts."""

    index_version: str
    chunks: dict[str, DocumentChunk]
    vector_chunk_ids: tuple[str, ...]
    vectors: tuple[tuple[float, ...], ...]
    vector_norms: tuple[float, ...]
    keyword_postings: dict[str, tuple[str, ...]]
    metadata: dict[str, Any]
    manifest: dict[str, Any] | None = None

@dataclass(frozen=True, slots=True)
class PublishedIndexArtifacts:
    """A validated manifest and the artifact files it publishes."""

    index_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    chunks_path: Path
    vector_index_path: Path
    keyword_index_path: Path
    metadata_path: Path


class IndexNotBuiltError(RuntimeError):
    """Raised when retrieval is requested before an index is published."""


def load_published_index_artifacts(
    config: RetrievalConfig | None = None,
) -> PublishedIndexArtifacts:
    """Read the active manifest and resolve its complete published artifact set."""
    config = config or load_retrieval_config()
    index_dir = config.index_dir.resolve()
    manifest_path = index_dir / "manifest.json"

    try:
        with manifest_path.open("r", encoding="utf-8") as source:
            manifest = json.load(source)
    except FileNotFoundError as exc:
        raise IndexNotBuiltError(
            f"索引尚未建立：缺少 {manifest_path}，请先运行建库"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取索引 manifest: {manifest_path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"索引 manifest 格式无效: {manifest_path}: 顶层必须是 JSON 对象")
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"不支持的索引 manifest schema_version: "
            f"{manifest.get('schema_version')!r}"
        )

    status = manifest.get("status")
    if status == "not_built":
        raise IndexNotBuiltError("索引尚未建立，请先运行建库")
    if status != "ready":
        raise ValueError(f"索引 manifest 状态无效: {status!r}，期望 'ready'")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("索引 manifest 缺少有效的 artifacts 对象")

    resolved: dict[str, Path] = {}
    for key in ("chunks", "vectors", "keywords", "metadata"):
        value = artifacts.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"索引 manifest 缺少有效产物路径: {key}")
        path = (index_dir / value).resolve()
        try:
            path.relative_to(index_dir)
        except ValueError as exc:
            raise ValueError(f"索引产物路径越界: {key}: {value!r}") from exc
        if not path.is_file():
            raise ValueError(f"索引产物缺失或不是文件: {key}: {path}")
        resolved[key] = path

    return PublishedIndexArtifacts(
        index_dir=index_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        chunks_path=resolved["chunks"],
        vector_index_path=resolved["vectors"],
        keyword_index_path=resolved["keywords"],
        metadata_path=resolved["metadata"],
    )

def load_published_chunks(artifacts: PublishedIndexArtifacts) -> dict[str, DocumentChunk]:
    """Load and validate the source-preserving chunks in a published index."""
    expected_count = artifacts.manifest.get("chunk_count")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < 0
    ):
        raise ValueError("Index manifest has an invalid chunk_count")

    chunks: dict[str, DocumentChunk] = {}
    try:
        with artifacts.chunks_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")

                heading_path = record.get("heading_path")
                if not isinstance(heading_path, (list, tuple)) or not all(
                    isinstance(heading, str) for heading in heading_path
                ):
                    raise ValueError("invalid heading_path")

                chunk_id = record.get("chunk_id")
                start_line = record.get("start_line")
                end_line = record.get("end_line")
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    raise ValueError("chunk_id must be a non-empty string")
                if chunk_id in chunks:
                    raise ValueError(f"duplicate chunk_id: {chunk_id!r}")
                if (
                    not isinstance(start_line, int)
                    or isinstance(start_line, bool)
                    or not isinstance(end_line, int)
                    or isinstance(end_line, bool)
                    or start_line < 1
                    or end_line < start_line
                ):
                    raise ValueError("invalid source line range")

                text = record.get("text")
                source_path = record.get("source_path")
                content_hash = record.get("content_hash")
                content_type = record.get("content_type", "markdown")
                if not isinstance(text, str) or not text:
                    raise ValueError("text must be a non-empty string")
                if not isinstance(source_path, str) or not source_path:
                    raise ValueError("source_path must be a non-empty string")
                if not isinstance(content_hash, str) or not content_hash:
                    raise ValueError("content_hash must be a non-empty string")
                if content_type != "markdown":
                    raise ValueError(f"unsupported content_type: {content_type!r}")

                chunks[chunk_id] = DocumentChunk(
                    text=text,
                    source_path=source_path,
                    heading_path=tuple(heading_path),
                    start_line=start_line,
                    end_line=end_line,
                    content_hash=content_hash,
                    content_type=content_type,
                    chunk_id=chunk_id,
                )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        location = f"{artifacts.chunks_path}:{locals().get('line_number', 0)}"
        raise ValueError(f"Unable to load published chunks at {location}: {exc}") from exc

    if len(chunks) != expected_count:
        raise ValueError(
            f"Published chunk count mismatch: manifest={expected_count}, actual={len(chunks)}"
        )
    return chunks


def load_index_snapshot(config: RetrievalConfig | None = None) -> IndexSnapshot:
    """Load and validate one complete, consistently published index snapshot."""
    artifacts = load_published_index_artifacts(config)
    chunks = load_published_chunks(artifacts)

    def load_json(path: Path, label: str) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as source:
                value = json.load(source)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to load published {label} at {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Published {label} must be a JSON object: {path}")
        return value

    vector_index = load_json(artifacts.vector_index_path, "vector index")
    if vector_index.get("format") != "flat-cosine-v1":
        raise ValueError(f"Unsupported vector index format: {vector_index.get('format')!r}")
    dimension = vector_index.get("dimension")
    raw_ids = vector_index.get("chunk_ids")
    raw_vectors = vector_index.get("vectors")
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension <= 0
    ):
        raise ValueError("Vector index dimension must be a positive integer")
    if not isinstance(raw_ids, list) or not isinstance(raw_vectors, list):
        raise ValueError("Vector index chunk_ids and vectors must be arrays")
    if len(raw_ids) != len(raw_vectors):
        raise ValueError("Vector index has different chunk ID and vector counts")

    vector_ids: list[str] = []
    vectors: list[tuple[float, ...]] = []
    vector_norms: list[float] = []
    seen_vector_ids: set[str] = set()
    for position, (chunk_id, raw_vector) in enumerate(zip(raw_ids, raw_vectors, strict=True)):
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id not in chunks:
            raise ValueError(f"Vector index has invalid chunk ID at position {position}: {chunk_id!r}")
        if chunk_id in seen_vector_ids:
            raise ValueError(f"Vector index has duplicate chunk ID: {chunk_id!r}")
        if not isinstance(raw_vector, list) or len(raw_vector) != dimension:
            raise ValueError(f"Vector for {chunk_id!r} does not have dimension {dimension}")

        vector: list[float] = []
        squared_norm = 0.0
        for raw_value in raw_vector:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"Vector for {chunk_id!r} contains a non-numeric value")
            try:
                value = float(raw_value)
            except OverflowError as exc:
                raise ValueError(
                    f"Vector for {chunk_id!r} contains a non-finite value"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"Vector for {chunk_id!r} contains a non-finite value")
            vector.append(value)
            squared_norm += value * value
        norm = math.sqrt(squared_norm)
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError(f"Vector for {chunk_id!r} must have a finite, non-zero norm")

        seen_vector_ids.add(chunk_id)
        vector_ids.append(chunk_id)
        vectors.append(tuple(vector))
        vector_norms.append(norm)

    if len(vector_ids) != len(chunks):
        raise ValueError(
            f"Vector index chunk count mismatch: chunks={len(chunks)}, vectors={len(vector_ids)}"
        )

    keyword_index = load_json(artifacts.keyword_index_path, "keyword index")
    if keyword_index.get("format") != "inverted-keyword-v1":
        raise ValueError(f"Unsupported keyword index format: {keyword_index.get('format')!r}")
    keyword_ids = keyword_index.get("chunk_ids")
    raw_postings = keyword_index.get("postings")
    if not isinstance(keyword_ids, list) or not isinstance(raw_postings, dict):
        raise ValueError("Keyword index chunk_ids must be an array and postings must be an object")
    if any(not isinstance(chunk_id, str) for chunk_id in keyword_ids):
        raise ValueError("Keyword index contains a non-string chunk ID")
    if (
        len(keyword_ids) != len(chunks)
        or len(set(keyword_ids)) != len(keyword_ids)
        or any(chunk_id not in chunks for chunk_id in keyword_ids)
    ):
        raise ValueError("Keyword index chunk IDs do not match the published chunks")

    postings: dict[str, tuple[str, ...]] = {}
    for term, raw_chunk_ids in raw_postings.items():
        if not isinstance(term, str):
            raise ValueError("Keyword posting terms must be strings")
        if not term.strip():
            continue
        if not isinstance(raw_chunk_ids, list):
            raise ValueError("Keyword postings must map terms to chunk ID arrays")
        if any(not isinstance(chunk_id, str) or chunk_id not in chunks for chunk_id in raw_chunk_ids):
            raise ValueError(f"Keyword posting {term!r} refers to an unknown chunk")
        postings[term] = tuple(raw_chunk_ids)

    metadata = load_json(artifacts.metadata_path, "index metadata")
    if metadata.get("format") != "knowledge-index-v1":
        raise ValueError(f"Unsupported index metadata format: {metadata.get('format')!r}")
    if metadata.get("chunk_count") != len(chunks):
        raise ValueError("Index metadata chunk count does not match the published chunks")
    sources = metadata.get("sources")
    if not isinstance(sources, dict) or any(
        not isinstance(chunk_id, str) or chunk_id not in chunks for chunk_id in sources
    ):
        raise ValueError("Index metadata sources refer to unknown published chunks")

    index_version = artifacts.manifest.get("built_at")
    if not isinstance(index_version, str) or not index_version:
        raise ValueError("Index manifest has no valid built_at version")

    current_manifest = load_json(artifacts.manifest_path, "manifest")
    if current_manifest != artifacts.manifest:
        raise RuntimeError("Published index changed while its snapshot was being loaded")

    return IndexSnapshot(
        index_version=index_version,
        chunks=chunks,
        vector_chunk_ids=tuple(vector_ids),
        vectors=tuple(vectors),
        vector_norms=tuple(vector_norms),
        keyword_postings=postings,
        metadata=metadata,
        manifest=artifacts.manifest,
    )

def extract_query_keywords(query_text: str) -> tuple[str, ...]:
    """Normalize and deduplicate query terms with the indexing-time rules."""
    return extract_keywords(query_text)

class EmbeddingCompatibilityError(RuntimeError):
    """Raised when the active embedding model cannot query a stored index."""


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
    """Create the query adapter and verify it exactly matches *snapshot*."""
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
    """The original question, actual embedding input, and validated vector."""

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
    """Validate and embed one user question using the index query strategy."""
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
    """One vector-channel candidate, before cross-channel fusion."""

    chunk_id: str
    score: float
    rank: int


def retrieve_vector_candidates(
    query: PreparedQuery,
    snapshot: IndexSnapshot,
    config: RetrievalConfig,
) -> tuple[VectorCandidate, ...]:
    """Return the highest-cosine vector candidates from one index snapshot."""
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
    """One keyword-channel candidate, before cross-channel fusion."""

    chunk_id: str
    score: int
    rank: int
    matched_terms: tuple[str, ...]


def retrieve_keyword_candidates(
    query: PreparedQuery,
    snapshot: IndexSnapshot,
    config: RetrievalConfig,
) -> tuple[KeywordCandidate, ...]:
    """Rank meaningful topic matches, using the existing index for candidate lookup.

    Chinese postings contain individual characters. Intersect them to discover
    candidates, then require the original consecutive pair in the source text;
    single-character/question-word overlap is not a meaningful keyword hit.
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
    """Per-call signals, safe to retain without sharing mutable request state."""

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
    """Gate raw candidates before fusion, never interpreting RRF as confidence.

    Keep candidates supported by both cosine and topic coverage, high-cosine
    paraphrases without lexical overlap, or an exact match of all topic hints
    from the keyword channel. These are relevance heuristics, not proof that a
    passage answers every requested detail. The answering stage still checks
    evidence sufficiency. All-rejected candidates are a normal empty retrieval.
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
    """One chunk deduplicated across retrieval channels and ranked by RRF."""

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
    """Fuse ranked channels with RRF, deduplicate by chunk ID, and apply top-k."""
    if not isinstance(config, RetrievalConfig):
        raise TypeError("config 必须是 RetrievalConfig")
    if config.top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if config.rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")

    # Each channel contributes at most once per chunk.  Keeping the best entry
    # also makes this safe for externally supplied candidate sequences.
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
    """Resolve ranked candidates to source-preserving chunks in one snapshot."""
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
    """Source-preserving context assembled from one index snapshot."""

    source_path: str
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    text: str
    chunks: tuple[DocumentChunk, ...]
    complete_section: bool


def _merge_context_text(chunks: tuple[DocumentChunk, ...]) -> str:
    """Join ordered chunks while removing their line-aligned overlap."""
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
                # Never discard text from an inconsistent externally-built snapshot.
                parts.append(chunk.text)
        else:
            parts.append(chunk.text)
        if chunk.end_line > previous.end_line:
            previous = chunk
    return "".join(parts)


def _starts_heading_section(chunk: DocumentChunk) -> bool:
    """Return whether a chunk starts with the heading represented by its path."""
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
    """Read adjacent chunks or the hit's related section from *snapshot*.

    A section is the contiguous run of chunks with the same full source path
    and heading path. This keeps repeated same-named sections distinct. The hit
    is always returned, even when it alone exceeds the character budget.
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

    # Nearest-first selection keeps the hit and its closest context under tight budgets.
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
    """Read a bounded source-order window around one indexed chunk."""
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
    """Read available indexed chunks belonging to the hit's exact section."""
    return read_context(
        snapshot,
        chunk_id,
        whole_section=True,
        max_chunks=max_chunks,
        max_chars=max_chars,
    )

class IndexRefreshError(RuntimeError):
    """Raised when refresh fails while an earlier snapshot remains active."""

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
    """Results and relevance diagnostics captured from the same index/request."""

    results: tuple[RetrievalResult, ...]
    diagnostics: RetrievalDiagnostics
    index_version: str


class KnowledgeRetriever:
    """Thread-safe, reusable entry point for index retrieval and context reads."""

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
        """Load the active index once; subsequent calls reuse the same snapshot."""
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
        if runtime is None:  # Defensive: load either publishes a runtime or raises.
            raise RuntimeError("Retrieval runtime was not loaded")
        return runtime

    @property
    def index_version(self) -> str:
        """Return the version of the snapshot currently serving queries."""
        return self._current_runtime().snapshot.index_version

    def retrieve(self, question: str) -> tuple[RetrievalResult, ...]:
        """Return relevant source-preserving results; an empty tuple is valid."""
        return self.retrieve_with_diagnostics(question).results

    def retrieve_with_diagnostics(self, question: str) -> RetrievalOutcome:
        """Run both channels and relevance filtering against one captured snapshot.

        No mutable ``last_stats`` are stored, so concurrent callers cannot read
        each other's diagnostics. Technical failures still propagate as errors.
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
        """Read context from the snapshot active when this call begins."""
        snapshot = self._current_runtime().snapshot
        return read_context(snapshot, chunk_id, **kwargs)

    def read_adjacent_chunks(self, chunk_id: str, **kwargs: Any) -> ContextReadResult:
        """Read adjacent chunks from the snapshot active when this call begins."""
        snapshot = self._current_runtime().snapshot
        return read_adjacent_chunks(snapshot, chunk_id, **kwargs)

    def read_related_section(self, chunk_id: str, **kwargs: Any) -> ContextReadResult:
        """Read a section from the snapshot active when this call begins."""
        snapshot = self._current_runtime().snapshot
        return read_related_section(snapshot, chunk_id, **kwargs)

    def refresh(self, _published: Any = None) -> IndexSnapshot:
        """Atomically publish a newly loaded runtime, or retain and report the old one.

        The optional argument makes this method directly usable as ingest's
        ``refresh_callback``.
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


# 第二阶段：开发全部完成后统一验证。以下 TODO 暂不执行。

# TODO 15：集中编写并执行离线功能测试。
# 后续在 tests/unit/knowledge/ 补充检索用例，使用人工片段、可控向量和假的 Embedding 适配器。
# 原测试占位文件已清理，待验证事项集中记录在 tests/README.md。
# 覆盖索引加载、已知余弦排序、关键词命中、两路融合、去重与稳定排序。
# 覆盖 top_k、一条渠道为空、重复文本位于不同文件、来源与原文位置对应。
# 离线测试不依赖 Ollama 服务，也不读写项目真实索引。

# TODO 16：集中验证异常与模型兼容性处理。
# 覆盖 not_built、产物缺失、损坏 JSON、重复或未知片段 ID、数量不一致。
# 覆盖维度不匹配、NaN/Infinity、零向量、同名模型 digest 变化和未知文本策略。
# 覆盖空问题、输入超长、服务断开与超时，确认错误不会被伪装成正常的空结果。
# 覆盖加载期间版本变化和刷新失败，确认不会发布或使用混合版本的快照。

# TODO 17：集中执行真实 Ollama 建库与检索联调。
# 先完成一次真实建库，确认 manifest 为 ready 且所有索引产物属于同一批次。
# 使用自然语言问题“如何让 Agent 记住聊天历史？”与精确技术标识符查询，
# 核对是否能命中相关章节、代码片段，并回到对应文件检查引用行号。
# 使用 evaluation/questions.jsonl 中的预期来源评估检索覆盖情况。
# 这里只验证检索结果，最终回答质量与拒答能力留给后续 Agent 联调。

# TODO 18：集中验证索引更新及记录检索效果。
# 在临时文档集上分别新增、修改、删除文件并更新索引，刷新后检查结果变化。
# 确认被删除内容不再召回，修改内容使用新的来源记录，未变内容仍可正常检索。
# 记录典型问题的命中来源、候选数量和查询耗时，据此调整召回数量与融合参数。
# 校准 RETRIEVAL_MIN_COSINE / RETRIEVAL_SEMANTIC_MIN_COSINE / TOPIC_COVERAGE。
# 覆盖改写但无词面重合、精确 API/C++ 标识符、中文主题、纯提问词和库外问题。
# 验证过滤在 RRF/top-k 前生效，相关性为空与索引/Embedding 错误分别记录。
# 阈值需结合相关及不相关问题确定；不把“返回了 top_k 条”当作检索正确的证明。
# 验证完成后汇总通过项、失败项及待修复问题，不能将未执行项目标记为通过。
