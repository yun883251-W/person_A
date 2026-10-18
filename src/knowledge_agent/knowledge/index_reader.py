"""读取并校验已经发布的本地索引。

只接收索引目录，校验 manifest、片段、向量、关键词和元数据的一致性。
不加载检索配置、不调用 Embedding、不写入索引。文件格式与原读取规则保持一致。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas import DocumentChunk


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    """一个完整的、单一版本的内存检索产物集合。"""

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
    """经校验的 manifest 及其发布的产物文件。"""

    index_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    chunks_path: Path
    vector_index_path: Path
    keyword_index_path: Path
    metadata_path: Path


class IndexNotBuiltError(RuntimeError):
    """在索引发布之前请求检索时抛出。"""


def load_published_index_artifacts(index_dir: Path) -> PublishedIndexArtifacts:
    """读取活动 manifest，并解析其完整的已发布产物集合。"""
    index_dir = Path(index_dir).resolve()
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
    """加载并校验已发布索引中保留原文的片段。"""
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


def load_index_snapshot(index_dir: Path) -> IndexSnapshot:
    """加载并校验一个完整且一致发布的索引快照。"""
    artifacts = load_published_index_artifacts(index_dir)
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
