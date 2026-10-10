"""应用配置。

读取项目根目录的 .env，校验对话模型与 Embedding 模型配置。
文档目录默认指向项目内 docs，索引目录默认指向 data/index。
相对路径应按项目根目录解析，避免受启动目录影响。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dotenv_values(path: Path) -> dict[str, str]:
    """Read the small, dependency-free subset of dotenv syntax used here."""
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


@dataclass(frozen=True, slots=True)
class IngestSettings:
    docs_dir: str
    index_dir: str
    chunk_size: int
    chunk_overlap: int
    project_root: Path = PROJECT_ROOT
    embedding_provider: str = "ollama"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_base_url: str = "http://localhost:11434"
    embedding_batch_size: int = 16
    embedding_request_timeout: float = 60.0
    embedding_max_retries: int = 2
    embedding_retry_delay: float = 0.5


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    index_dir: str
    top_k: int
    vector_candidate_count: int
    keyword_candidate_count: int
    rrf_k: int
    request_timeout: float
    project_root: Path = PROJECT_ROOT


def get_ingest_settings() -> IngestSettings:
    """Return raw ingestion settings; path and range checks happen at ingestion."""
    dotenv = _dotenv_values(PROJECT_ROOT / ".env")

    def value(name: str, default: str) -> str:
        return os.environ.get(name, dotenv.get(name, default))

    return IngestSettings(
        docs_dir=value("KNOWLEDGE_DOCS_DIR", "docs"),
        index_dir=value("KNOWLEDGE_INDEX_DIR", "data/index"),
        chunk_size=int(value("KNOWLEDGE_CHUNK_SIZE", "1000")),
        chunk_overlap=int(value("KNOWLEDGE_CHUNK_OVERLAP", "200")),
        embedding_provider=value("EMBEDDING_PROVIDER", "ollama"),
        embedding_model=value("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        embedding_base_url=value("EMBEDDING_BASE_URL", "http://localhost:11434"),
        embedding_batch_size=int(value("EMBEDDING_BATCH_SIZE", "16")),
        embedding_request_timeout=float(value("EMBEDDING_REQUEST_TIMEOUT", "60")),
        embedding_max_retries=int(value("EMBEDDING_MAX_RETRIES", "2")),
        embedding_retry_delay=float(value("EMBEDDING_RETRY_DELAY", "0.5")),
    )


def get_retrieval_settings() -> RetrievalSettings:
    """Return raw retrieval settings; path and range checks happen at retrieval."""
    dotenv = _dotenv_values(PROJECT_ROOT / ".env")

    def value(name: str, default: str) -> str:
        return os.environ.get(name, dotenv.get(name, default))

    return RetrievalSettings(
        index_dir=value("KNOWLEDGE_INDEX_DIR", "data/index"),
        top_k=int(value("RETRIEVAL_TOP_K", "5")),
        vector_candidate_count=int(value("RETRIEVAL_VECTOR_CANDIDATES", "20")),
        keyword_candidate_count=int(value("RETRIEVAL_KEYWORD_CANDIDATES", "20")),
        rrf_k=int(value("RETRIEVAL_RRF_K", "60")),
        request_timeout=float(value("EMBEDDING_REQUEST_TIMEOUT", "60")),
    )
