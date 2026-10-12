"""应用配置。

读取项目根目录的 .env，校验对话模型与 Embedding 模型配置。
文档目录默认指向项目内 docs，索引目录默认指向 data/index。
相对路径应按项目根目录解析，避免受启动目录影响。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


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
class KnowledgeBaseSettings:
    """Path-only settings shared by read-only knowledge-base operations."""

    docs_dir: str
    index_dir: str
    project_root: Path = PROJECT_ROOT


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    index_dir: str
    top_k: int
    vector_candidate_count: int
    keyword_candidate_count: int
    rrf_k: int
    request_timeout: float
    project_root: Path = PROJECT_ROOT


@dataclass(frozen=True, slots=True)
class ChatSettings:
    provider: str
    model: str
    base_url: str
    api_key: str | None = None
    request_timeout: float = 60.0
    max_retries: int = 2
    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class IntentSettings:
    """Select deterministic intent rules or optional model assistance."""

    mode: Literal["rules", "hybrid"] = "rules"


@dataclass(frozen=True, slots=True)
class McpSettings:
    """Location of the optional MCP server configuration."""

    config_path: Path | None = None
    project_root: Path = PROJECT_ROOT


@dataclass(frozen=True, slots=True)
class SessionStoreSettings:
    """Validated SQLite session-store configuration."""

    db_path: Path
    lock_timeout: float = 5.0
    project_root: Path = PROJECT_ROOT


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Hard limits for one answer and for retained conversation history."""

    max_model_calls: int = 4
    max_tool_calls: int = 6
    max_followup_searches: int = 2
    max_context_chars: int = 24_000
    max_history_turns: int = 6

    def __post_init__(self) -> None:
        for name in (
            "max_model_calls",
            "max_tool_calls",
            "max_context_chars",
            "max_history_turns",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 必须大于 0")
        if self.max_followup_searches < 0:
            raise ValueError("max_followup_searches 不能小于 0")


def get_chat_settings() -> ChatSettings:
    """Return chat settings without falling back to embedding configuration."""
    dotenv = _dotenv_values(PROJECT_ROOT / ".env")

    def value(name: str, default: str = "") -> str:
        return os.environ.get(name, dotenv.get(name, default))

    def optional_float(name: str) -> float | None:
        raw = value(name).strip()
        return float(raw) if raw else None

    def optional_int(name: str) -> int | None:
        raw = value(name).strip()
        return int(raw) if raw else None

    provider = value("CHAT_PROVIDER").strip()
    model = value("CHAT_MODEL").strip()
    missing = [
        name
        for name, configured in (("CHAT_PROVIDER", provider), ("CHAT_MODEL", model))
        if not configured
    ]
    if missing:
        raise ValueError(f"缺少对话模型配置: {', '.join(missing)}")

    api_key = value("CHAT_API_KEY").strip()
    return ChatSettings(
        provider=provider,
        model=model,
        base_url=value("CHAT_BASE_URL").strip(),
        api_key=api_key or None,
        request_timeout=float(value("CHAT_REQUEST_TIMEOUT", "60")),
        max_retries=int(value("CHAT_MAX_RETRIES", "2")),
        temperature=optional_float("CHAT_TEMPERATURE"),
        max_output_tokens=optional_int("CHAT_MAX_OUTPUT_TOKENS"),
    )


def get_intent_settings() -> IntentSettings:
    """Return intent settings without initializing or contacting a model."""
    dotenv = _dotenv_values(PROJECT_ROOT / ".env")
    mode = os.environ.get("INTENT_MODE", dotenv.get("INTENT_MODE", "rules"))
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"rules", "hybrid"}:
        raise ValueError("INTENT_MODE 必须是 rules 或 hybrid")
    return IntentSettings(mode=normalized_mode)  # type: ignore[arg-type]


def get_mcp_settings() -> McpSettings:
    """Resolve MCP_CONFIG_PATH without reading the referenced configuration."""
    dotenv = _dotenv_values(PROJECT_ROOT / ".env")
    default_path = PROJECT_ROOT / "config" / "mcp_servers.json"
    raw_path = os.environ.get(
        "MCP_CONFIG_PATH", dotenv.get(
            "MCP_CONFIG_PATH", str(default_path) if default_path.is_file() else ""
        )
    ).strip()
    if not raw_path:
        return McpSettings(project_root=PROJECT_ROOT)

    config_path = Path(raw_path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return McpSettings(
        config_path=config_path.resolve(strict=False),
        project_root=PROJECT_ROOT,
    )


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


def get_knowledge_base_settings() -> KnowledgeBaseSettings:
    """Return document/index paths without parsing unrelated ingest settings."""
    dotenv = _dotenv_values(PROJECT_ROOT / ".env")

    def value(name: str, default: str) -> str:
        return os.environ.get(name, dotenv.get(name, default))

    return KnowledgeBaseSettings(
        docs_dir=value("KNOWLEDGE_DOCS_DIR", "docs"),
        index_dir=value("KNOWLEDGE_INDEX_DIR", "data/index"),
        project_root=PROJECT_ROOT,
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


def get_session_store_settings() -> SessionStoreSettings:
    """Resolve and validate SQLite settings without touching the database."""
    dotenv = _dotenv_values(PROJECT_ROOT / ".env")

    def value(name: str, default: str) -> str:
        return os.environ.get(name, dotenv.get(name, default))

    raw_path = value("SESSION_DB_PATH", "data/sessions/chat.db").strip()
    if not raw_path:
        raise ValueError("SESSION_DB_PATH 不能为空")

    db_path = Path(raw_path).expanduser()
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    db_path = db_path.resolve(strict=False)
    if db_path.is_dir():
        raise ValueError(f"SESSION_DB_PATH 不能指向目录: {db_path}")

    raw_timeout = value(
        "SESSION_DB_LOCK_TIMEOUT",
        value("SESSION_DB_TIMEOUT", "5"),
    ).strip()
    try:
        lock_timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError("SESSION_DB_LOCK_TIMEOUT 必须是有效数字") from exc
    if not math.isfinite(lock_timeout) or lock_timeout <= 0:
        raise ValueError("SESSION_DB_LOCK_TIMEOUT 必须大于 0")

    return SessionStoreSettings(db_path=db_path, lock_timeout=lock_timeout)
