"""知识库管理服务模块（待开发）。

模块功能：为页面提供文档列表、原文预览、索引状态、Markdown 上传和手动更新。
输入：服务端配置的文档/索引目录、受控的文档标识、上传内容或建库操作。
输出：文档与索引的结构化状态、带行号的原文，以及上传或更新结果。

职责分工：
1. 本模块管理文档文件及知识库操作，不负责对话生成或 HTTP 请求解析。
2. knowledge/ingest.py 继续负责文档解析、切分、向量化、完整建库和增量更新。
3. api.py 负责请求校验、响应转换和 ServiceRuntime 的工作线程调度。
4. Agent 通过已有 refresh_index() 加载新索引，前端负责展示状态和操作入口。

当前仅创建职责说明和 TODO，不包含业务实现，不改变已有问答、文件或索引。
第一版只管理 Markdown；暂不提供删除文档、在线编辑或后台任务队列。
开发顺序：TODO 01—07 只读功能 → TODO 08—12 上传与更新 → TODO 13—14 接入。
全部开发完成后统一执行 TODO 15—18，本阶段不测试、不建库、不调用模型。
"""


# 第一阶段：只读管理能力。

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from pathlib import PureWindowsPath
import tempfile
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Literal

from ..config import get_knowledge_base_settings

if TYPE_CHECKING:
    from .ingest import IngestRunResult


DocumentSyncStatus = Literal[
    "pending",
    "synced",
    "modified",
    "removed",
    "unreadable",
    "unknown",
]
IndexDiskStatus = Literal["not_built", "published", "incomplete", "corrupt", "unknown"]
IndexOperation = Literal["build", "update"]
OperationErrorCategory = Literal[
    "configuration",
    "validation",
    "ingest",
    "publish",
    "refresh",
    "busy",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class KnowledgeBaseConfig:
    """管理服务使用的文件系统位置和硬性限制。"""

    docs_dir: Path
    index_dir: Path
    max_upload_bytes: int = 10 * 1024 * 1024
    max_preview_lines: int = 1_000

    def __post_init__(self) -> None:
        if self.max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be greater than zero")
        if self.max_preview_lines <= 0:
            raise ValueError("max_preview_lines must be greater than zero")


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    """一个受管文档当前的磁盘与同步状态。"""

    path: str
    exists: bool
    size_bytes: int | None
    modified_at: str | None
    sync_status: DocumentSyncStatus
    content_hash: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentPreview:
    """Markdown 文档中保留原文、带行号定位的切片。"""

    path: str
    content: str
    start_line: int
    end_line: int
    total_lines: int
    content_hash: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class DocumentChanges:
    """等待反映到已发布索引中的文档变更。"""

    added: int | None = None
    modified: int | None = None
    removed: int | None = None
    unreadable: int | None = None

    @property
    def total(self) -> int | None:
        values = (self.added, self.modified, self.removed, self.unreadable)
        if any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)


@dataclass(frozen=True, slots=True)
class IndexStatus:
    """已发布索引的元数据与当前文档同步状态。"""

    disk_status: IndexDiskStatus
    published_version: str | None
    built_at: str | None
    disk_document_count: int | None
    indexed_document_count: int | None
    chunk_count: int | None
    pending_changes: DocumentChanges
    active_version: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class IndexOperationResult:
    """一次索引操作中可独立观测的各个阶段。"""

    operation: IndexOperation
    ingest_result: IngestRunResult | None
    published: bool | None
    refreshed: bool | None
    published_version: str | None = None
    active_version: str | None = None
    error_category: OperationErrorCategory | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return (
            self.error_category is None
            and self.ingest_result is not None
            and self.ingest_result.succeeded
            and self.published is True
            and self.refreshed is not False
        )


RefreshCallback = Callable[[], object]
IndexCallback = Callable[..., "IngestRunResult"]
ProgressCallback = Callable[[int, int], None]


class KnowledgeBaseService:
    """知识库管理操作的接口。

    构造时只记录依赖；它刻意不执行任何文件访问、目录创建、
    摄取或运行时刷新。
    """

    def __init__(
        self,
        config: KnowledgeBaseConfig,
        *,
        refresh_callback: RefreshCallback | None = None,
        build_callback: IndexCallback | None = None,
        update_callback: IndexCallback | None = None,
    ) -> None:
        self.config = config
        self.refresh_callback = refresh_callback
        self.build_callback = build_callback
        self.update_callback = update_callback

    def list_documents(self, *, offset: int = 0, limit: int = 100) -> tuple[DocumentSummary, ...]:
        return _list_documents(self.config, offset=offset, limit=limit)

    def read_document(
        self,
        path: str,
        *,
        start_line: int = 1,
        limit: int = 200,
        content_hash: str | None = None,
    ) -> DocumentPreview:
        return _read_document(
            self.config,
            path,
            start_line=start_line,
            limit=limit,
            content_hash=content_hash,
        )

    def get_index_status(self) -> IndexStatus:
        return _get_index_status(self.config)

    def upload_document(self, filename: str, content: bytes | BinaryIO) -> DocumentSummary:
        return _upload_document(self.config, filename, content)

    def build_index(
        self, *, progress: ProgressCallback | None = None
    ) -> IndexOperationResult:
        return _run_index_operation(
            self.config,
            "build",
            callback=self.build_callback,
            refresh_callback=self.refresh_callback,
            progress=progress,
        )

    def update_index(
        self, *, progress: ProgressCallback | None = None
    ) -> IndexOperationResult:
        return _run_index_operation(
            self.config,
            "update",
            callback=self.update_callback,
            refresh_callback=self.refresh_callback,
            progress=progress,
        )


def _resolve_configured_path(project_root: Path, configured_path: str) -> Path:
    """独立于进程工作目录解析一个由服务端控制的路径。"""
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def load_knowledge_base_config() -> KnowledgeBaseConfig:
    """只加载知识库路径，不校验写入或模型。"""
    settings = get_knowledge_base_settings()
    project_root = settings.project_root.resolve(strict=False)
    return KnowledgeBaseConfig(
        docs_dir=_resolve_configured_path(project_root, settings.docs_dir),
        index_dir=_resolve_configured_path(project_root, settings.index_dir),
    )


def _require_readable_directory(path: Path, *, description: str) -> Path:
    """为读取操作校验目录，但不创建或修改它。"""
    if not path.exists():
        raise FileNotFoundError(f"{description}不存在: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description}不是目录: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{description}不可读: {path}")
    return path

class InvalidDocumentPathError(ValueError):
    """请求的路径超出受管文档命名空间时抛出。"""


def resolve_document_path(docs_dir: Path, document_path: str) -> Path:
    """解析相对 Markdown 路径，不允许访问 *docs_dir* 之外。"""
    if not isinstance(document_path, str) or not document_path or "\x00" in document_path:
        raise InvalidDocumentPathError("文档路径无效")

    # 即使服务在非 Windows 主机上测试或部署，PureWindowsPath 也能识别
    # 盘符相对路径和 UNC 路径。两种斜杠都视为分隔符，
    # 以免同一路径在不同平台上含义发生变化。
    windows_path = PureWindowsPath(document_path)
    normalized = document_path.replace("\\", "/")
    parts = normalized.split("/")
    if (
        Path(normalized).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or normalized.startswith("/")
        or any(part == ".." for part in parts)
    ):
        raise InvalidDocumentPathError("文档路径必须是知识库内的相对路径")

    relative_path = Path(*parts)
    if relative_path.suffix.lower() != ".md":
        raise InvalidDocumentPathError("仅支持 Markdown 文件")

    root = _require_readable_directory(Path(docs_dir).resolve(strict=True), description="文档目录")
    try:
        resolved = (root / relative_path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"文档不存在: {relative_path.as_posix()}") from exc
    except (OSError, RuntimeError) as exc:
        raise InvalidDocumentPathError("文档路径无法解析") from exc

    try:
        managed_relative_path = resolved.relative_to(root)
    except ValueError as exc:
        raise InvalidDocumentPathError("文档路径超出知识库范围") from exc

    if not resolved.is_file():
        raise InvalidDocumentPathError("文档路径不是普通文件")
    if managed_relative_path.suffix.lower() != ".md":
        raise InvalidDocumentPathError("仅支持 Markdown 文件")
    if not os.access(resolved, os.R_OK):
        raise PermissionError(f"文档不可读: {managed_relative_path.as_posix()}")
    return resolved

@dataclass(frozen=True, slots=True)
class PublishedManifestFile:
    """已发布 manifest 中一条经校验的文档记录。"""

    path: str
    content_hash: str
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublishedIndexManifest:
    """某个已发布索引经校验的公开元数据和本地路径。"""

    version: str
    built_at: str
    files: tuple[PublishedManifestFile, ...]
    chunk_count: int
    chunks_path: Path
    vector_index_path: Path
    keyword_index_path: Path
    metadata_path: Path


class IndexManifestNotFoundError(FileNotFoundError):
    """不存在任何已发布索引 manifest 时抛出。"""


class InvalidIndexManifestError(ValueError):
    """已发布索引 manifest 格式错误或不完整时抛出。"""


class IncompleteIndexError(InvalidIndexManifestError):
    """有效 manifest 引用了缺失的索引产物时抛出。"""


def _manifest_file(value: object, position: int) -> PublishedManifestFile:
    if not isinstance(value, dict):
        raise InvalidIndexManifestError(f"索引 manifest 的 files[{position}] 必须是对象")

    path = value.get("path")
    content_hash = value.get("content_hash")
    chunk_ids = value.get("chunk_ids")
    if not isinstance(path, str) or not path.strip():
        raise InvalidIndexManifestError(f"索引 manifest 的 files[{position}].path 无效")
    if not isinstance(content_hash, str) or not content_hash.strip():
        raise InvalidIndexManifestError(
            f"索引 manifest 的 files[{position}].content_hash 无效"
        )
    if not isinstance(chunk_ids, list) or any(
        not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids
    ):
        raise InvalidIndexManifestError(
            f"索引 manifest 的 files[{position}].chunk_ids 无效"
        )
    return PublishedManifestFile(path, content_hash, tuple(chunk_ids))


def _manifest_artifact(index_dir: Path, artifacts: dict[str, Any], key: str) -> Path:
    value = artifacts.get(key)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise InvalidIndexManifestError(f"索引 manifest 缺少有效产物路径: {key}")
    try:
        path = (index_dir / value).resolve(strict=True)
        path.relative_to(index_dir)
    except FileNotFoundError as exc:
        raise IncompleteIndexError(f"索引产物缺失: {key}: {value!r}") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise InvalidIndexManifestError(f"索引产物路径无效或越界: {key}: {value!r}") from exc
    if not path.is_file():
        raise IncompleteIndexError(f"索引产物缺失或不是文件: {key}: {path}")
    if not os.access(path, os.R_OK):
        raise InvalidIndexManifestError(f"索引产物不可读: {key}: {path}")
    return path


def read_published_index_manifest(index_dir: Path) -> PublishedIndexManifest:
    """读取并校验已发布索引的元数据，不加载索引内容。"""
    root = Path(index_dir).resolve(strict=False)
    manifest_path = root / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as source:
            manifest = json.load(source)
    except FileNotFoundError as exc:
        raise IndexManifestNotFoundError(f"索引清单不存在: {manifest_path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidIndexManifestError(f"无法读取索引清单: {manifest_path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise InvalidIndexManifestError("索引 manifest 顶层必须是 JSON 对象")
    if manifest.get("schema_version") != 1:
        raise InvalidIndexManifestError(
            f"不支持的索引 manifest schema_version: {manifest.get('schema_version')!r}"
        )
    if manifest.get("status") != "ready":
        raise InvalidIndexManifestError(
            f"索引 manifest 状态无效: {manifest.get('status')!r}，期望 'ready'"
        )

    built_at = manifest.get("built_at")
    files = manifest.get("files")
    chunk_count = manifest.get("chunk_count")
    artifacts = manifest.get("artifacts")
    if not isinstance(built_at, str) or not built_at.strip():
        raise InvalidIndexManifestError("索引 manifest 缺少有效的 built_at 版本")
    if not isinstance(files, list):
        raise InvalidIndexManifestError("索引 manifest 缺少有效的 files 数组")
    if (
        not isinstance(chunk_count, int)
        or isinstance(chunk_count, bool)
        or chunk_count < 0
    ):
        raise InvalidIndexManifestError("索引 manifest 的 chunk_count 无效")
    if not isinstance(artifacts, dict):
        raise InvalidIndexManifestError("索引 manifest 缺少有效的 artifacts 对象")

    published_files = tuple(
        _manifest_file(value, position) for position, value in enumerate(files)
    )
    resolved_artifacts = {
        key: _manifest_artifact(root, artifacts, key)
        for key in ("chunks", "vectors", "keywords", "metadata")
    }
    return PublishedIndexManifest(
        version=built_at,
        built_at=built_at,
        files=published_files,
        chunk_count=chunk_count,
        chunks_path=resolved_artifacts["chunks"],
        vector_index_path=resolved_artifacts["vectors"],
        keyword_index_path=resolved_artifacts["keywords"],
        metadata_path=resolved_artifacts["metadata"],
    )

def _list_documents(
    config: KnowledgeBaseConfig,
    *,
    offset: int,
    limit: int,
    manifest_state: tuple[
        PublishedIndexManifest | None, InvalidIndexManifestError | None
    ]
    | None = None,
) -> tuple[DocumentSummary, ...]:
    """返回磁盘文档与已发布文档的确定性快照。"""
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    from .ingest import get_document_identity, read_markdown_file

    docs_root = _require_readable_directory(
        Path(config.docs_dir).resolve(strict=True), description="文档目录"
    )
    disk_files: dict[str, Path] = {}
    directories = [docs_root]
    while directories:
        directory = directories.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    directories.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".md"):
                    path = Path(entry.path)
                    disk_files[path.relative_to(docs_root).as_posix()] = path

    if manifest_state is None:
        try:
            manifest_state = (read_published_index_manifest(config.index_dir), None)
        except IndexManifestNotFoundError:
            manifest_state = (None, None)
        except InvalidIndexManifestError as exc:
            manifest_state = (None, exc)
    manifest, manifest_exception = manifest_state
    published_files = (
        {item.path: item for item in manifest.files} if manifest is not None else {}
    )
    manifest_error = str(manifest_exception) if manifest_exception is not None else None

    page_paths = sorted(disk_files.keys() | published_files.keys())[
        offset : offset + limit
    ]
    summaries: list[DocumentSummary] = []
    for relative_path in page_paths:
        path = disk_files.get(relative_path)
        published = published_files.get(relative_path)
        if path is None:
            summaries.append(
                DocumentSummary(
                    path=relative_path,
                    exists=False,
                    size_bytes=None,
                    modified_at=None,
                    sync_status="removed",
                    content_hash=published.content_hash if published else None,
                )
            )
            continue

        before = None
        try:
            before = path.stat()
            document = read_markdown_file(path)
            identity = get_document_identity(document, docs_root)
            after = path.stat()
        except Exception as exc:
            exists = path.is_file()
            if published is not None and not exists:
                summaries.append(
                    DocumentSummary(
                        path=relative_path,
                        exists=False,
                        size_bytes=None,
                        modified_at=None,
                        sync_status="removed",
                        content_hash=published.content_hash,
                    )
                )
                continue
            summaries.append(
                DocumentSummary(
                    path=relative_path,
                    exists=exists,
                    size_bytes=before.st_size if before is not None else None,
                    modified_at=(
                        datetime.fromtimestamp(
                            before.st_mtime, tz=timezone.utc
                        ).isoformat()
                        if before is not None
                        else None
                    ),
                    sync_status="unreadable",
                    error=str(exc),
                )
            )
            continue

        changed_during_scan = (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or getattr(before, "st_ino", None) != getattr(after, "st_ino", None)
        )
        if changed_during_scan or identity.relative_path != relative_path:
            sync_status: DocumentSyncStatus = "unknown"
            error = "文档在扫描期间发生变化"
        elif manifest_error is not None:
            sync_status = "unknown"
            error = manifest_error
        elif published is None:
            sync_status = "pending"
            error = None
        elif identity.content_hash != published.content_hash:
            sync_status = "modified"
            error = None
        else:
            sync_status = "synced"
            error = None

        summaries.append(
            DocumentSummary(
                path=relative_path,
                exists=True,
                size_bytes=after.st_size,
                modified_at=datetime.fromtimestamp(
                    after.st_mtime, tz=timezone.utc
                ).isoformat(),
                sync_status=sync_status,
                content_hash=identity.content_hash,
                error=error,
            )
        )

    return tuple(summaries)

class DocumentReferenceMismatchError(RuntimeError):
    """当前源文本不再匹配被引用文档时抛出。"""


class DocumentTooLargeError(ValueError):
    """源文档超过配置的读取上限时抛出。"""


def _read_document(
    config: KnowledgeBaseConfig,
    document_path: str,
    *,
    start_line: int,
    limit: int,
    content_hash: str | None,
) -> DocumentPreview:
    """从一个受管文档中读取有界、保留原文的切片。"""
    if not isinstance(start_line, int) or isinstance(start_line, bool) or start_line < 1:
        raise ValueError("start_line must be a positive integer")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if limit > config.max_preview_lines:
        raise ValueError(f"limit must not exceed {config.max_preview_lines}")
    if content_hash is not None and (
        not isinstance(content_hash, str) or not content_hash.strip()
    ):
        raise ValueError("content_hash must be a non-empty string")

    from .ingest import get_document_identity, read_markdown_file

    try:
        path = resolve_document_path(config.docs_dir, document_path)
    except FileNotFoundError as exc:
        if content_hash is not None:
            raise DocumentReferenceMismatchError(
                "引用的文档已不存在，无法按原引用核对"
            ) from exc
        raise

    size_bytes = path.stat().st_size
    if size_bytes > config.max_upload_bytes:
        raise DocumentTooLargeError(
            f"文档大小超过预览限制: {size_bytes} > {config.max_upload_bytes} bytes"
        )

    docs_root = Path(config.docs_dir).resolve(strict=True)
    document = read_markdown_file(path)
    identity = get_document_identity(document, docs_root)
    if content_hash is not None and identity.content_hash != content_hash:
        raise DocumentReferenceMismatchError(
            "当前文档内容已发生变化，无法按原引用核对"
        )

    total_lines = len(document.lines)
    if total_lines == 0:
        if start_line != 1:
            raise ValueError("空文档只能从第 1 行开始预览")
        return DocumentPreview(
            path=identity.relative_path,
            content="",
            start_line=0,
            end_line=0,
            total_lines=0,
            content_hash=identity.content_hash,
            truncated=False,
        )
    if start_line > total_lines:
        raise ValueError(
            f"start_line 超出文档范围: {start_line}（文档共 {total_lines} 行）"
        )

    selected_lines = document.lines[start_line - 1 : start_line - 1 + limit]
    actual_start = selected_lines[0].number
    actual_end = selected_lines[-1].number
    return DocumentPreview(
        path=identity.relative_path,
        content="".join(line.text for line in selected_lines),
        start_line=actual_start,
        end_line=actual_end,
        total_lines=total_lines,
        content_hash=identity.content_hash,
        truncated=actual_end < total_lines,
    )

def _missing_manifest_disk_status(index_dir: Path) -> tuple[IndexDiskStatus, str | None]:
    """区分干净但尚未建库的目录与中断的发布过程。"""
    try:
        with os.scandir(index_dir) as entries:
            has_artifacts = next(entries, None) is not None
    except FileNotFoundError:
        return "not_built", None
    except OSError as exc:
        return "unknown", f"无法检查索引目录: {index_dir}: {exc}"
    if has_artifacts:
        return "incomplete", f"索引目录中存在文件，但缺少 manifest.json: {index_dir}"
    return "not_built", None


def _get_index_status(config: KnowledgeBaseConfig) -> IndexStatus:
    """返回已发布元数据与源变更的一致快照。"""
    manifest: PublishedIndexManifest | None
    manifest_exception: InvalidIndexManifestError | None
    try:
        manifest = read_published_index_manifest(config.index_dir)
        manifest_exception = None
        disk_status: IndexDiskStatus = "published"
        error = None
    except IndexManifestNotFoundError:
        manifest = None
        manifest_exception = None
        disk_status, error = _missing_manifest_disk_status(config.index_dir)
    except IncompleteIndexError as exc:
        manifest = None
        manifest_exception = exc
        disk_status = "incomplete"
        error = str(exc)
    except InvalidIndexManifestError as exc:
        manifest = None
        manifest_exception = exc
        disk_status = "corrupt"
        error = str(exc)

    documents = _list_documents(
        config,
        offset=0,
        limit=2**63 - 1,
        manifest_state=(manifest, manifest_exception),
    )
    disk_document_count = sum(document.exists for document in documents)

    if manifest_exception is not None or any(
        document.sync_status == "unknown" for document in documents
    ):
        pending_changes = DocumentChanges()
    else:
        counts = {status: 0 for status in ("pending", "modified", "removed", "unreadable")}
        for document in documents:
            if document.sync_status in counts:
                counts[document.sync_status] += 1
        pending_changes = DocumentChanges(
            added=counts["pending"],
            modified=counts["modified"],
            removed=counts["removed"],
            unreadable=counts["unreadable"],
        )

    return IndexStatus(
        disk_status=disk_status,
        published_version=manifest.version if manifest is not None else None,
        built_at=manifest.built_at if manifest is not None else None,
        disk_document_count=disk_document_count,
        indexed_document_count=len(manifest.files) if manifest is not None else None,
        chunk_count=manifest.chunk_count if manifest is not None else None,
        pending_changes=pending_changes,
        active_version=None,
        error=error,
    )


# 第二阶段：上传与索引更新。

@dataclass(frozen=True, slots=True)
class ValidatedMarkdownUpload:
    """有界的上传内容，可安全发布到文档根目录下。"""

    filename: str
    target_path: Path
    content: bytes
    text: str


class InvalidUploadFilenameError(ValueError):
    """上传名称不是单个安全的 Markdown 文件名时抛出。"""


class UploadTooLargeError(ValueError):
    """上传内容超过配置的字节上限时抛出。"""


class InvalidUploadEncodingError(ValueError):
    """上传内容不是有效 UTF-8（可带 BOM）时抛出。"""


class DocumentAlreadyExistsError(FileExistsError):
    """上传会覆盖已存在的目录项时抛出。"""


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{suffix}" for suffix in (*range(1, 10), "¹", "²", "³")),
    *(f"LPT{suffix}" for suffix in (*range(1, 10), "¹", "²", "³")),
}
_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"/\\|?*')


def _validate_upload_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename or filename in {".", ".."}:
        raise InvalidUploadFilenameError("上传文件名无效")
    if (
        "\x00" in filename
        or any(character in _WINDOWS_INVALID_FILENAME_CHARS for character in filename)
        or any(ord(character) < 32 for character in filename)
        or filename[-1] in {" ", "."}
    ):
        raise InvalidUploadFilenameError("上传文件名包含非法字符")

    windows_path = PureWindowsPath(filename)
    if windows_path.drive or windows_path.name != filename:
        raise InvalidUploadFilenameError("上传文件名必须是单个文件名")
    if Path(filename).suffix.lower() != ".md":
        raise InvalidUploadFilenameError("仅支持 Markdown 文件")
    if filename.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise InvalidUploadFilenameError("上传文件名为系统保留名称")
    return filename


def _read_bounded_upload(content: bytes | BinaryIO, max_bytes: int) -> bytes:
    if isinstance(content, bytes):
        data = content
    elif hasattr(content, "read"):
        chunks: list[bytes] = []
        size = 0
        while size <= max_bytes:
            chunk = content.read(min(64 * 1024, max_bytes + 1 - size))
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise TypeError("上传内容流必须返回 bytes")
            chunks.append(chunk)
            size += len(chunk)
        data = b"".join(chunks)
    else:
        raise TypeError("上传内容必须是 bytes 或二进制流")

    if len(data) > max_bytes:
        raise UploadTooLargeError(
            f"上传文件超过大小限制: {len(data)} > {max_bytes} bytes"
        )
    return data


def validate_markdown_upload(
    config: KnowledgeBaseConfig,
    filename: str,
    content: bytes | BinaryIO,
) -> ValidatedMarkdownUpload:
    """在不修改文件系统的情况下校验一次不覆盖的 Markdown 上传。"""
    safe_filename = _validate_upload_filename(filename)
    docs_root = _require_readable_directory(
        Path(config.docs_dir).resolve(strict=True), description="文档目录"
    )
    target_path = docs_root / safe_filename
    if os.path.lexists(target_path):
        raise DocumentAlreadyExistsError(f"文档已存在: {safe_filename}")

    data = _read_bounded_upload(content, config.max_upload_bytes)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InvalidUploadEncodingError("上传文件必须使用 UTF-8 或 UTF-8 BOM 编码") from exc

    return ValidatedMarkdownUpload(
        filename=safe_filename,
        target_path=target_path,
        content=data,
        text=text,
    )


def _upload_document(
    config: KnowledgeBaseConfig,
    filename: str,
    content: bytes | BinaryIO,
) -> DocumentSummary:
    """校验并原子地发布一个新的 Markdown 文档。"""
    upload = validate_markdown_upload(config, filename, content)
    temp_path: Path | None = None
    published = False
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=".knowledge-upload-", suffix=".tmp", dir=upload.target_path.parent
        )
        temp_path = Path(temp_name)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(upload.content)
            destination.flush()
            os.fsync(destination.fileno())

        file_stat = temp_path.stat()
        result = DocumentSummary(
            path=upload.filename,
            exists=True,
            size_bytes=file_stat.st_size,
            modified_at=datetime.fromtimestamp(
                file_stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            sync_status="pending",
            content_hash=hashlib.sha256(upload.text.encode("utf-8")).hexdigest(),
        )
        try:
            os.link(temp_path, upload.target_path)
        except FileExistsError as exc:
            raise DocumentAlreadyExistsError(f"文档已存在: {upload.filename}") from exc
        published = True
        return result
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                if not published:
                    raise

def _ingest_failure_summary(result: "IngestRunResult") -> str:
    return "; ".join(
        f"{failure.source_path}: {failure.error}" for failure in result.failures
    )


def _run_index_operation(
    config: KnowledgeBaseConfig,
    operation: IndexOperation,
    *,
    callback: IndexCallback | None,
    refresh_callback: RefreshCallback | None,
    progress: ProgressCallback | None,
) -> IndexOperationResult:
    """运行一个摄取入口，并报告其实际发布结果。

    ``progress`` 原样转发，因此表示的是 Embedding 批次进度（已完成/总数），
    而不是为整个操作虚构的百分比。
    """
    if callback is None:
        from . import ingest

        callback = ingest.build_index if operation == "build" else ingest.update_index

    try:
        ingest_result = callback(progress=progress) if progress is not None else callback()
    except Exception as exc:
        return IndexOperationResult(
            operation=operation,
            ingest_result=None,
            published=False,
            refreshed=None,
            error_category="ingest",
            error=str(exc),
        )

    if ingest_result.failures:
        return IndexOperationResult(
            operation=operation,
            ingest_result=ingest_result,
            published=False,
            refreshed=None,
            error_category="ingest",
            error=_ingest_failure_summary(ingest_result),
        )

    published_version: str | None = None
    try:
        published_version = read_published_index_manifest(config.index_dir).version
    except (IndexManifestNotFoundError, InvalidIndexManifestError):
        # 摄取结果才是发布的权威依据。单独读取状态快照失败，
        # 不得导致本层修改任何产物。
        pass

    if refresh_callback is None:
        return IndexOperationResult(
            operation=operation,
            ingest_result=ingest_result,
            published=True,
            refreshed=None,
            published_version=published_version,
        )

    try:
        snapshot = refresh_callback()
    except Exception as exc:
        return IndexOperationResult(
            operation=operation,
            ingest_result=ingest_result,
            published=True,
            refreshed=False,
            published_version=published_version,
            active_version=getattr(exc, "retained_index_version", None),
            error_category="refresh",
            error=str(exc),
        )

    return IndexOperationResult(
        operation=operation,
        ingest_result=ingest_result,
        published=True,
        refreshed=True,
        published_version=published_version,
        active_version=getattr(snapshot, "index_version", None),
    )

# 第三阶段：开发后接入 API 和页面，本次不修改其他文件。

# 第四阶段：开发完成后统一验证，本阶段不执行。

# TODO 15：验证文档列表、变化识别与预览。
# 使用独立临时目录，覆盖空目录、未建库、新增/修改/删除、坏清单和单个文件不可读。
# 覆盖中文路径、UTF-8 BOM、CRLF/LF、空文件、预览截断及引用哈希不一致。
# 确认路径越界被拒绝，读取与列状态不会创建模型、修改索引或触发真实问答。

# TODO 16：验证上传和文件保存。
# 覆盖合法上传、同名冲突、过大文件、错误编码、非法名称与保存失败。
# 确认原文件不被覆盖，失败不留下可被扫描的半成品，成功只改变文档和待同步状态。

# TODO 17：验证索引操作和刷新边界。
# 用可控入库与刷新回调覆盖成功、IngestRunResult.failures、异常、无变化与刷新失败。
# 确认发布和运行时刷新分别记录，互斥有效，没有重复刷新、重复发布或工作线程死锁。
# 验证断连后后台任务仍按真实状态结束，busy 状态直到实际完成才释放。

# TODO 18：统一联调页面到新资料问答的完整流程。
# 在独立文档/索引目录与临时会话中完成上传、更新、预览、提问和来源核对。
# 确认新问答使用新索引，旧引用遇到已变化文档时有明确提示。
# 再覆盖重启、配置错误和模型服务不可用；不拿用户现有文档、索引或聊天记录做破坏性验证。
