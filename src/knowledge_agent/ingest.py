"""知识库入库与更新。

职责：将 Markdown 文档转换为带准确来源的片段，再建立并维护检索索引。
输入：文档目录、索引目录、分块配置，以及后续使用的 Embedding 模型。
输出：chunks.jsonl、向量索引、manifest.json 和本次处理统计。

实现顺序：先验证已实现的读取与分块，再按 TODO 12—18 接入 Ollama 和增量更新。
本模块不负责对话、答案生成或检索排序，也不执行文档内的代码与 Prompt。
当前包含第一阶段实现；本地向量化及完整索引发布仍待实现。
"""


# 第一阶段：读取与分块。此阶段不调用模型，不需要 API Key。

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import get_ingest_settings
from .keywords import extract_keywords
from .models import create_embedding_model
from .schemas import DocumentChunk


@dataclass(frozen=True, slots=True)
class IngestConfig:
    docs_dir: Path
    index_dir: Path
    chunk_size: int
    chunk_overlap: int
    embedding_provider: str = "ollama"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_base_url: str = "http://localhost:11434"
    embedding_batch_size: int = 16
    embedding_request_timeout: float = 60.0
    embedding_max_retries: int = 2
    embedding_retry_delay: float = 0.5


def load_ingest_config() -> IngestConfig:
    """Load and validate the filesystem and chunking configuration."""
    settings = get_ingest_settings()
    project_root = settings.project_root.resolve()

    def resolve_path(value: str | os.PathLike[str]) -> Path:
        path = Path(value).expanduser()
        return (project_root / path if not path.is_absolute() else path).resolve()

    docs_dir = resolve_path(settings.docs_dir)
    index_dir = resolve_path(settings.index_dir)

    if not docs_dir.exists():
        raise FileNotFoundError(f"文档目录不存在: {docs_dir}")
    if not docs_dir.is_dir():
        raise NotADirectoryError(f"文档路径不是目录: {docs_dir}")
    if not os.access(docs_dir, os.R_OK):
        raise PermissionError(f"文档目录不可读: {docs_dir}")

    if index_dir.exists() and not index_dir.is_dir():
        raise NotADirectoryError(f"索引路径不是目录: {index_dir}")
    writable_location = index_dir
    while not writable_location.exists():
        parent = writable_location.parent
        if parent == writable_location:
            break
        writable_location = parent
    if not writable_location.is_dir() or not os.access(writable_location, os.W_OK):
        raise PermissionError(f"索引目录不可写: {index_dir}")

    if settings.chunk_size <= 0:
        raise ValueError("片段长度必须大于 0")
    if settings.chunk_overlap < 0:
        raise ValueError("片段重叠长度不能小于 0")
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("片段重叠长度必须小于片段长度")

    embedding_provider = settings.embedding_provider.strip().lower()
    embedding_model = settings.embedding_model.strip()
    embedding_base_url = settings.embedding_base_url.strip().rstrip("/")
    if embedding_provider != "ollama":
        raise ValueError(f"不支持的 Embedding provider: {settings.embedding_provider}")
    if not embedding_model:
        raise ValueError("EMBEDDING_MODEL 不能为空")
    if not embedding_base_url:
        raise ValueError("EMBEDDING_BASE_URL 不能为空")
    if settings.embedding_batch_size <= 0:
        raise ValueError("EMBEDDING_BATCH_SIZE 必须大于 0")
    if settings.embedding_request_timeout <= 0:
        raise ValueError("EMBEDDING_REQUEST_TIMEOUT 必须大于 0")
    if settings.embedding_max_retries < 0:
        raise ValueError("EMBEDDING_MAX_RETRIES 不能小于 0")
    if settings.embedding_retry_delay < 0:
        raise ValueError("EMBEDDING_RETRY_DELAY 不能小于 0")

    return IngestConfig(
        docs_dir=docs_dir,
        index_dir=index_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        embedding_batch_size=settings.embedding_batch_size,
        embedding_request_timeout=settings.embedding_request_timeout,
        embedding_max_retries=settings.embedding_max_retries,
        embedding_retry_delay=settings.embedding_retry_delay,
    )

def scan_markdown_files(docs_dir: Path) -> list[Path]:
    """Return Markdown files below *docs_dir* in deterministic path order."""
    docs_dir = docs_dir.resolve()
    markdown_files: list[Path] = []
    directories = [docs_dir]

    while directories:
        directory = directories.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    directories.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".md"):
                    markdown_files.append(Path(entry.path))

    if not markdown_files:
        raise FileNotFoundError(f"文档目录中未发现 Markdown 文件: {docs_dir}")

    markdown_files.sort(key=lambda path: path.relative_to(docs_dir).as_posix())
    return markdown_files

@dataclass(frozen=True, slots=True)
class SourceLine:
    """One source line, including its original line ending when present."""

    number: int
    text: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """Decoded source text together with a one-based source-line mapping."""

    path: Path
    text: str
    lines: tuple[SourceLine, ...]

    @property
    def is_empty(self) -> bool:
        return not self.text


class DocumentReadError(RuntimeError):
    """Raised when a source document cannot be decoded or read."""


def read_markdown_file(path: Path) -> SourceDocument:
    """Read UTF-8 Markdown without changing its line endings or blank lines."""
    path = Path(path)
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise DocumentReadError(f"Failed to read Markdown file {path}: {exc}") from exc

    raw_lines = text.splitlines(keepends=True)
    lines: list[SourceLine] = []
    offset = 0
    append = lines.append
    for number, line in enumerate(raw_lines, start=1):
        end_offset = offset + len(line)
        append(SourceLine(number, line, offset, end_offset))
        offset = end_offset

    return SourceDocument(path=path, text=text, lines=tuple(lines))

@dataclass(frozen=True, slots=True)
class DocumentIdentity:
    """Stable source identity and a fingerprint of its decoded content."""

    relative_path: str
    content_hash: str


def get_document_identity(document: SourceDocument, docs_dir: Path) -> DocumentIdentity:
    """Return the normalized docs-relative path and SHA-256 content hash."""
    docs_dir = Path(docs_dir).resolve()
    path = document.path.resolve()
    try:
        relative_path = path.relative_to(docs_dir).as_posix()
    except ValueError as exc:
        raise ValueError(f"文档不在文档目录中: {path}") from exc

    content_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    return DocumentIdentity(relative_path=relative_path, content_hash=content_hash)

@dataclass(frozen=True, slots=True)
class MarkdownBlock:
    """A source-preserving Markdown text or fenced-code block."""

    kind: Literal["text", "code"]
    text: str
    content: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    language: str | None = None
    info_string: str | None = None
    fence_character: str | None = None
    fence_length: int | None = None
    closed: bool = True


@dataclass(frozen=True, slots=True)
class MarkdownParseIssue:
    """A non-fatal problem found while recognizing Markdown blocks."""

    message: str
    line: int
    offset: int


@dataclass(frozen=True, slots=True)
class MarkdownParseResult:
    blocks: tuple[MarkdownBlock, ...]
    issues: tuple[MarkdownParseIssue, ...]


def _opening_fence(line: str) -> tuple[str, int, str] | None:
    candidate = line.rstrip("\r\n")
    indent = len(candidate) - len(candidate.lstrip(" "))
    if indent > 3 or indent == len(candidate):
        return None
    candidate = candidate[indent:]
    character = candidate[0]
    if character not in "`~":
        return None
    length = len(candidate) - len(candidate.lstrip(character))
    if length < 3:
        return None
    info = candidate[length:].strip()
    if character == "`" and "`" in info:
        return None
    return character, length, info


def _is_closing_fence(line: str, character: str, minimum_length: int) -> bool:
    candidate = line.rstrip("\r\n")
    indent = len(candidate) - len(candidate.lstrip(" "))
    if indent > 3:
        return False
    candidate = candidate[indent:]
    length = len(candidate) - len(candidate.lstrip(character))
    return length >= minimum_length and not candidate[length:].strip(" \t")


def parse_markdown_blocks(document: SourceDocument) -> MarkdownParseResult:
    """Split *document* into text and fenced-code blocks in one linear pass."""
    lines = document.lines
    blocks: list[MarkdownBlock] = []
    issues: list[MarkdownParseIssue] = []
    text_start = 0
    index = 0

    def add_text(start: int, stop: int) -> None:
        if start >= stop:
            return
        first, last = lines[start], lines[stop - 1]
        raw = document.text[first.start_offset:last.end_offset]
        blocks.append(MarkdownBlock(
            kind="text", text=raw, content=raw,
            start_line=first.number, end_line=last.number,
            start_offset=first.start_offset, end_offset=last.end_offset,
        ))

    while index < len(lines):
        opening = _opening_fence(lines[index].text)
        if opening is None:
            index += 1
            continue

        add_text(text_start, index)
        character, fence_length, info = opening
        closing = index + 1
        while closing < len(lines) and not _is_closing_fence(
            lines[closing].text, character, fence_length
        ):
            closing += 1

        closed = closing < len(lines)
        stop = closing + 1 if closed else len(lines)
        first, last = lines[index], lines[stop - 1]
        content_start = first.end_offset
        content_end = lines[closing].start_offset if closed else last.end_offset
        language = info.split(None, 1)[0] if info else None
        blocks.append(MarkdownBlock(
            kind="code",
            text=document.text[first.start_offset:last.end_offset],
            content=document.text[content_start:content_end],
            start_line=first.number,
            end_line=last.number,
            start_offset=first.start_offset,
            end_offset=last.end_offset,
            language=language,
            info_string=info or None,
            fence_character=character,
            fence_length=fence_length,
            closed=closed,
        ))
        if not closed:
            issues.append(MarkdownParseIssue(
                message=f"Unclosed code fence in {document.path} at line {first.number}",
                line=first.number,
                offset=first.start_offset,
            ))
        index = stop
        text_start = stop

    add_text(text_start, len(lines))
    return MarkdownParseResult(tuple(blocks), tuple(issues))

@dataclass(frozen=True, slots=True)
class MarkdownSection:
    """A heading-scoped, source-preserving part of a Markdown document."""

    source_path: Path
    heading_path: tuple[str, ...]
    text: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    heading_level: int | None = None
    heading: str | None = None


_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)?[ \t]*|)[\r\n]*$")
_CLOSING_HASHES = re.compile(r"[ \t]+#+[ \t]*$")


def _parse_atx_heading(line: str) -> tuple[int, str] | None:
    match = _ATX_HEADING.match(line)
    if match is None:
        return None
    title = _CLOSING_HASHES.sub("", match.group(2) or "").strip()
    return len(match.group(1)), title


def organize_markdown_sections(
    document: SourceDocument,
    parsed: MarkdownParseResult | None = None,
) -> tuple[MarkdownSection, ...]:
    """Organize a document into exact source ranges using ATX heading levels."""
    if not document.lines:
        return ()

    parsed = parsed or parse_markdown_blocks(document)
    headings: list[tuple[int, int, str]] = []
    append_heading = headings.append

    # Only text blocks are inspected, so heading-like lines inside code fences
    # remain part of their surrounding section.
    for block in parsed.blocks:
        if block.kind != "text":
            continue
        for line_index in range(block.start_line - 1, block.end_line):
            heading = _parse_atx_heading(document.lines[line_index].text)
            if heading is not None:
                level, title = heading
                append_heading((line_index, level, title))

    if not headings:
        return (MarkdownSection(
            source_path=document.path,
            heading_path=(),
            text=document.text,
            start_line=1,
            end_line=document.lines[-1].number,
            start_offset=0,
            end_offset=len(document.text),
        ),)

    sections: list[MarkdownSection] = []
    stack: list[tuple[int, str]] = []

    def add_section(
        start_line_index: int,
        stop_line_index: int,
        heading_path: tuple[str, ...],
        level: int | None = None,
        heading: str | None = None,
    ) -> None:
        first = document.lines[start_line_index]
        last = document.lines[stop_line_index - 1]
        sections.append(MarkdownSection(
            source_path=document.path,
            heading_path=heading_path,
            text=document.text[first.start_offset:last.end_offset],
            start_line=first.number,
            end_line=last.number,
            start_offset=first.start_offset,
            end_offset=last.end_offset,
            heading_level=level,
            heading=heading,
        ))

    first_heading_line = headings[0][0]
    if first_heading_line:
        add_section(0, first_heading_line, ())

    for index, (line_index, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        stop_line_index = (
            headings[index + 1][0] if index + 1 < len(headings) else len(document.lines)
        )
        add_section(
            line_index,
            stop_line_index,
            tuple(item[1] for item in stack),
            level,
            title,
        )

    return tuple(sections)

@dataclass(frozen=True, slots=True)
class MarkdownSectionChunk:
    """A contiguous source slice; length is measured in Unicode characters."""

    source_path: Path
    heading_path: tuple[str, ...]
    text: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class OversizedMarkdownBlock:
    """An indivisible source block which is larger than the configured limit."""

    source_path: Path
    heading_path: tuple[str, ...]
    kind: Literal["paragraph", "code"]
    text: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    character_count: int


@dataclass(frozen=True, slots=True)
class MarkdownChunkingResult:
    chunks: tuple[MarkdownSectionChunk, ...]
    oversized_blocks: tuple[OversizedMarkdownBlock, ...]
    length_unit: Literal["unicode_characters"] = "unicode_characters"


@dataclass(frozen=True, slots=True)
class _SectionContentBlock:
    kind: Literal["paragraph", "code"]
    start: int
    end: int
    start_line: int
    end_line: int


def _section_content_blocks(section: MarkdownSection) -> list[_SectionContentBlock]:
    """Partition a section without splitting paragraphs or fenced code blocks."""
    lines = section.text.splitlines(keepends=True)
    if not lines:
        return []

    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    blocks: list[_SectionContentBlock] = []

    def add_paragraphs(first: int, stop: int) -> None:
        index = first
        while index < stop:
            block_start = index
            # Blank lines belong to the preceding paragraph. Leading blank lines
            # form their own block, which keeps the source partition lossless.
            if lines[index].strip(" \t\r\n"):
                index += 1
                while index < stop and lines[index].strip(" \t\r\n"):
                    index += 1
            while index < stop and not lines[index].strip(" \t\r\n"):
                index += 1
            block_end = starts[index] if index < len(lines) else len(section.text)
            blocks.append(_SectionContentBlock(
                kind="paragraph",
                start=starts[block_start],
                end=block_end,
                start_line=section.start_line + block_start,
                end_line=section.start_line + index - 1,
            ))

    text_start = 0
    index = 0
    while index < len(lines):
        opening = _opening_fence(lines[index])
        if opening is None:
            index += 1
            continue

        add_paragraphs(text_start, index)
        character, minimum_length, _ = opening
        closing = index + 1
        while closing < len(lines) and not _is_closing_fence(
            lines[closing], character, minimum_length
        ):
            closing += 1
        stop = closing + 1 if closing < len(lines) else len(lines)
        while stop < len(lines) and not lines[stop].strip(" \t\r\n"):
            stop += 1
        block_end = starts[stop] if stop < len(lines) else len(section.text)
        blocks.append(_SectionContentBlock(
            kind="code",
            start=starts[index],
            end=block_end,
            start_line=section.start_line + index,
            end_line=section.start_line + stop - 1,
        ))
        index = stop
        text_start = stop

    add_paragraphs(text_start, len(lines))
    return blocks


def split_markdown_sections(
    sections: tuple[MarkdownSection, ...] | list[MarkdownSection],
    chunk_size: int,
    chunk_overlap: int,
) -> MarkdownChunkingResult:
    """Split sections on natural boundaries using character-sized limits.

    Paragraphs (including Markdown tables and unfenced diagrams) and fenced code
    are atomic. Whole trailing blocks are reused for overlap, so no source block
    is cut merely to hit an exact limit. An atomic block over ``chunk_size`` is
    emitted intact as a chunk and also returned in ``oversized_blocks``.
    """
    if chunk_size <= 0:
        raise ValueError("片段长度必须大于 0")
    if chunk_overlap < 0:
        raise ValueError("片段重叠长度不能小于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("片段重叠长度必须小于片段长度")

    chunks: list[MarkdownSectionChunk] = []
    oversized: list[OversizedMarkdownBlock] = []

    for section in sections:
        blocks = _section_content_blocks(section)
        for block in blocks:
            block_length = block.end - block.start
            if block_length > chunk_size:
                oversized.append(OversizedMarkdownBlock(
                    source_path=section.source_path,
                    heading_path=section.heading_path,
                    kind=block.kind,
                    text=section.text[block.start:block.end],
                    start_line=block.start_line,
                    end_line=block.end_line,
                    start_offset=section.start_offset + block.start,
                    end_offset=section.start_offset + block.end,
                    character_count=block_length,
                ))

        start = 0
        while start < len(blocks):
            stop = start + 1
            while stop < len(blocks) and blocks[stop].end - blocks[start].start <= chunk_size:
                stop += 1

            first, last = blocks[start], blocks[stop - 1]
            chunks.append(MarkdownSectionChunk(
                source_path=section.source_path,
                heading_path=section.heading_path,
                text=section.text[first.start:last.end],
                start_line=first.start_line,
                end_line=last.end_line,
                start_offset=section.start_offset + first.start,
                end_offset=section.start_offset + last.end,
            ))
            if stop == len(blocks):
                break

            next_start = stop
            overlap_length = 0
            # Reuse only complete blocks and always advance by at least one
            # block, preventing large paragraphs from causing a stalled loop.
            while next_start > start + 1:
                candidate = blocks[next_start - 1]
                candidate_length = candidate.end - candidate.start
                if (
                    overlap_length + candidate_length > chunk_overlap
                    or blocks[stop].end - candidate.start > chunk_size
                ):
                    break
                overlap_length += candidate_length
                next_start -= 1
            start = next_start

    return MarkdownChunkingResult(tuple(chunks), tuple(oversized))

def add_chunk_metadata(
    document: SourceDocument,
    chunks: tuple[MarkdownSectionChunk, ...] | list[MarkdownSectionChunk],
    docs_dir: Path,
) -> tuple[DocumentChunk, ...]:
    """Convert source slices to traceable chunks without altering their text."""
    identity = get_document_identity(document, docs_dir)
    source_path = document.path.resolve()
    document_text = document.text
    document_lines = document.lines
    text_length = len(document_text)
    line_count = len(document_lines)
    result: list[DocumentChunk] = []
    append = result.append
    resolved_paths: dict[Path, Path] = {}

    for chunk in chunks:
        chunk_source_path = resolved_paths.get(chunk.source_path)
        if chunk_source_path is None:
            chunk_source_path = chunk.source_path.resolve()
            resolved_paths[chunk.source_path] = chunk_source_path
        if chunk_source_path != source_path:
            raise ValueError(f"片段来源与文档不一致: {chunk.source_path}")
        if not (0 <= chunk.start_offset <= chunk.end_offset <= text_length):
            raise ValueError(f"片段偏移越界: {identity.relative_path}")
        if document_text[chunk.start_offset:chunk.end_offset] != chunk.text:
            raise ValueError(f"片段正文无法对应原文: {identity.relative_path}")
        if not (1 <= chunk.start_line <= chunk.end_line <= line_count):
            raise ValueError(f"片段行号越界: {identity.relative_path}")

        first_line = document_lines[chunk.start_line - 1]
        last_line = document_lines[chunk.end_line - 1]
        if not (
            first_line.start_offset <= chunk.start_offset < first_line.end_offset
            and last_line.start_offset < chunk.end_offset <= last_line.end_offset
        ):
            raise ValueError(f"片段行号与原文偏移不一致: {identity.relative_path}")

        append(DocumentChunk(
            text=chunk.text,
            source_path=identity.relative_path,
            heading_path=chunk.heading_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            content_hash=identity.content_hash,
        ))

    return tuple(result)

@dataclass(frozen=True, slots=True)
class ChunkIdAssignment:
    chunks: tuple[DocumentChunk, ...]
    file_chunk_ids: dict[str, tuple[str, ...]]


def assign_chunk_ids(
    chunks: tuple[DocumentChunk, ...] | list[DocumentChunk],
) -> ChunkIdAssignment:
    """Assign reproducible IDs and group them by source file."""
    assigned: list[DocumentChunk] = []
    ids_by_file: dict[str, list[str]] = {}
    seen: set[str] = set()

    for chunk in chunks:
        text_digest = hashlib.sha256(chunk.text.encode("utf-8")).digest()
        identity = hashlib.sha256()
        for value in (
            chunk.source_path.encode("utf-8"),
            str(chunk.start_line).encode("ascii"),
            str(chunk.end_line).encode("ascii"),
            text_digest,
        ):
            identity.update(len(value).to_bytes(8, "big"))
            identity.update(value)
        chunk_id = identity.hexdigest()

        if chunk_id in seen:
            raise ValueError(f"重复的片段身份: {chunk.source_path}:{chunk.start_line}-{chunk.end_line}")
        seen.add(chunk_id)
        assigned.append(replace(chunk, chunk_id=chunk_id))
        ids_by_file.setdefault(chunk.source_path, []).append(chunk_id)

    return ChunkIdAssignment(
        chunks=tuple(assigned),
        file_chunk_ids={path: tuple(ids) for path, ids in ids_by_file.items()},
    )

def validate_and_export_chunks(
    chunks: tuple[DocumentChunk, ...] | list[DocumentChunk],
    docs_dir: Path,
    output_path: Path,
) -> Path:
    """Validate *chunks* and atomically publish them as UTF-8 JSONL."""
    docs_dir = Path(docs_dir).resolve()
    output_path = Path(output_path)
    seen_ids: set[str] = set()
    source_line_counts: dict[str, int] = {}

    for chunk in chunks:
        if not chunk.text.strip():
            raise ValueError(
                f"片段正文为空: {chunk.source_path}:{chunk.start_line}-{chunk.end_line}"
            )
        if not chunk.chunk_id:
            raise ValueError(
                f"片段缺少 ID: {chunk.source_path}:{chunk.start_line}-{chunk.end_line}"
            )
        if chunk.chunk_id in seen_ids:
            raise ValueError(f"重复的片段 ID: {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)

        line_count = source_line_counts.get(chunk.source_path)
        if line_count is None:
            source_path = docs_dir / Path(chunk.source_path)
            if not source_path.is_file():
                raise FileNotFoundError(f"片段来源文件不存在: {source_path}")
            try:
                with source_path.open("rb") as source:
                    line_count = sum(1 for _ in source)
            except OSError as exc:
                raise OSError(f"无法读取片段来源文件: {source_path}") from exc
            source_line_counts[chunk.source_path] = line_count

        if (
            chunk.start_line < 1
            or chunk.end_line < chunk.start_line
            or chunk.end_line > line_count
        ):
            raise ValueError(
                f"片段行号范围无效: {chunk.source_path}:"
                f"{chunk.start_line}-{chunk.end_line} (共 {line_count} 行)"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for chunk in chunks:
                record = {
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                    "heading_path": chunk.heading_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content_hash": chunk.content_hash,
                    "content_type": chunk.content_type,
                    "chunk_id": chunk.chunk_id,
                }
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return output_path

@dataclass(frozen=True, slots=True)
class FirstStageStatistics:
    discovered_files: int
    read_files: int
    blank_files: int
    failed_files: int
    sections: int
    chunks: int
    oversized_blocks: int


@dataclass(frozen=True, slots=True)
class DocumentReadFailure:
    source_path: str
    error: str


@dataclass(frozen=True, slots=True)
class FirstStageSpotCheck:
    category: Literal["paragraph", "code", "table", "26-file"]
    source_path: str
    start_line: int
    end_line: int
    chunk_id: str


@dataclass(frozen=True, slots=True)
class FirstStageResult:
    statistics: FirstStageStatistics
    chunks: tuple[DocumentChunk, ...]
    file_chunk_ids: dict[str, tuple[str, ...]]
    failures: tuple[DocumentReadFailure, ...]
    spot_checks: tuple[FirstStageSpotCheck, ...]
    output_path: Path | None


_FENCED_CODE = re.compile(r"(?m)^ {0,3}(?:`{3,}|~{3,})")
_TABLE_SEPARATOR = re.compile(
    r"(?m)^ {0,3}\|?[ \t]*:?-{3,}:?[ \t]*"
    r"(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t\r]*$"
)


def _verify_document_coverage(
    document: SourceDocument,
    chunks: tuple[MarkdownSectionChunk, ...],
) -> None:
    """Ensure the chunks cover the complete source, allowing only overlap."""
    if not chunks:
        raise ValueError(f"非空文档未生成片段: {document.path}")

    cursor = 0
    for chunk in sorted(chunks, key=lambda item: (item.start_offset, item.end_offset)):
        if chunk.start_offset > cursor:
            raise ValueError(
                f"文档内容未被分块覆盖: {document.path} offset {cursor}"
            )
        cursor = max(cursor, chunk.end_offset)
    if cursor != len(document.text):
        raise ValueError(f"文档末尾未被分块覆盖: {document.path} offset {cursor}")


def _verify_chunk_citation(chunk: DocumentChunk, document: SourceDocument) -> None:
    """Check that a chunk is the exact source slice named by its line citation."""
    if not (1 <= chunk.start_line <= chunk.end_line <= len(document.lines)):
        raise ValueError(
            f"片段行号越界: {chunk.source_path}:{chunk.start_line}-{chunk.end_line}"
        )
    start = document.lines[chunk.start_line - 1].start_offset
    end = document.lines[chunk.end_line - 1].end_offset
    if document.text[start:end] != chunk.text:
        raise ValueError(
            f"片段正文与引用位置不一致: "
            f"{chunk.source_path}:{chunk.start_line}-{chunk.end_line}"
        )


def _select_first_stage_spot_checks(
    chunks: tuple[DocumentChunk, ...],
) -> tuple[FirstStageSpotCheck, ...]:
    selected: list[tuple[Literal["paragraph", "code", "table", "26-file"], DocumentChunk]] = []

    def first_matching(
        category: Literal["paragraph", "code", "table"],
        predicate: Callable[[str], object],
    ) -> None:
        match = next((chunk for chunk in chunks if predicate(chunk.text)), None)
        if match is not None:
            selected.append((category, match))

    first_matching("code", _FENCED_CODE.search)
    first_matching("table", _TABLE_SEPARATOR.search)
    first_matching(
        "paragraph",
        lambda text: bool(text.strip())
        and _FENCED_CODE.search(text) is None
        and _TABLE_SEPARATOR.search(text) is None,
    )

    seen_26_files: set[str] = set()
    for chunk in chunks:
        if Path(chunk.source_path).name.startswith("26-") and chunk.source_path not in seen_26_files:
            selected.append(("26-file", chunk))
            seen_26_files.add(chunk.source_path)
            if len(seen_26_files) == 2:
                break

    return tuple(
        FirstStageSpotCheck(
            category=category,
            source_path=chunk.source_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_id=chunk.chunk_id or "",
        )
        for category, chunk in selected
    )


def run_first_stage(
    config: IngestConfig | None = None,
    output_path: Path | None = None,
) -> FirstStageResult:
    """Read, chunk, verify, and export the non-vector first-stage corpus."""
    config = config or load_ingest_config()
    files = scan_markdown_files(config.docs_dir)
    documents: dict[str, SourceDocument] = {}
    failures: list[DocumentReadFailure] = []
    all_chunks: list[DocumentChunk] = []
    section_count = 0
    oversized_count = 0
    blank_count = 0
    read_count = 0

    for path in files:
        relative_path = path.resolve().relative_to(config.docs_dir.resolve()).as_posix()
        try:
            document = read_markdown_file(path)
        except DocumentReadError as exc:
            failures.append(DocumentReadFailure(relative_path, str(exc)))
            continue

        read_count += 1
        documents[relative_path] = document
        if not document.text.strip():
            blank_count += 1
            continue

        parsed = parse_markdown_blocks(document)
        sections = organize_markdown_sections(document, parsed)
        chunking = split_markdown_sections(
            sections, config.chunk_size, config.chunk_overlap
        )
        _verify_document_coverage(document, chunking.chunks)
        all_chunks.extend(add_chunk_metadata(document, chunking.chunks, config.docs_dir))
        section_count += len(sections)
        oversized_count += len(chunking.oversized_blocks)

    assignment = assign_chunk_ids(all_chunks)
    for chunk in assignment.chunks:
        _verify_chunk_citation(chunk, documents[chunk.source_path])

    published_path: Path | None = None
    if not failures:
        published_path = output_path or config.index_dir / "chunks.jsonl"
        validate_and_export_chunks(assignment.chunks, config.docs_dir, published_path)

    statistics = FirstStageStatistics(
        discovered_files=len(files),
        read_files=read_count,
        blank_files=blank_count,
        failed_files=len(failures),
        sections=section_count,
        chunks=len(assignment.chunks),
        oversized_blocks=oversized_count,
    )
    return FirstStageResult(
        statistics=statistics,
        chunks=assignment.chunks,
        file_chunk_ids=assignment.file_chunk_ids,
        failures=tuple(failures),
        spot_checks=_select_first_stage_spot_checks(assignment.chunks),
        output_path=published_path,
    )


# 第二阶段：通过本地 Ollama 生成向量并发布完整索引。
# 第一阶段验证通过后再实现；以下配置名称是待接入约定，不代表已经生效。
# 接口参考：https://docs.ollama.com/api/embed
# 模型清单参考：https://docs.ollama.com/api/tags

@dataclass(frozen=True, slots=True)
class OllamaEmbeddingModel:
    """A verified Ollama embedding adapter and its immutable model identity."""

    adapter: Any
    provider: str
    model: str
    base_url: str
    digest: str
    dimension: int


class OllamaEmbeddingError(RuntimeError):
    """Raised when the configured local embedding model cannot be used."""


def check_ollama_embedding_model(
    config: IngestConfig | None = None,
    *,
    timeout: float = 5.0,
    opener: Callable[..., Any] = urlopen,
    adapter: Any | None = None,
) -> OllamaEmbeddingModel:
    """Verify Ollama, resolve the model digest, and probe its output dimension."""
    config = config or load_ingest_config()
    if config.embedding_provider != "ollama":
        raise ValueError(f"不支持的 Embedding provider: {config.embedding_provider}")

    tags_url = f"{config.embedding_base_url.rstrip('/')}/api/tags"
    request = Request(tags_url, headers={"Accept": "application/json"})
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise OllamaEmbeddingError(
            f"Ollama 模型清单请求失败（HTTP {exc.code}）: {tags_url}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OllamaEmbeddingError(
            f"无法连接 Ollama 服务，请确认服务已启动: {config.embedding_base_url}"
        ) from exc
    except (UnicodeError, json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise OllamaEmbeddingError(f"Ollama 模型清单响应无效: {tags_url}") from exc

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise OllamaEmbeddingError(f"Ollama 模型清单响应缺少 models: {tags_url}")

    model_info = next(
        (
            item
            for item in models
            if isinstance(item, dict)
            and config.embedding_model in (item.get("name"), item.get("model"))
        ),
        None,
    )
    if model_info is None:
        raise OllamaEmbeddingError(
            f"Ollama 尚未下载 Embedding 模型 {config.embedding_model}；"
            f"请先运行 ollama pull {config.embedding_model}"
        )
    digest = model_info.get("digest")
    if not isinstance(digest, str) or not digest:
        raise OllamaEmbeddingError(f"Ollama 未返回模型 digest: {config.embedding_model}")

    embedding_adapter = adapter
    if embedding_adapter is None:
        embedding_adapter = create_embedding_model(
            provider=config.embedding_provider,
            model=config.embedding_model,
            base_url=config.embedding_base_url,
            request_timeout=config.embedding_request_timeout,
        )
    try:
        vectors = embedding_adapter.embed_documents(["embedding model health check"])
    except Exception as exc:
        raise OllamaEmbeddingError(
            f"Ollama 模型无法生成 Embedding: {config.embedding_model}: {exc}"
        ) from exc

    if not isinstance(vectors, list) or len(vectors) != 1 or not vectors[0]:
        raise OllamaEmbeddingError(f"Ollama 模型返回了无效向量: {config.embedding_model}")
    vector = vectors[0]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
        raise OllamaEmbeddingError(f"Ollama 模型返回了非有限向量值: {config.embedding_model}")

    return OllamaEmbeddingModel(
        adapter=embedding_adapter,
        provider=config.embedding_provider,
        model=config.embedding_model,
        base_url=config.embedding_base_url,
        digest=digest,
        dimension=len(vector),
    )

@dataclass(frozen=True, slots=True)
class DocumentEmbeddingResult:
    """Validated vectors in exactly the same order as the input chunks."""

    chunk_ids: tuple[str, ...]
    vectors: tuple[tuple[float, ...], ...]
    dimension: int
    completed_batches: int

    @property
    def by_chunk_id(self) -> dict[str, tuple[float, ...]]:
        return dict(zip(self.chunk_ids, self.vectors, strict=True))


@dataclass(frozen=True, slots=True)
class TemporaryIndexBuildResult:
    """Artifacts and probe result for a fully validated temporary index."""

    index_dir: Path
    vector_index_path: Path
    keyword_index_path: Path
    metadata_path: Path
    chunk_count: int
    dimension: int
    validation_chunk_id: str | None


_TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "PoolTimeout",
        "ReadTimeout",
        "TimeoutException",
        "WriteTimeout",
    }
)
_LOGGER = logging.getLogger(__name__)


def _is_transient_embedding_error(exc: BaseException) -> bool:
    """Recognize transport, timeout, rate-limit, and temporary server failures."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, ConnectionError, URLError)):
            return True
        if isinstance(current, HTTPError):
            return current.code in _TRANSIENT_HTTP_STATUS
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status is None:
            status = getattr(current, "status_code", None)
        if status in _TRANSIENT_HTTP_STATUS:
            return True
        if type(current).__name__ in _TRANSIENT_EXCEPTION_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


def _validate_embedding_batch(
    vectors: Any,
    *,
    expected_count: int,
    expected_dimension: int,
    chunk_ids: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        actual = len(vectors) if isinstance(vectors, list) else "非列表"
        raise OllamaEmbeddingError(
            f"Ollama 返回向量数量错误（期望 {expected_count}，实际 {actual}）；"
            f"片段 ID: {', '.join(chunk_ids)}"
        )

    validated: list[tuple[float, ...]] = []
    for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
        if not isinstance(vector, (list, tuple)) or len(vector) != expected_dimension:
            actual = len(vector) if isinstance(vector, (list, tuple)) else "非序列"
            raise OllamaEmbeddingError(
                f"片段 {chunk_id} 的向量维度错误（期望 {expected_dimension}，实际 {actual}）"
            )
        if not all(
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in vector
        ):
            raise OllamaEmbeddingError(f"片段 {chunk_id} 的向量包含非有限或非数值元素")
        if not any(value != 0 for value in vector):
            raise OllamaEmbeddingError(f"片段 {chunk_id} 的向量为零向量")
        validated.append(tuple(float(value) for value in vector))
    return tuple(validated)


def generate_document_embeddings(
    chunks: tuple[DocumentChunk, ...] | list[DocumentChunk],
    embedding_model: OllamaEmbeddingModel,
    config: IngestConfig | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> DocumentEmbeddingResult:
    """Generate and validate document vectors in bounded Ollama batches."""
    config = config or load_ingest_config()
    ordered_chunks = tuple(chunks)
    chunk_ids = tuple(chunk.chunk_id or "" for chunk in ordered_chunks)
    if any(not chunk_id for chunk_id in chunk_ids):
        raise ValueError("所有片段必须先分配 chunk_id 才能生成向量")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("片段 chunk_id 必须唯一")
    if embedding_model.dimension <= 0:
        raise ValueError("Embedding 向量维度必须大于 0")

    total = len(ordered_chunks)
    vectors: list[tuple[float, ...]] = []
    completed_batches = 0
    for start in range(0, total, config.embedding_batch_size):
        batch = ordered_chunks[start : start + config.embedding_batch_size]
        batch_ids = chunk_ids[start : start + len(batch)]
        texts = [chunk.retrieval_text for chunk in batch]

        for attempt in range(config.embedding_max_retries + 1):
            try:
                raw_vectors = embedding_model.adapter.embed_documents(texts)
                validated = _validate_embedding_batch(
                    raw_vectors,
                    expected_count=len(batch),
                    expected_dimension=embedding_model.dimension,
                    chunk_ids=batch_ids,
                )
                break
            except OllamaEmbeddingError:
                raise
            except Exception as exc:
                if attempt >= config.embedding_max_retries or not _is_transient_embedding_error(exc):
                    raise OllamaEmbeddingError(
                        f"Ollama 向量化失败；片段 ID: {', '.join(batch_ids)}: {exc}"
                    ) from exc
                sleeper(config.embedding_retry_delay * (2**attempt))

        vectors.extend(validated)
        completed_batches += 1
        done = min(start + len(batch), total)
        _LOGGER.info("Ollama 向量化进度: %d/%d", done, total)
        if progress is not None:
            progress(done, total)

    return DocumentEmbeddingResult(
        chunk_ids=chunk_ids,
        vectors=tuple(vectors),
        dimension=embedding_model.dimension,
        completed_batches=completed_batches,
    )


def _write_json_atomically(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _cosine_scores(
    query: tuple[float, ...], vectors: tuple[tuple[float, ...], ...]
) -> list[float]:
    query_norm = math.sqrt(sum(value * value for value in query))
    if query_norm == 0:
        raise OllamaEmbeddingError("试查询返回了零向量")
    scores: list[float] = []
    for vector in vectors:
        vector_norm = math.sqrt(sum(value * value for value in vector))
        scores.append(
            sum(left * right for left, right in zip(query, vector, strict=True))
            / (query_norm * vector_norm)
        )
    return scores


def build_temporary_vector_index(
    chunks: tuple[DocumentChunk, ...] | list[DocumentChunk],
    embeddings: DocumentEmbeddingResult,
    embedding_model: OllamaEmbeddingModel,
    temporary_dir: Path | None = None,
    *,
    config: IngestConfig | None = None,
) -> TemporaryIndexBuildResult:
    """Build vector and keyword indexes from one chunk snapshot and probe them."""
    ordered_chunks = tuple(chunks)
    chunk_ids = tuple(chunk.chunk_id or "" for chunk in ordered_chunks)
    if any(not chunk_id for chunk_id in chunk_ids):
        raise ValueError("所有索引片段都必须具有 chunk_id")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("索引片段的 chunk_id 必须唯一")
    if embeddings.chunk_ids != chunk_ids:
        raise ValueError("向量与片段 ID 或顺序不一致")
    if len(embeddings.vectors) != len(ordered_chunks):
        raise ValueError("向量数量与片段数量不一致")
    if embeddings.dimension != embedding_model.dimension or embeddings.dimension <= 0:
        raise ValueError("向量维度与当前 Embedding 模型不一致")

    validated_vectors = _validate_embedding_batch(
        list(embeddings.vectors),
        expected_count=len(ordered_chunks),
        expected_dimension=embeddings.dimension,
        chunk_ids=chunk_ids,
    )
    chunk_by_id = dict(zip(chunk_ids, ordered_chunks, strict=True))

    if temporary_dir is None:
        config = config or load_ingest_config()
        config.index_dir.mkdir(parents=True, exist_ok=True)
        build_dir = Path(tempfile.mkdtemp(prefix=".build-", dir=config.index_dir))
    else:
        build_dir = Path(temporary_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

    postings: dict[str, list[str]] = {}
    for chunk in ordered_chunks:
        chunk_id = chunk.chunk_id or ""
        for token in extract_keywords(chunk.retrieval_text):
            postings.setdefault(token, []).append(chunk_id)

    vector_path = build_dir / "vectors.json"
    keyword_path = build_dir / "keywords.json"
    metadata_path = build_dir / "index_metadata.json"
    _write_json_atomically(
        vector_path,
        {
            "format": "flat-cosine-v1",
            "dimension": embeddings.dimension,
            "chunk_ids": chunk_ids,
            "vectors": validated_vectors,
        },
    )
    _write_json_atomically(
        keyword_path,
        {
            "format": "inverted-keyword-v1",
            "chunk_ids": chunk_ids,
            "postings": postings,
        },
    )
    _write_json_atomically(
        metadata_path,
        {
            "format": "knowledge-index-v1",
            "chunk_count": len(ordered_chunks),
            "embedding": {
                "provider": embedding_model.provider,
                "model": embedding_model.model,
                "digest": embedding_model.digest,
                "dimension": embedding_model.dimension,
                "document_strategy": "retrieval_text-v1",
                "query_strategy": "adapter-embed_query-v1",
            },
            "chunking": (
                {"size": config.chunk_size, "overlap": config.chunk_overlap}
                if config is not None
                else None
            ),
            "sources": {
                chunk_id: {
                    "source_path": chunk.source_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content_hash": chunk.content_hash,
                }
                for chunk_id, chunk in chunk_by_id.items()
            },
        },
    )

    # Validate the persisted ID associations, not only the in-memory inputs.
    with vector_path.open("r", encoding="utf-8") as source:
        stored_vectors = json.load(source)
    with keyword_path.open("r", encoding="utf-8") as source:
        stored_keywords = json.load(source)
    stored_ids = tuple(stored_vectors.get("chunk_ids", ()))
    if stored_ids != chunk_ids or tuple(stored_keywords.get("chunk_ids", ())) != chunk_ids:
        raise ValueError("临时索引中的片段关联不一致")
    if any(chunk_id not in chunk_by_id for chunk_id in stored_ids):
        raise ValueError("临时索引结果无法关联到原文")

    validation_chunk_id: str | None = None
    if ordered_chunks:
        try:
            raw_query = embedding_model.adapter.embed_query(ordered_chunks[0].retrieval_text)
        except Exception as exc:
            raise OllamaEmbeddingError(f"Ollama 试查询向量化失败: {exc}") from exc
        query_vector = _validate_embedding_batch(
            [raw_query],
            expected_count=1,
            expected_dimension=embeddings.dimension,
            chunk_ids=("<validation-query>",),
        )[0]
        scores = _cosine_scores(query_vector, validated_vectors)
        best_index = max(range(len(scores)), key=scores.__getitem__)
        validation_chunk_id = stored_ids[best_index]
        matched_chunk = chunk_by_id.get(validation_chunk_id)
        if matched_chunk is None or not matched_chunk.source_path:
            raise ValueError("试查询结果无法关联到有效原文来源")

    return TemporaryIndexBuildResult(
        index_dir=build_dir,
        vector_index_path=vector_path,
        keyword_index_path=keyword_path,
        metadata_path=metadata_path,
        chunk_count=len(ordered_chunks),
        dimension=embeddings.dimension,
        validation_chunk_id=validation_chunk_id,
    )


build_temporary_indexes = build_temporary_vector_index

@dataclass(frozen=True, slots=True)
class PublishedIndexResult:
    """Paths belonging to one completely published index generation."""

    index_dir: Path
    chunks_path: Path
    vector_index_path: Path
    keyword_index_path: Path
    metadata_path: Path
    manifest_path: Path
    chunk_count: int


def _manifest_files(
    docs_dir: Path,
    file_chunk_ids: dict[str, tuple[str, ...]],
) -> list[dict[str, object]]:
    """Snapshot every source file and fail if it changed or became unreadable."""
    records: list[dict[str, object]] = []
    for path in scan_markdown_files(docs_dir):
        document = read_markdown_file(path)
        identity = get_document_identity(document, docs_dir)
        records.append(
            {
                "path": identity.relative_path,
                "content_hash": identity.content_hash,
                "chunk_ids": file_chunk_ids.get(identity.relative_path, ()),
            }
        )
    return records


def publish_complete_index(
    first_stage: FirstStageResult,
    temporary_index: TemporaryIndexBuildResult,
    embedding_model: OllamaEmbeddingModel,
    config: IngestConfig | None = None,
    *,
    model_identity_checker: Callable[..., OllamaEmbeddingModel] | None = None,
) -> PublishedIndexResult:
    """Create a manifest and transactionally publish one complete index batch."""
    config = config or load_ingest_config()
    if first_stage.failures:
        raise ValueError("文档读取存在失败，不能发布索引")
    if temporary_index.chunk_count != len(first_stage.chunks):
        raise ValueError("临时索引与片段数量不一致")
    if temporary_index.dimension != embedding_model.dimension:
        raise ValueError("临时索引与 Embedding 维度不一致")

    build_dir = temporary_index.index_dir.resolve()
    index_dir = config.index_dir.resolve()
    if build_dir == index_dir:
        raise ValueError("临时构建目录不能是当前索引目录")
    expected_artifacts = (
        temporary_index.vector_index_path,
        temporary_index.keyword_index_path,
        temporary_index.metadata_path,
    )
    if any(path.resolve().parent != build_dir or not path.is_file() for path in expected_artifacts):
        raise ValueError("临时索引产物不完整或不属于同一构建目录")

    chunks_path = build_dir / "chunks.jsonl"
    validate_and_export_chunks(first_stage.chunks, config.docs_dir, chunks_path)
    files = _manifest_files(config.docs_dir, first_stage.file_chunk_ids)
    if len(files) != first_stage.statistics.discovered_files:
        raise ValueError("文档清单在分块完成后发生了变化，请重新构建")
    chunk_hashes = {chunk.source_path: chunk.content_hash for chunk in first_stage.chunks}
    if any(
        record["path"] in chunk_hashes
        and record["content_hash"] != chunk_hashes[record["path"]]
        for record in files
    ):
        raise ValueError("文档在分块完成后发生了变化，请重新构建")

    checker = model_identity_checker or check_ollama_embedding_model
    current_model = checker(config=config, adapter=embedding_model.adapter)
    if (
        current_model.provider != embedding_model.provider
        or current_model.model != embedding_model.model
        or current_model.digest != embedding_model.digest
        or current_model.dimension != embedding_model.dimension
    ):
        raise OllamaEmbeddingError("Embedding 模型在构建期间发生了变化")

    manifest_path = build_dir / "manifest.json"
    _write_json_atomically(
        manifest_path,
        {
            "schema_version": 1,
            "status": "ready",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "docs_dir": os.fspath(config.docs_dir),
            "chunking": {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            },
            "embedding": {
                "provider": embedding_model.provider,
                "model": embedding_model.model,
                "digest": embedding_model.digest,
                "dimension": embedding_model.dimension,
                "document_text": "heading_path_then_verbatim_source",
                "query_text": "verbatim",
                "options": {},
            },
            "files": files,
            "chunk_count": len(first_stage.chunks),
            "artifacts": {
                "chunks": "chunks.jsonl",
                "vectors": temporary_index.vector_index_path.name,
                "keywords": temporary_index.keyword_index_path.name,
                "metadata": temporary_index.metadata_path.name,
            },
        },
    )

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix=f".{index_dir.name}.backup-", dir=index_dir.parent))
    backup_dir.rmdir()
    previous_exists = index_dir.exists()
    moved_build_dir = build_dir
    try:
        if previous_exists:
            os.replace(index_dir, backup_dir)
            try:
                moved_build_dir = backup_dir / build_dir.relative_to(index_dir)
            except ValueError:
                pass
        os.replace(moved_build_dir, index_dir)
    except BaseException:
        if previous_exists and backup_dir.exists() and not index_dir.exists():
            os.replace(backup_dir, index_dir)
        raise
    finally:
        if backup_dir.exists() and index_dir.exists():
            try:
                shutil.rmtree(backup_dir)
            except OSError:
                _LOGGER.warning("无法清理索引发布备份: %s", backup_dir)

    return PublishedIndexResult(
        index_dir=index_dir,
        chunks_path=index_dir / chunks_path.name,
        vector_index_path=index_dir / temporary_index.vector_index_path.name,
        keyword_index_path=index_dir / temporary_index.keyword_index_path.name,
        metadata_path=index_dir / temporary_index.metadata_path.name,
        manifest_path=index_dir / manifest_path.name,
        chunk_count=len(first_stage.chunks),
    )


save_manifest_and_publish = publish_complete_index
publish_index = publish_complete_index


# 第三阶段：支持文档新增、修改和删除。

_DOCUMENT_EMBEDDING_STRATEGY = "heading_path_then_verbatim_source"
_QUERY_EMBEDDING_STRATEGY = "verbatim"


@dataclass(frozen=True, slots=True)
class ManifestFileRecord:
    """One validated file entry from a current or previously published manifest."""

    path: str
    content_hash: str
    chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ManifestComparison:
    """File-level changes and compatibility decisions for an incremental update."""

    current_files: tuple[ManifestFileRecord, ...]
    previous_files: tuple[ManifestFileRecord, ...]
    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]
    unchanged: tuple[str, ...]
    requires_full_rebuild: bool
    requires_query_validation: bool
    rebuild_reasons: tuple[str, ...]

    @property
    def has_file_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    # More explicit spellings are useful to callers and keep the compact names
    # pleasant when reporting the comparison on the command line.
    @property
    def added_files(self) -> tuple[str, ...]:
        return self.added

    @property
    def modified_files(self) -> tuple[str, ...]:
        return self.modified

    @property
    def deleted_files(self) -> tuple[str, ...]:
        return self.deleted

    @property
    def unchanged_files(self) -> tuple[str, ...]:
        return self.unchanged


def _read_previous_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取上一版 manifest: {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"上一版 manifest 格式无效: {path}")
    return manifest


def _manifest_file_records(value: object, *, source: str) -> tuple[ManifestFileRecord, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{source} 的 files 必须是列表")

    records: list[ManifestFileRecord] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{source} 包含无效的文件记录")
        path = item.get("path")
        content_hash = item.get("content_hash")
        raw_chunk_ids = item.get("chunk_ids", ())
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(content_hash, str)
            or not content_hash
            or not isinstance(raw_chunk_ids, (list, tuple))
            or not all(isinstance(chunk_id, str) for chunk_id in raw_chunk_ids)
        ):
            raise ValueError(f"{source} 包含无效的文件记录")
        if path in seen:
            raise ValueError(f"{source} 包含重复文件: {path}")
        seen.add(path)
        records.append(ManifestFileRecord(path, content_hash, tuple(raw_chunk_ids)))
    records.sort(key=lambda record: record.path)
    return tuple(records)


def _snapshot_manifest_files(docs_dir: Path) -> tuple[ManifestFileRecord, ...]:
    """Read a complete Markdown snapshot, propagating every scan/read failure."""
    docs_dir = Path(docs_dir).resolve()
    paths: list[Path] = []
    directories = [docs_dir]
    while directories:
        directory = directories.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        directories.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".md"):
                        paths.append(Path(entry.path))
        except OSError as exc:
            raise DocumentReadError(f"Failed to scan Markdown directory {directory}: {exc}") from exc

    paths.sort(key=lambda path: path.relative_to(docs_dir).as_posix())
    records: list[ManifestFileRecord] = []
    for path in paths:
        document = read_markdown_file(path)
        identity = get_document_identity(document, docs_dir)
        records.append(ManifestFileRecord(identity.relative_path, identity.content_hash))
    return tuple(records)


def compare_manifest(
    config: IngestConfig,
    embedding_model: OllamaEmbeddingModel,
    manifest_path: Path | None = None,
    *,
    document_strategy: str = _DOCUMENT_EMBEDDING_STRATEGY,
    query_strategy: str = _QUERY_EMBEDDING_STRATEGY,
) -> ManifestComparison:
    """Compare a consistent source snapshot with the last published manifest."""
    current_files = _snapshot_manifest_files(config.docs_dir)
    manifest_path = Path(manifest_path) if manifest_path is not None else config.index_dir / "manifest.json"

    if manifest_path.exists():
        previous_manifest = _read_previous_manifest(manifest_path)
        previous_files = _manifest_file_records(
            previous_manifest.get("files"), source="上一版 manifest"
        )
    else:
        previous_manifest = {}
        previous_files = ()

    current_by_path = {record.path: record for record in current_files}
    previous_by_path = {record.path: record for record in previous_files}
    current_paths = current_by_path.keys()
    previous_paths = previous_by_path.keys()
    added = tuple(sorted(current_paths - previous_paths))
    deleted = tuple(sorted(previous_paths - current_paths))
    common = current_paths & previous_paths
    modified = tuple(
        sorted(
            path
            for path in common
            if current_by_path[path].content_hash != previous_by_path[path].content_hash
        )
    )
    modified_set = set(modified)
    unchanged = tuple(sorted(path for path in common if path not in modified_set))

    rebuild_reasons: list[str] = []
    previous_chunking = previous_manifest.get("chunking")
    previous_embedding = previous_manifest.get("embedding")
    if not previous_manifest:
        rebuild_reasons.append("missing_manifest")
    else:
        if not isinstance(previous_chunking, dict) or (
            previous_chunking.get("chunk_size") != config.chunk_size
            or previous_chunking.get("chunk_overlap") != config.chunk_overlap
        ):
            rebuild_reasons.append("chunking_changed")
        if not isinstance(previous_embedding, dict):
            rebuild_reasons.append("embedding_metadata_changed")
        else:
            if previous_embedding.get("provider") != embedding_model.provider:
                rebuild_reasons.append("embedding_provider_changed")
            if previous_embedding.get("digest") != embedding_model.digest:
                rebuild_reasons.append("embedding_digest_changed")
            if previous_embedding.get("dimension") != embedding_model.dimension:
                rebuild_reasons.append("embedding_dimension_changed")
            if previous_embedding.get("document_text") != document_strategy:
                rebuild_reasons.append("document_strategy_changed")

    requires_query_validation = bool(previous_manifest) and (
        not isinstance(previous_embedding, dict)
        or previous_embedding.get("query_text") != query_strategy
    )
    return ManifestComparison(
        current_files=current_files,
        previous_files=previous_files,
        added=added,
        modified=modified,
        deleted=deleted,
        unchanged=unchanged,
        requires_full_rebuild=bool(rebuild_reasons),
        requires_query_validation=requires_query_validation,
        rebuild_reasons=tuple(rebuild_reasons),
    )


compare_current_files_to_manifest = compare_manifest
compare_with_previous_manifest = compare_manifest

@dataclass(frozen=True, slots=True)
class IncrementalUpdateResult:
    """Outcome of one atomic incremental index update."""

    comparison: ManifestComparison
    published: PublishedIndexResult
    reused_chunks: int
    embedded_chunks: int
    removed_chunks: int
    full_rebuild: bool


def _load_published_chunks(path: Path) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("record is not an object")
                heading_path = item.get("heading_path")
                if not isinstance(heading_path, (list, tuple)) or not all(
                    isinstance(value, str) for value in heading_path
                ):
                    raise ValueError("invalid heading_path")
                chunk = DocumentChunk(
                    text=item["text"],
                    source_path=item["source_path"],
                    heading_path=tuple(heading_path),
                    start_line=item["start_line"],
                    end_line=item["end_line"],
                    content_hash=item["content_hash"],
                    content_type=item.get("content_type", "markdown"),
                    chunk_id=item["chunk_id"],
                )
                if (
                    not isinstance(chunk.text, str)
                    or not isinstance(chunk.source_path, str)
                    or not isinstance(chunk.start_line, int)
                    or isinstance(chunk.start_line, bool)
                    or not isinstance(chunk.end_line, int)
                    or isinstance(chunk.end_line, bool)
                    or not isinstance(chunk.content_hash, str)
                    or chunk.content_type != "markdown"
                    or not isinstance(chunk.chunk_id, str)
                    or not chunk.chunk_id
                ):
                    raise ValueError("invalid chunk fields")
                chunks.append(chunk)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"无法读取上一版片段: {path}:{line_number if 'line_number' in locals() else 0}: {exc}") from exc
    return tuple(chunks)


def _artifact_path(index_dir: Path, manifest: dict[str, Any], key: str) -> Path:
    artifacts = manifest.get("artifacts")
    value = artifacts.get(key) if isinstance(artifacts, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"上一版 manifest 缺少产物: {key}")
    path = (index_dir / value).resolve()
    if path.parent != index_dir.resolve() or not path.is_file():
        raise ValueError(f"上一版产物路径无效: {key}")
    return path


def _load_published_embeddings(
    path: Path, expected_dimension: int
) -> DocumentEmbeddingResult:
    try:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        if not isinstance(payload, dict):
            raise ValueError("向量产物不是 JSON 对象")
        chunk_ids = payload.get("chunk_ids")
        vectors = payload.get("vectors")
        if (
            payload.get("dimension") != expected_dimension
            or not isinstance(chunk_ids, list)
            or not all(isinstance(chunk_id, str) and chunk_id for chunk_id in chunk_ids)
        ):
            raise ValueError("向量产物元数据无效")
        ids = tuple(chunk_ids)
        validated = _validate_embedding_batch(
            vectors,
            expected_count=len(ids),
            expected_dimension=expected_dimension,
            chunk_ids=ids,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"无法读取上一版向量: {path}: {exc}") from exc
    return DocumentEmbeddingResult(ids, validated, expected_dimension, 0)


def _chunk_changed_files(
    config: IngestConfig, paths: tuple[str, ...]
) -> tuple[tuple[DocumentChunk, ...], int, int]:
    chunks: list[DocumentChunk] = []
    section_count = 0
    oversized_count = 0
    docs_root = config.docs_dir.resolve()
    for relative_path in paths:
        document = read_markdown_file(docs_root / Path(relative_path))
        identity = get_document_identity(document, docs_root)
        if identity.relative_path != relative_path:
            raise ValueError(f"文档路径在更新期间发生了变化: {relative_path}")
        if not document.text.strip():
            continue
        parsed = parse_markdown_blocks(document)
        sections = organize_markdown_sections(document, parsed)
        chunking = split_markdown_sections(sections, config.chunk_size, config.chunk_overlap)
        _verify_document_coverage(document, chunking.chunks)
        document_chunks = add_chunk_metadata(document, chunking.chunks, docs_root)
        assigned = assign_chunk_ids(document_chunks).chunks
        for chunk in assigned:
            _verify_chunk_citation(chunk, document)
        chunks.extend(assigned)
        section_count += len(sections)
        oversized_count += len(chunking.oversized_blocks)
    return tuple(chunks), section_count, oversized_count


def _published_result_from_manifest(
    config: IngestConfig, manifest: dict[str, Any]
) -> PublishedIndexResult:
    index_dir = config.index_dir.resolve()
    return PublishedIndexResult(
        index_dir=index_dir,
        chunks_path=_artifact_path(index_dir, manifest, "chunks"),
        vector_index_path=_artifact_path(index_dir, manifest, "vectors"),
        keyword_index_path=_artifact_path(index_dir, manifest, "keywords"),
        metadata_path=_artifact_path(index_dir, manifest, "metadata"),
        manifest_path=index_dir / "manifest.json",
        chunk_count=manifest.get("chunk_count", 0),
    )


def update_index_incrementally(
    config: IngestConfig,
    embedding_model: OllamaEmbeddingModel,
    comparison: ManifestComparison | None = None,
    *,
    refresh_callback: Callable[[PublishedIndexResult], None] | None = None,
    model_identity_checker: Callable[..., OllamaEmbeddingModel] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> IncrementalUpdateResult:
    """Reuse unchanged records and atomically publish a validated new snapshot."""
    comparison = comparison or compare_manifest(config, embedding_model)
    manifest_path = config.index_dir / "manifest.json"
    previous_manifest = (
        _read_previous_manifest(manifest_path) if manifest_path.is_file() else {}
    )

    if (
        not comparison.requires_full_rebuild
        and not comparison.has_file_changes
        and not comparison.requires_query_validation
    ):
        published = _published_result_from_manifest(config, previous_manifest)
        return IncrementalUpdateResult(
            comparison, published, published.chunk_count, 0, 0, False
        )

    old_chunks: tuple[DocumentChunk, ...] = ()
    old_vectors_by_id: dict[str, tuple[float, ...]] = {}
    previous_ids: set[str] = set()
    if not comparison.requires_full_rebuild:
        chunks_path = _artifact_path(config.index_dir, previous_manifest, "chunks")
        vectors_path = _artifact_path(config.index_dir, previous_manifest, "vectors")
        old_chunks = _load_published_chunks(chunks_path)
        old_embeddings = _load_published_embeddings(vectors_path, embedding_model.dimension)
        old_ids = tuple(chunk.chunk_id or "" for chunk in old_chunks)
        if old_ids != old_embeddings.chunk_ids or len(set(old_ids)) != len(old_ids):
            raise ValueError("上一版片段与向量关联不一致")
        old_vectors_by_id = old_embeddings.by_chunk_id
        previous_ids = set(old_ids)

    if comparison.requires_full_rebuild:
        changed_paths = tuple(record.path for record in comparison.current_files)
        reused_by_path: dict[str, tuple[DocumentChunk, ...]] = {}
    else:
        changed_paths = tuple(sorted((*comparison.added, *comparison.modified)))
        unchanged = set(comparison.unchanged)
        current_hashes = {record.path: record.content_hash for record in comparison.current_files}
        grouped: dict[str, list[DocumentChunk]] = {path: [] for path in unchanged}
        for chunk in old_chunks:
            if chunk.source_path in unchanged:
                if chunk.content_hash != current_hashes[chunk.source_path]:
                    raise ValueError(f"上一版片段指纹与未变文档不一致: {chunk.source_path}")
                grouped[chunk.source_path].append(chunk)
        previous_by_path = {record.path: record for record in comparison.previous_files}
        for path in unchanged:
            actual = tuple(chunk.chunk_id or "" for chunk in grouped[path])
            if actual != previous_by_path[path].chunk_ids:
                raise ValueError(f"上一版 manifest 与片段关联不一致: {path}")
        reused_by_path = {path: tuple(items) for path, items in grouped.items()}

    changed_chunks, section_count, oversized_count = _chunk_changed_files(
        config, changed_paths
    )
    changed_by_path: dict[str, list[DocumentChunk]] = {path: [] for path in changed_paths}
    for chunk in changed_chunks:
        changed_by_path[chunk.source_path].append(chunk)

    ordered_chunks = tuple(
        chunk
        for record in comparison.current_files
        for chunk in (
            changed_by_path.get(record.path, ())
            if record.path in changed_by_path
            else reused_by_path.get(record.path, ())
        )
    )
    ordered_ids = tuple(chunk.chunk_id or "" for chunk in ordered_chunks)
    if any(not chunk_id for chunk_id in ordered_ids) or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("更新后的片段 ID 缺失或重复")

    generated = generate_document_embeddings(
        changed_chunks, embedding_model, config, progress=progress
    )
    generated_by_id = generated.by_chunk_id
    vectors = tuple(
        generated_by_id[chunk_id]
        if chunk_id in generated_by_id
        else old_vectors_by_id[chunk_id]
        for chunk_id in ordered_ids
    )
    embeddings = DocumentEmbeddingResult(
        ordered_ids, vectors, embedding_model.dimension, generated.completed_batches
    )

    ids_by_path: dict[str, list[str]] = {
        record.path: [] for record in comparison.current_files
    }
    for chunk in ordered_chunks:
        ids_by_path[chunk.source_path].append(chunk.chunk_id or "")
    file_chunk_ids = {path: tuple(ids) for path, ids in ids_by_path.items()}
    blank_files = sum(not file_chunk_ids[record.path] for record in comparison.current_files)
    first_stage = FirstStageResult(
        statistics=FirstStageStatistics(
            discovered_files=len(comparison.current_files),
            read_files=len(comparison.current_files),
            blank_files=blank_files,
            failed_files=0,
            sections=section_count,
            chunks=len(ordered_chunks),
            oversized_blocks=oversized_count,
        ),
        chunks=ordered_chunks,
        file_chunk_ids=file_chunk_ids,
        failures=(),
        spot_checks=_select_first_stage_spot_checks(ordered_chunks),
        output_path=None,
    )

    config.index_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(
        tempfile.mkdtemp(prefix=f".{config.index_dir.name}.update-", dir=config.index_dir.parent)
    )
    try:
        temporary_index = build_temporary_vector_index(
            ordered_chunks, embeddings, embedding_model, build_dir, config=config
        )
        published = publish_complete_index(
            first_stage,
            temporary_index,
            embedding_model,
            config,
            model_identity_checker=model_identity_checker,
        )
    except BaseException:
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)
        raise

    if refresh_callback is not None:
        refresh_callback(published)
    reused_ids = set(ordered_ids) & previous_ids - set(generated.chunk_ids)
    removed_ids = previous_ids - set(ordered_ids)
    return IncrementalUpdateResult(
        comparison=comparison,
        published=published,
        reused_chunks=len(reused_ids),
        embedded_chunks=len(generated.chunk_ids),
        removed_chunks=len(removed_ids),
        full_rebuild=comparison.requires_full_rebuild,
    )


incremental_update = update_index_incrementally
update_incremental_index = update_index_incrementally

@dataclass(frozen=True, slots=True)
class IngestRunResult:
    """Structured result returned by the ingestion entry points."""

    operation: Literal["chunk", "build", "update"]
    discovered_files: int
    read_files: int
    chunk_count: int
    changed_file_count: int
    failures: tuple[DocumentReadFailure, ...]
    output_paths: tuple[Path, ...]
    full_rebuild: bool = False
    reused_chunks: int = 0
    embedded_chunks: int = 0
    removed_chunks: int = 0

    @property
    def succeeded(self) -> bool:
        return not self.failures

    @property
    def file_count(self) -> int:
        return self.discovered_files

    @property
    def output_path(self) -> Path | None:
        return self.output_paths[0] if self.output_paths else None


def _published_output_paths(published: PublishedIndexResult) -> tuple[Path, ...]:
    return (
        published.index_dir,
        published.chunks_path,
        published.vector_index_path,
        published.keyword_index_path,
        published.metadata_path,
        published.manifest_path,
    )


def chunk_documents(
    config: IngestConfig | None = None,
    output_path: Path | None = None,
) -> IngestRunResult:
    """Read, validate, and chunk documents without creating vector indexes."""
    config = config or load_ingest_config()
    stage = run_first_stage(config, output_path)
    return IngestRunResult(
        operation="chunk",
        discovered_files=stage.statistics.discovered_files,
        read_files=stage.statistics.read_files,
        chunk_count=stage.statistics.chunks,
        changed_file_count=0,
        failures=stage.failures,
        output_paths=(stage.output_path,) if stage.output_path is not None else (),
    )


def build_index(
    config: IngestConfig | None = None,
    *,
    embedding_model: OllamaEmbeddingModel | None = None,
    model_identity_checker: Callable[..., OllamaEmbeddingModel] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> IngestRunResult:
    """Build and atomically publish a complete index generation."""
    config = config or load_ingest_config()
    config.index_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(
        tempfile.mkdtemp(prefix=f".{config.index_dir.name}.build-", dir=config.index_dir.parent)
    )
    try:
        stage = run_first_stage(config, build_dir / "chunks.jsonl")
        if stage.failures:
            return IngestRunResult(
                operation="build",
                discovered_files=stage.statistics.discovered_files,
                read_files=stage.statistics.read_files,
                chunk_count=stage.statistics.chunks,
                changed_file_count=stage.statistics.read_files,
                failures=stage.failures,
                output_paths=(),
                full_rebuild=True,
            )

        model = embedding_model or check_ollama_embedding_model(config)
        embeddings = generate_document_embeddings(
            stage.chunks, model, config, progress=progress
        )
        temporary_index = build_temporary_vector_index(
            stage.chunks, embeddings, model, build_dir, config=config
        )
        published = publish_complete_index(
            stage,
            temporary_index,
            model,
            config,
            model_identity_checker=model_identity_checker,
        )
        return IngestRunResult(
            operation="build",
            discovered_files=stage.statistics.discovered_files,
            read_files=stage.statistics.read_files,
            chunk_count=published.chunk_count,
            changed_file_count=stage.statistics.read_files,
            failures=(),
            output_paths=_published_output_paths(published),
            full_rebuild=True,
            embedded_chunks=len(embeddings.chunk_ids),
        )
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)


def update_index(
    config: IngestConfig | None = None,
    *,
    embedding_model: OllamaEmbeddingModel | None = None,
    refresh_callback: Callable[[PublishedIndexResult], None] | None = None,
    model_identity_checker: Callable[..., OllamaEmbeddingModel] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> IngestRunResult:
    """Incrementally update and atomically publish the current index."""
    config = config or load_ingest_config()
    model = embedding_model or check_ollama_embedding_model(config)
    comparison = compare_manifest(config, model)
    updated = update_index_incrementally(
        config,
        model,
        comparison,
        refresh_callback=refresh_callback,
        model_identity_checker=model_identity_checker,
        progress=progress,
    )
    changed_file_count = len(comparison.added) + len(comparison.modified) + len(
        comparison.deleted
    )
    return IngestRunResult(
        operation="update",
        discovered_files=len(comparison.current_files),
        read_files=len(comparison.current_files),
        chunk_count=updated.published.chunk_count,
        changed_file_count=changed_file_count,
        failures=(),
        output_paths=_published_output_paths(updated.published),
        full_rebuild=updated.full_rebuild,
        reused_chunks=updated.reused_chunks,
        embedded_chunks=updated.embedded_chunks,
        removed_chunks=updated.removed_chunks,
    )


read_and_chunk = chunk_documents
build_knowledge_index = build_index
update_knowledge_index = update_index
