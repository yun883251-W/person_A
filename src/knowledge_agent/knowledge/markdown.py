"""Markdown 文档读取、结构解析和可追溯分块。

保留原文、行号和偏移，按标题与段落/代码围栏边界分块，并生成稳定片段 ID。
只依赖标准库和 DocumentChunk；不加载模型、不发布索引、不执行文档内容。
建库编排和 JSONL 导出仍由 ingest.py 负责。
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from ..schemas import DocumentChunk


def scan_markdown_files(docs_dir: Path) -> list[Path]:
    """按确定的路径顺序返回 *docs_dir* 下的 Markdown 文件。"""
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
    """一行源文本，包含其原始的换行符（如有）。"""

    number: int
    text: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """解码后的源文本，以及从 1 开始计数的源行映射。"""

    path: Path
    text: str
    lines: tuple[SourceLine, ...]

    @property
    def is_empty(self) -> bool:
        return not self.text


class DocumentReadError(RuntimeError):
    """源文档无法解码或读取时抛出。"""


def read_markdown_file(path: Path) -> SourceDocument:
    """读取 UTF-8 Markdown，不改变换行符或空行。"""
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
    """稳定的源文件身份及其解码内容的指纹。"""

    relative_path: str
    content_hash: str


def get_document_identity(document: SourceDocument, docs_dir: Path) -> DocumentIdentity:
    """返回规范化的文档目录相对路径和 SHA-256 内容哈希。"""
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
    """保留原文的 Markdown 文本块或围栏代码块。"""

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
    """识别 Markdown 块时发现的非致命问题。"""

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
    """以单次线性扫描把 *document* 拆分为文本块和围栏代码块。"""
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
    """Markdown 文档中按标题划分、保留原文的一部分。"""

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
    """使用 ATX 标题层级，把文档组织成精确的源区间。"""
    if not document.lines:
        return ()

    parsed = parsed or parse_markdown_blocks(document)
    headings: list[tuple[int, int, str]] = []
    append_heading = headings.append

    # 只检查文本块，因此代码围栏内形似标题的行
    # 仍然属于其所在的章节。
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
    """连续的源切片；长度以 Unicode 字符数计量。"""

    source_path: Path
    heading_path: tuple[str, ...]
    text: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class OversizedMarkdownBlock:
    """超过配置上限、无法再拆分的源块。"""

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
    """划分章节，但不拆散段落或围栏代码块。"""
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
            # 空行归属于前一个段落。开头的空行自成一个块，
            # 从而保证源文本划分无损。
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
    """按自然边界拆分章节，长度上限以字符数计。

    段落（包括 Markdown 表格和无围栏图示）与围栏代码是原子单元。
    重叠部分复用完整的尾部块，因此不会仅为凑准上限而切断源块。
    超过 ``chunk_size`` 的原子块会原样输出为一个片段，同时也在
    ``oversized_blocks`` 中返回。
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
            # 只复用完整的块，且至少前进一个块，
            # 避免超长段落导致循环停滞。
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
    """把源切片转换为可追溯的片段，且不改变其文本。"""
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
    """分配可复现的 ID，并按源文件分组。"""
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
