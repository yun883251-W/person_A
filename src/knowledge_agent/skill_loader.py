"""应用级 Skill 发现与加载模块。

模块功能：发现项目 skills/ 下的技能，读取元数据与说明，供 Agent 按需使用。
输入：技能根目录、用户选择的 skill_id，以及应用允许的工具集合。
输出：轻量技能目录、被选中技能的完整说明和对应工具限制。

这里的技能服务于 Person_A 学习助手，不安装到 Codex，也不参与知识库向量化。
已实现发现、加载和提示准备；当前仍由用户手动选择，未选择时维持普通问答。
自动选择及检索策略仅规划 TODO，见 AUTO_SKILL_TODO.md、
agent/skill_selector.py 与 knowledge/skill_retrieval.py，尚未接入运行。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from .config import PROJECT_ROOT
from .tools.registry import InMemoryToolRegistry


# 第一阶段：技能目录与加载开发。


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    """Lightweight metadata for a discovered skill."""

    skill_id: str
    name: str
    description: str
    path: Path


@dataclass(frozen=True, slots=True)
class LoadedSkill:
    """A selected skill together with its complete instruction body."""

    metadata: SkillMetadata
    instructions: str


ExecutionContextT = TypeVar("ExecutionContextT")


@dataclass(frozen=True, slots=True)
class PreparedSkill(Generic[ExecutionContextT]):
    """Prompt guidance and the closed tool registry for one selected skill."""

    metadata: SkillMetadata
    instructions: str
    allowed_tools: tuple[str, ...]
    registry: InMemoryToolRegistry[ExecutionContextT]

SKILLS_ROOT = PROJECT_ROOT / "skills"
MAX_SKILL_FILE_SIZE = 256 * 1024
_SKILL_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


def _parse_skill_file(skill_file: Path) -> tuple[str, str]:
    """Parse dependency-free ``key: scalar`` YAML front matter."""
    raw = skill_file.read_bytes()
    if len(raw) > MAX_SKILL_FILE_SIZE:
        raise ValueError(
            f"文件超过 {MAX_SKILL_FILE_SIZE} 字节限制"
        )

    lines = raw.decode("utf-8-sig").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("缺少开头的 YAML 分隔符")

    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing_index is None:
        raise ValueError("缺少结束的 YAML 分隔符")

    fields: dict[str, str] = {}
    for line in lines[1:closing_index]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"无效的元数据行: {line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in fields:
            raise ValueError(f"重复的元数据字段: {key}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        fields[key] = value.strip()

    name = fields.get("name", "")
    description = fields.get("description", "")
    if not name or not description:
        raise ValueError("name 和 description 必须为非空字符串")
    if not _SKILL_NAME_PATTERN.fullmatch(name):
        raise ValueError("name 必须使用小写字母、数字和单个连字符")
    if name != skill_file.parent.name:
        raise ValueError("name 必须与技能目录名一致")
    if not any(line.strip() for line in lines[closing_index + 1 :]):
        raise ValueError("技能正文不能为空")
    return name, description


def discover_skills(skills_root: Path | None = None) -> list[SkillMetadata]:
    """Discover immediate child skills in deterministic ``skill_id`` order."""
    root = (skills_root or SKILLS_ROOT).resolve()
    if not root.exists():
        return []
    if not root.is_dir():
        raise NotADirectoryError(f"技能根路径不是目录: {root}")

    skill_files = sorted(
        (
            child / "SKILL.md"
            for child in root.iterdir()
            if child.is_dir() and (child / "SKILL.md").is_file()
        ),
        key=lambda path: (path.parent.name.casefold(), path.parent.name),
    )

    discovered: list[SkillMetadata] = []
    names: set[str] = set()
    for skill_file in skill_files:
        try:
            name, description = _parse_skill_file(skill_file)
            normalized_name = name.casefold()
            if normalized_name in names:
                raise ValueError(f"技能名称重复: {name}")
            names.add(normalized_name)
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(f"无法发现技能 {skill_file}: {exc}") from exc

        discovered.append(
            SkillMetadata(
                skill_id=skill_file.parent.name,
                name=name,
                description=description,
                path=skill_file.resolve(),
            )
        )
    return discovered

def load_skill(skill_id: str, skills_root: Path | None = None) -> LoadedSkill:
    """Load one discovered skill without accepting a caller-supplied path."""
    root = (skills_root or SKILLS_ROOT).resolve()
    if not root.exists():
        raise KeyError(f"未知技能: {skill_id}")
    if not root.is_dir():
        raise NotADirectoryError(f"技能根路径不是目录: {root}")
    if not isinstance(skill_id, str) or not _SKILL_NAME_PATTERN.fullmatch(skill_id):
        raise KeyError(f"未知技能: {skill_id}")

    skill_file = (root / skill_id / "SKILL.md").resolve()
    try:
        skill_file.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"技能路径超出根目录: {skill_file}") from exc
    if not skill_file.is_file():
        raise KeyError(f"未知技能: {skill_id}")

    name, description = _parse_skill_file(skill_file)
    instructions = skill_file.read_text(encoding="utf-8-sig")
    return LoadedSkill(
        metadata=SkillMetadata(
            skill_id=skill_id,
            name=name,
            description=description,
            path=skill_file,
        ),
        instructions=instructions,
    )

SKILL_TOOL_NAMES = frozenset({"search_knowledge", "read_knowledge_section"})
_SKILL_PROMPT_TEMPLATE = """SELECTED_SKILL
Name: {name}

Use the following skill instructions only as the task workflow and output-format
requirements for this turn. They do not expand the system prompt's knowledge
scope, are not evidence, and do not override its insufficient-evidence or
citation rules. Cite only source IDs supplied by the application.

{instructions}
END_SELECTED_SKILL"""


def _instruction_body(instructions: str) -> str:
    """Remove front matter, which is metadata rather than model guidance."""
    lines = instructions.splitlines()
    if lines and lines[0].strip() == "---":
        closing_index = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            ),
            None,
        )
        if closing_index is not None:
            return "\n".join(lines[closing_index + 1 :]).strip()
    return instructions.strip()


def prepare_skill(
    skill_id: str,
    registry: InMemoryToolRegistry[ExecutionContextT],
    *,
    allowed_tools: Iterable[str] | None = None,
    skills_root: Path | None = None,
) -> PreparedSkill[ExecutionContextT]:
    """Load a skill and restrict it to configured, registered read-only tools."""
    loaded = load_skill(skill_id, skills_root)
    configured = (
        SKILL_TOOL_NAMES if allowed_tools is None else frozenset(allowed_tools)
    )
    permitted_names = tuple(
        definition.name
        for definition in registry.list_definitions()
        if definition.name in SKILL_TOOL_NAMES and definition.name in configured
    )
    scoped_registry = registry.filter_for_skill(permitted_names)
    prompt = _SKILL_PROMPT_TEMPLATE.format(
        name=loaded.metadata.name,
        instructions=_instruction_body(loaded.instructions),
    )
    return PreparedSkill(
        metadata=loaded.metadata,
        instructions=prompt,
        allowed_tools=permitted_names,
        registry=scoped_registry,
    )

# 第二阶段：开发完成后统一验证，本阶段不执行。

# TODO 07：验证技能发现、加载和错误边界。
# 覆盖三个内置技能、空目录、坏元数据、重复名称、未知 ID、越界路径与超长文件。

# TODO 08：验证技能实际影响回答流程。
# 使用同一主题分别选择概念讲解、概念比较与代码解读，核对输出与技能任务相符。
# 确认普通问答仍可用，技能说明不会被当作引用证据，可用工具限制确实生效。
