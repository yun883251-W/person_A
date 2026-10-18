"""引用渲染，以及历史回答独立的、不可引用的视图。

公开回答会为 UI 和持久化保留来源元数据。模型历史、计划依赖和记忆
不得重放这些渲染结果或轮次内的 ID。
本模块不校验、修复或放宽新模型回答中的引用。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from knowledge_agent.schemas import AnswerSource


SOURCE_TOKEN_RE = re.compile(r"\[(S[1-9]\d*)\]")
SOURCE_LIKE_TOKEN_RE = re.compile(r"\[S[^\]\r\n]*\]", re.IGNORECASE)
WRITTEN_CITATION_RE = re.compile(
    r"(?:[^\s【】]+\.md\s*[·|][^\r\n【】]*?(?:第\s*)?\d+\s*"
    r"(?:[-–—~～至]\s*\d+)?\s*行)|"
    r"(?:【[^\r\n【】]*\.md[^\r\n【】]*\d+\s*"
    r"(?:[-–—~～至]\s*\d+)?\s*行】)",
    re.IGNORECASE,
)
_HISTORICAL_SOURCE_TOKEN_RE = re.compile(r"\[S\d+\]", re.IGNORECASE)
# 仅用于历史的兜底规则，处理渲染标题本身包含【】的情况。
# 保持上方当前回答的校验模式不变。
_RENDERED_HISTORY_CITATION_RE = re.compile(
    r"【[^\r\n【】]*\.md · [^\r\n]*?原文第 \d+(?:-\d+)? 行】",
    re.IGNORECASE,
)


def format_source(source: AnswerSource) -> str:
    """仅依据索引来源元数据渲染公开引用。"""
    parts = [source.source_path]
    if source.heading_path:
        parts.append(" > ".join(source.heading_path))
    if source.start_line == source.end_line:
        parts.append(f"原文第 {source.start_line} 行")
    else:
        parts.append(f"原文第 {source.start_line}-{source.end_line} 行")
    return f"【{' · '.join(parts)}】"


def historical_answer_text(text: str, sources: Sequence[AnswerSource] = ()) -> str:
    """在截断背景或发送给模型之前移除旧引用。

    先精确匹配已知的渲染形式，包括不常见的路径/标题。该模式也覆盖
    缺少来源元数据的旧摘要和存储回答。
    绝不能把旧引用转换成 [S1]：ID 只属于当前证据。
    此投影不改变存储/展示的回答和来源对象。
    """
    cleaned = text
    for source in sources:
        cleaned = cleaned.replace(format_source(source), "")
    cleaned = _RENDERED_HISTORY_CITATION_RE.sub("", cleaned)
    cleaned = WRITTEN_CITATION_RE.sub("", cleaned)
    return _HISTORICAL_SOURCE_TOKEN_RE.sub("", cleaned).strip()
