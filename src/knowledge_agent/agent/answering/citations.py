"""Citation rendering and the separate, non-citable view of past answers.

Public answers retain source metadata for the UI and persistence. Model history,
plan dependencies and memory must not replay those renderings or turn-local IDs.
Nothing here validates, repairs or relaxes citations in a new model answer.
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
# History-only fallback for rendered headings that themselves contain 【】.
# Keep the current-answer validation pattern above unchanged.
_RENDERED_HISTORY_CITATION_RE = re.compile(
    r"【[^\r\n【】]*\.md · [^\r\n]*?原文第 \d+(?:-\d+)? 行】",
    re.IGNORECASE,
)


def format_source(source: AnswerSource) -> str:
    """Render a public citation solely from indexed source metadata."""
    parts = [source.source_path]
    if source.heading_path:
        parts.append(" > ".join(source.heading_path))
    if source.start_line == source.end_line:
        parts.append(f"原文第 {source.start_line} 行")
    else:
        parts.append(f"原文第 {source.start_line}-{source.end_line} 行")
    return f"【{' · '.join(parts)}】"


def historical_answer_text(text: str, sources: Sequence[AnswerSource] = ()) -> str:
    """Remove old citations before truncating or sending background to a model.

    Match known renderings exactly first, including unusual paths/headings. The
    pattern also covers old summaries and stored answers lacking source metadata.
    Never turn an old reference into [S1]: IDs belong only to the current evidence.
    This projection leaves the stored/displayed answer and source objects intact.
    """
    cleaned = text
    for source in sources:
        cleaned = cleaned.replace(format_source(source), "")
    cleaned = _RENDERED_HISTORY_CITATION_RE.sub("", cleaned)
    cleaned = WRITTEN_CITATION_RE.sub("", cleaned)
    return _HISTORICAL_SOURCE_TOKEN_RE.sub("", cleaned).strip()
