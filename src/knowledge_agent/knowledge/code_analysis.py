"""Share bounded extraction of code that is already visible in a conversation.

This module only scans text. It never reads paths, imports a named module, or
executes the user's code. Selection and retrieval use the same extraction rule.
"""

from __future__ import annotations

import re

MAX_VISIBLE_CODE_CHARS = 12_000
_SCAN_CHARS = MAX_VISIBLE_CODE_CHARS * 4
_PYTHON_FENCE = re.compile(
    r"^[ \t]*(?:```|~~~)(?:python|py)?[ \t]*\r?\n"
    r"([\s\S]*?)(?:^[ \t]*(?:```|~~~)[ \t]*$|\Z)",
    re.I | re.M,
)
_ANY_FENCE = re.compile(r"(?:```|~~~)[\s\S]*?(?:(?:```|~~~)|\Z)")
_CODE_START = re.compile(
    r"^\s*(?:from\s+[.\w]+\s+import\b|import\s+[\w.]|"
    r"(?:async\s+)?def\s+\w+\s*\(|class\s+\w+\s*[:(]|"
    r"@\w|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*(?::[^=\n]+)?=(?!=))"
)
_CALL = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(")
_CONTINUATION = re.compile(
    r"^\s*(?:[\])}]|return\b|yield\b|raise\b|pass\b|break\b|continue\b|"
    r"(?:async\s+)?(?:for|with)\b|if\b|elif\b|else\s*:|while\b|"
    r"try\s*:|except\b|finally\s*:|assert\b|#)"
)


def _extract(text: str) -> str:
    bounded = text[:_SCAN_CHARS]
    fenced = [match.group(1).strip("\r\n") for match in _PYTHON_FENCE.finditer(bounded)]
    fenced = [block for block in fenced if block.strip()]
    if fenced:
        return "\n\n".join(fenced)[:MAX_VISIBLE_CODE_CHARS]

    # Unknown language fences do not become Python merely because they contain
    # parentheses. Outside fences, accept imports, definitions and visible calls.
    plain = _ANY_FENCE.sub("\n", bounded)
    selected: list[str] = []
    active = False
    for line in plain.splitlines():
        if _CODE_START.search(line) or _CALL.search(line):
            selected.append(line)
            active = True
        elif active and (not line.strip() or _CONTINUATION.search(line) or line.startswith((" ", "\t"))):
            selected.append(line)
        else:
            active = False
    return "\n".join(selected).strip("\r\n")[:MAX_VISIBLE_CODE_CHARS]


def extract_visible_code(question: str, visible_context: str = "") -> tuple[str, str]:
    """Return visible code plus ``question``/``visible_context`` provenance.

    Current-question code takes precedence over context, whether fenced or bare.
    A file path alone is not visible code. The output remains capped at 12,000
    characters and is an analysis subject, never executable authorization.
    """
    if not isinstance(question, str) or not isinstance(visible_context, str):
        raise TypeError("question and visible_context must be strings")
    code = _extract(question)
    if code:
        return code, "question"
    code = _extract(visible_context)
    if code:
        return code, "visible_context"
    return "", "question"
