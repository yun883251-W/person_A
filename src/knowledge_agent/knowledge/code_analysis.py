"""共享“仅从对话中已可见内容提取有界代码”的能力。

本模块只扫描文本。它绝不读取路径、导入指定模块或执行用户代码。
选择器和检索使用同一套提取规则。
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

    # 未知语言的围栏不会仅仅因为包含括号就被当成 Python。
    # 在围栏之外，接受 import、定义以及可见的调用。
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
    """返回可见代码，以及 ``question``/``visible_context`` 的来源标记。

    无论是否使用围栏，当前问题中的代码都优先于上下文。
    仅凭文件路径不算可见代码。输出仍上限为 12,000 个字符，
    且只作为分析对象，绝不构成可执行授权。
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
