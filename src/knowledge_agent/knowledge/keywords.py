"""索引与检索共享的关键词规范化。"""

from __future__ import annotations

import re


_KEYWORD_TOKEN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_.:-]*|\d+(?:\.\d+)*|[\u3400-\u9fff]"
)


def extract_keywords(text: str) -> tuple[str, ...]:
    """按首次出现顺序返回去重后的规范化关键词。"""
    return tuple(dict.fromkeys(token.lower() for token in _KEYWORD_TOKEN.findall(text)))
