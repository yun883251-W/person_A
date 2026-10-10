"""Shared keyword normalization for indexing and retrieval."""

from __future__ import annotations

import re


_KEYWORD_TOKEN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_.:-]*|\d+(?:\.\d+)*|[\u3400-\u9fff]"
)


def extract_keywords(text: str) -> tuple[str, ...]:
    """Return unique normalized keywords in their first-seen order."""
    return tuple(dict.fromkeys(token.lower() for token in _KEYWORD_TOKEN.findall(text)))
