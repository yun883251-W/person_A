"""Cheap, deterministic relevance signals; no model calls or corpus mutation.

Cosine cutoffs are model-dependent heuristics, not calibrated probabilities.
Lexical support uses technical words and consecutive Chinese character pairs,
not the single-character posting matches used to build the legacy index. A
separate, higher cosine cutoff keeps paraphrases with no shared wording eligible.
"""

from __future__ import annotations

import re


_TECHNICAL_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_.:+#/-]*")
_CHINESE_RUN = re.compile(r"[\u3400-\u9fff]{2,}")
_CURRENT_QUESTION = re.compile(r"(?:^|\n)当前追问[:：]\s*")
_HISTORY_LABEL = re.compile(r"(?:^|\n)(?:先前问题|最近知识上下文|历史知识背景)[:：]\s*")
_QUESTION_HISTORY_LABEL = re.compile(r"(?:^|\n)(?:先前问题|最近知识上下文)[:：]\s*")
_ANY_QUERY_LABEL = re.compile(r"(?:^|\n)(?:当前追问|先前问题|最近知识上下文|历史知识背景)[:：]\s*")
_ENGLISH_QUESTION_WORDS = frozenset({
    "a", "an", "and", "are", "can", "compare", "could", "describe", "difference",
    "do", "does", "explain", "for", "how", "i", "in", "introduce", "is", "it",
    "me", "of", "please", "tell", "the", "to", "what", "which", "why", "with",
    "work", "works", "would", "you",
})
# Remove complete request expressions before making Chinese pairs, so e.g.
# “解释一下” and “工作原理” cannot make an unrelated passage look relevant.
# This is deliberately query-language normalization, not a list of allowed topics.
_QUESTION_EXPRESSIONS = re.compile("|".join(re.escape(value) for value in sorted({
    "请帮我", "麻烦你", "请问", "请你", "帮我", "我想了解", "我想知道",
    "解释一下", "介绍一下", "讲解一下", "说明一下", "什么是", "是什么",
    "工作原理", "基本原理", "有什么作用", "有什么区别", "有什么不同",
    "有何区别", "有何不同", "的工作方式", "怎么工作", "如何工作",
    "之间的区别", "之间的不同", "的区别", "的作用", "的含义", "的概念",
    "解释", "介绍", "讲解", "讲讲", "说明", "比较", "对比", "分析",
    "首先", "然后", "最后", "一下", "什么", "如何", "怎样", "为什么",
    "这个", "那个", "这些", "那些", "它们", "它", "继续", "举个例子",
    "举例", "例子", "优缺点", "优点", "缺点", "注意事项", "怎么用",
    "以及", "和", "与", "及", "的", "吗", "呢", "请",
}, key=lambda value: (-len(value), value))))


def extract_topic_terms(question: str) -> tuple[str, ...]:
    """Return meaningful lexical hints without requiring them for semantic recall."""
    current_markers = tuple(_CURRENT_QUESTION.finditer(question))
    if current_markers:
        # Generated queries can contain nested older follow-ups. A query beginning
        # with the current marker owns its first line; the history-first format
        # appends its current question last. Never require every old topic, wrapper
        # label or summary instruction to match a single retrieved chunk.
        current = current_markers[0] if current_markers[0].start() == 0 else current_markers[-1]
        focused = question[current.end():].split("\n", 1)[0]
        terms = _extract_terms(focused)
        if terms:
            return terms
        history = (
            question[:current.start()] if current.start()
            else question[current.end() + len(focused):]
        )
        history_marker = _QUESTION_HISTORY_LABEL.search(history)
        if history_marker is None:
            # A free-form summary may contain instructions/metadata rather than
            # questions. Do not pretend these are lexical topic requirements.
            return ()
        history = _ANY_QUERY_LABEL.sub("\n", history[history_marker.start():]).strip()
        # A purely referential follow-up has no current topic. Use the nearest
        # question as a weak lexical hint; the full embedding query is unchanged.
        for recent_question in reversed(re.split(r"[；\n]", history)):
            terms = _extract_terms(recent_question)
            if terms:
                return terms
        return ()
    # Some planned follow-ups put the original goal before a history block.
    question = _HISTORY_LABEL.split(question, maxsplit=1)[0]
    return _extract_terms(question)


def _extract_terms(question: str) -> tuple[str, ...]:
    terms: list[str] = []
    for word in _TECHNICAL_WORD.findall(question):
        normalized = word.lower().rstrip(".:")
        if normalized and normalized not in _ENGLISH_QUESTION_WORDS:
            terms.append(normalized)
    chinese_text = _QUESTION_EXPRESSIONS.sub(" ", question)
    for run in _CHINESE_RUN.findall(chinese_text):
        terms.extend(run[index:index + 2] for index in range(len(run) - 1))
    return tuple(dict.fromkeys(terms))


def matching_topic_terms(terms: tuple[str, ...], text: str) -> tuple[str, ...]:
    """Match Chinese pairs contiguously and technical identifiers as whole words."""
    normalized = text.lower()
    technical_words = {
        word.lower().rstrip(".:") for word in _TECHNICAL_WORD.findall(text)
    }
    return tuple(
        term for term in terms
        if (term in technical_words if term.isascii() else term in normalized)
    )


def topic_coverage(terms: tuple[str, ...], matched: tuple[str, ...]) -> float:
    """Fraction of distinct lexical hints supported; never a relevance probability."""
    return len(matched) / len(terms) if terms else 0.0
