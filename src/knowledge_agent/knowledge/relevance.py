"""低成本、确定性的相关性信号；不调用模型，也不修改语料。

余弦阈值是依赖具体模型的启发式规则，不是经过校准的概率。
词法支持使用技术词和连续汉字二元组，而不是旧索引所用的单字倒排匹配。
另有一条更高的余弦阈值，使没有共同措辞的改写仍可入选。
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
# 在生成汉字二元组之前移除完整的请求表达，例如“解释一下”和“工作原理”
# 就不能让无关段落看起来相关。
# 这刻意是查询语言规范化，而不是允许主题的白名单。
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
    """返回有意义的词法提示，但不要求语义召回依赖它们。"""
    current_markers = tuple(_CURRENT_QUESTION.finditer(question))
    if current_markers:
        # 生成的查询可能嵌套更早的追问。以当前标记开头的查询由其首行主导；
        # 历史在前的格式则把当前问题放在最后。绝不要求每个旧主题、包装标签
        # 或摘要指令都匹配到同一个检索片段。
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
            # 自由形式的摘要可能包含指令/元数据而不是问题。
            # 不要把这些假装成词法主题要求。
            return ()
        history = _ANY_QUERY_LABEL.sub("\n", history[history_marker.start():]).strip()
        # 纯指代的追问没有当前主题。用最近的问题作为弱词法提示；
        # 完整的 Embedding 查询保持不变。
        for recent_question in reversed(re.split(r"[；\n]", history)):
            terms = _extract_terms(recent_question)
            if terms:
                return terms
        return ()
    # 一些计划中的追问把原始目标放在历史块之前。
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
    """连续匹配汉字二元组，并把技术标识符作为整词匹配。"""
    normalized = text.lower()
    technical_words = {
        word.lower().rstrip(".:") for word in _TECHNICAL_WORD.findall(text)
    }
    return tuple(
        term for term in terms
        if (term in technical_words if term.isascii() else term in normalized)
    )


def topic_coverage(terms: tuple[str, ...], matched: tuple[str, ...]) -> float:
    """不同词法提示被支持的比例；绝不是相关性概率。"""
    return len(matched) / len(terms) if terms else 0.0
