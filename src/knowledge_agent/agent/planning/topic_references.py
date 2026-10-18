"""Bind a Bilibili topic reference to text in this request, never an answer.

This module is deliberately narrower than general intent recognition. A full
positive search/watch request with an otherwise implicit topic may reuse one
unambiguous subject of earlier knowledge clauses. It cannot authorize a tool,
consult conversation history, choose between different subjects, or use a
previous step's generated result. The caller still owns tool availability,
authorization, argument validation and execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TopicBinding:
    """Runtime provenance copied exclusively from the current user request."""

    original_clause: str
    topic: str
    source_step_id: str
    source_clause: str


class TopicReferenceError(ValueError):
    """A fixed clarification code; do not copy arbitrary input into errors."""

    def __init__(self, code: str) -> None:
        if code not in {
            "plan_topic_reference_missing", "plan_topic_reference_ambiguous",
        }:
            raise ValueError("unsupported topic reference error")
        super().__init__(code)
        self.code = code


# Entire clauses, not keywords found inside quotations, questions, conditions,
# or negative instructions, must match this grammar. In particular, no optional
# arbitrary text may occur between the platform, action and referenced object.
_LEAD = (
    r"(?:(?:首先|然后|随后|接着|最后|其次|先|再|请帮我|请你|请|"
    r"麻烦你|麻烦|帮我|替我|为我|给我|我想要|我想|我要|我需要)\s*)*"
)
_SITE = r"(?:bilibili|bilili|bili|b\s*站|哔哩哔哩)"
_ACTION = (
    r"(?:搜索|搜一搜|搜一下|搜|查找|找一找|找|观看|看看|看一看|看)"
    r"\s*(?:(?:一下|一些|一点|几个|几条|些|点)\s*)?"
)
_MEDIA = r"(?:视频|教程|课程)"
_REFERENCE = (
    r"(?:相关(?:的)?\s*" + _MEDIA + r"|"
    r"(?:这个|该|上述|前述|刚才(?:的)?)(?:主题|概念)"
    r"(?:的)?(?:相关(?:的)?)?(?:" + _MEDIA + r")?|"
    r"(?:与|和)(?:这个主题|该主题|这个概念|该概念|它|之)"
    r"相关(?:的)?\s*" + _MEDIA + r")"
)
_REFERENCE_SEARCH = re.compile(
    r"^\s*" + _LEAD + r"(?:"
    r"(?:在|去|到|上)?\s*" + _SITE + r"\s*(?:上|里|里面)?\s*"
    + _ACTION + _REFERENCE + r"|"
    + _ACTION + r"(?:在|去|到|上)?\s*" + _SITE
    + r"\s*(?:上|里|里面)?\s*(?:的)?\s*" + _REFERENCE
    + r")\s*[。.!！]?\s*$",
    re.IGNORECASE,
)

_KNOWLEDGE_LEAD = re.compile(
    r"^\s*" + _LEAD
    + r"(?:解释|介绍|说明|讲解|讲讲)\s*(?:一下|下)?\s*"
    r"(?:(?:什么是|何为)\s*)?(?P<subject>.+?)\s*[。.!！?？]?\s*$",
    re.IGNORECASE,
)
_KNOWLEDGE_QUESTION = re.compile(
    r"^\s*" + _LEAD + r"(?P<subject>.+?)\s*(?:是什么|有什么作用)"
    r"\s*[。.!！?？]?\s*$",
    re.IGNORECASE,
)
_UNRESOLVED_KNOWLEDGE = re.compile(
    r"^\s*" + _LEAD + r"(?:比较|对比|分析|回答|讨论|总结|归纳|整理|"
    r"列出|给出|提供|告诉我|什么是|何为|解释|介绍|说明|讲解|讲讲|"
    r"制定|安排|生成|编写|写出|写个|写一个|实现|设计|演示|举例|举个例|"
    r"根据|基于|结合|参照|参考|对照|扩展|优化|补充|改进)\s*.*$",
    re.IGNORECASE,
)
_ASPECT = r"(?:工作原理|基础知识|基本概念|基本原理|实现原理|原理|作用|用途|用法|机制|概念|定义|基础)"
_ASPECT_SUFFIX = re.compile(
    r"\s*的\s*" + _ASPECT + r"(?:\s*(?:和|与|及|以及)\s*" + _ASPECT + r")*\s*$",
)
_QUESTION_SUFFIX = re.compile(r"\s*(?:是什么|有什么作用)\s*$")
_COMPOUND_SUBJECT = re.compile(
    r"和|与|以及|及|还有|同时|并且|并\s*(?:解释|介绍|说明|讲解|比较|分析|讨论)|"
    r"、|[,，;；/&]|\b(?:and|or|versus|vs)\b",
    re.IGNORECASE,
)
_UNSAFE_SUBJECT = re.compile(
    r"(?:https?://|www\.|://)|[\u0000-\u001f\u007f-\u009f\ud800-\udfff]|"
    r"[\"'`“”‘’「」『』:：;；!?！？=<>\\{}\[\]()]|"
    r"(?:忽略|不要|不必|无需|禁止|假如|如果|假设|否则|除非|"
    r"执行|调用|运行|打开|搜索|查找|删除|上传|发送|下载|安装|"
    r"指令|提示词|命令|上一步|前一步|回答|结果|输出|"
    r"上述|上文|前文|刚才|之前|前面|上面|这个|那个|该主题|相关视频|"
    r"它们|它|如何|怎样|怎么|然后|随后|接着|最后|首先|帮我|告诉我)|"
    r"\b(?:if|unless|ignore|execute|invoke|run|open|delete|upload|"
    r"download|instruction|prompt|answer|result|previous)\b",
    re.IGNORECASE,
)
_INSTRUCTION_SUBJECT = re.compile(
    r"^(?:请|麻烦|替我|为我|给我|让我|把|将|解释|介绍|说明|讲解|讲讲|"
    r"比较|对比|分析|讨论|总结|归纳|整理|列出|给出|提供|回答|什么是|何为)",
)
_SUBJECT_CHARS = re.compile(r"^[\w\s.+#\-]+$", re.UNICODE)
_NO_SUBJECT = frozenset({
    "什么", "这个", "那个", "主题", "概念", "内容", "原理", "作用", "用法",
    "一下", "下", "一些", "一点",
    "资料", "知识", "视频", "教程", "课程", "相关", "相关内容", "相关知识",
    "this", "that", "it", "topic", "concept",
})


def _subject_candidate(clause: str) -> tuple[str | None, bool]:
    """Return a literal subject and whether a knowledge clause is unresolved.

    An unsupported/compound knowledge clause remains an ambiguity signal. It
    must not be silently dropped in favor of another convenient earlier topic.
    A non-knowledge clause contributes no candidate to this narrow resolver.
    """
    match = _KNOWLEDGE_LEAD.fullmatch(clause) or _KNOWLEDGE_QUESTION.fullmatch(clause)
    if match is None:
        return None, _UNRESOLVED_KNOWLEDGE.fullmatch(clause) is not None
    subject = match.group("subject").strip()
    subject = _QUESTION_SUFFIX.sub("", subject).strip()
    subject = _ASPECT_SUFFIX.sub("", subject).strip()
    if (
        not subject or len(subject) > 200 or subject.casefold() in _NO_SUBJECT
        or _COMPOUND_SUBJECT.search(subject) is not None
        or _UNSAFE_SUBJECT.search(subject) is not None
        or _INSTRUCTION_SUBJECT.search(subject) is not None
        or _SUBJECT_CHARS.fullmatch(subject) is None
        or not any(character.isalnum() for character in subject)
    ):
        return None, True
    # All transformations remove prefixes/suffixes only. This explicit check
    # prevents a future extraction change from manufacturing a search topic.
    if subject not in clause:
        return None, True
    return subject, False


def resolve_topic_reference(
    clause: str, previous_clauses: tuple[str, ...],
) -> TopicBinding | None:
    """Resolve only a full positive Bilibili request with an implicit topic.

    ``None`` means this special path does not apply; normal intent/dependency
    rules retain ownership. Once this grammar matches, missing or ambiguous
    source topics request clarification rather than guessing from history or
    generated answers. Repeated identical topic text is one candidate;
    provenance always points to its first occurrence, never the nearest result.
    """
    if not isinstance(clause, str):
        raise TypeError("clause must be a string")
    if not isinstance(previous_clauses, tuple) or not all(
        isinstance(item, str) for item in previous_clauses
    ):
        raise TypeError("previous_clauses must be a tuple of strings")
    if _REFERENCE_SEARCH.fullmatch(clause) is None:
        return None

    candidates: dict[str, tuple[int, str, str]] = {}
    unresolved = False
    for index, previous in enumerate(previous_clauses, 1):
        topic, ambiguous = _subject_candidate(previous)
        unresolved = unresolved or ambiguous
        if topic is not None:
            candidates.setdefault(topic, (index, previous, topic))
    if unresolved or len(candidates) > 1:
        raise TopicReferenceError("plan_topic_reference_ambiguous")
    if not candidates:
        raise TopicReferenceError("plan_topic_reference_missing")
    index, source, topic = next(iter(candidates.values()))
    return TopicBinding(clause, topic, f"P{index}", source)


def validate_topic_binding(
    binding: TopicBinding, original_question_clauses: tuple[str, ...],
) -> bool:
    """Re-derive topic provenance for execution or read-only summary display.

    An execution caller must supply freshly split clauses of the actual current
    request, never clauses reconstructed from a stored plan or model response.
    A display validator may check stored summary consistency using this helper;
    that does not make stored data executable. Equality with a fresh resolution
    checks the exact original action, source position and literal topic together.
    This helper grants no tool permission by itself.
    """
    if (
        not isinstance(binding, TopicBinding)
        or not isinstance(original_question_clauses, tuple)
        or not 2 <= len(original_question_clauses) <= 3
        or not all(isinstance(item, str) for item in original_question_clauses)
        or original_question_clauses.count(binding.original_clause) != 1
    ):
        return False
    index = original_question_clauses.index(binding.original_clause)
    if index == 0:
        return False
    try:
        resolved = resolve_topic_reference(
            binding.original_clause, original_question_clauses[:index],
        )
    except (TypeError, TopicReferenceError):
        return False
    return resolved == binding
