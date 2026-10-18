"""Recognize one current Bilibili search request and its literal topic.

The browser MCP tool owns navigation. This module supplies a narrow routing
boundary and can extract a literal topic after that same boundary accepts it.
The caller must validate provenance and arguments at the execution boundary.
Quoted commands, explanations, hypothetical requests and compound work do not
authorize browser actions. A quoted search topic is ordinary input data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from browser_mcp.contracts import SEARCH_BILIBILI_TOOL, validate_search_arguments


_SITE = r"(?<![A-Za-z0-9_])(?:bilibili|bilili|bili|b\s*站|哔哩哔哩)(?![A-Za-z0-9_])"
_SITE_RE = re.compile(_SITE, re.I)
_PREFIX = (
    r"(?:(?:请你|请|麻烦你|麻烦|帮我|帮忙|替我|为我|给我|"
    r"我想要|我想|我要|我需要|能不能|能否|可不可以|可以|现在|先)\s*)*"
)
_BROWSER = (
    r"(?:(?:打开|启动)(?:一下)?\s*(?:默认)?浏览器\s*"
    r"(?:[，,]\s*)?(?:(?:并|然后|再)\s*)?)?"
)
_DESTINATION = rf"(?:(?:在|到|去|上|进入|访问|打开)\s*)?{_SITE}\s*(?:里面|上|里|网站)?\s*"
_SEARCH_RE = re.compile(
    rf"^(?P<prefix>{_PREFIX})(?P<browser>{_BROWSER}){_DESTINATION}"
    r"(?:[，,]\s*)?(?:(?:并|然后|再)\s*)?(?:(?:帮我|给我|为我)\s*)?"
    r"(?P<action>搜索|搜一下|搜一搜|搜|查找|查询|查一下|找一下|找一找|找|学习|学一下|学|观看|看一下|看看|看)"
    r"\s*(?:一下\s*)?(?P<topic>[\s\S]*)$",
    re.I,
)
_OPEN_RE = re.compile(
    rf"^{_PREFIX}{_BROWSER}(?:打开|进入|访问|去|到|上)\s*{_SITE}"
    r"\s*(?:网站|首页)?\s*(?:吧|好吗|可以吗|行吗|吗)?[。.!！?？]*$",
    re.I,
)
_QUOTED_RE = re.compile(
    r'“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』|"[^"\n]*"|\'[^\'\n]*\'|`[^`\n]*`'
)
_META_REFERENCE_RE = re.compile(r"这句话|这段话|这个需求|这个功能|这条指令|这个指令|这条命令")
_EXTRA_OPERATION_RE = re.compile(
    r"[，,；;。\r\n]|然后|随后|接着|并且|同时|另外|之后|以及|但是|不过|顺便|"
    r"如果|假如|假设|要是|除非|否则|不要|不用|不需要|不想|别(?:打开|搜索|执行)|"
    r"(?:并|再|且)(?:(?:帮我|给我|替我|为我)\s*)?"
    r"(?:打开|进入|搜索|调用|执行|运行|解释|总结|开始|播放|下载|保存|点赞|投币|收藏|评论|登录|发弹幕)|"
    r"\b(?:then|and\s+then|if|unless|do\s+not|don't)\b",
    re.I,
)
_CURRENT_REQUEST_RE = re.compile(r"我想|我要|我需要|请|麻烦|帮我|帮忙|替我|为我|给我|能不能|能否|可以")
_REFERENTIAL_TOPIC_RE = re.compile(r"^(?:它|这个|那个|这些|那些|上述|上面|前面|刚才)")
_EMPTY_TOPIC_RE = re.compile(r"(?:一下|下|一搜|点|一些)?")
# These are request modifiers for 找/看, not part of the named subject. Keep
# explicitly searched strings intact: 搜索“一点资讯” must not become “资讯”.
_QUANTITY_PREFIX_RE = re.compile(r"^(?:一些|一点|几个|几部|一部|点|些)\s*")
_POLITE_END_RE = re.compile(r"(?:好吗|好不好|可以吗|行吗|吧|吗)$")
_NON_REQUEST_RE = re.compile(
    r"不要|不用|不需要|不想|别(?:去|帮|找|打开|搜|看|执行)|"
    r"如果|假如|假设|要是|除非|比如|例如|举例|示例|"
    r"(?:昨天|之前|曾经|当时|上次)|"
    r"(?:怎么|如何|怎样|为什么|是否)|"
    r"(?:这句|这段|这条|该指令|提示词|意图识别)|"
    r"\b(?:if|unless|do\s+not|don't|example|how\s+to)\b",
    re.I,
)
_REQUEST_START_RE = re.compile(
    rf"^(?:(?:请你|请|麻烦你|麻烦|帮我|帮忙|替我|为我|给我|"
    rf"我想要|我想|我要|我需要|能不能|能否|可不可以|可以|现在|先)|{_DESTINATION})",
    re.I,
)
_CONSULTATION_RE = re.compile(
    r"^(?:(?:我想要|我想|我要|我需要|请你|请|帮我|可以|能否|能不能|可不可以)\s*)*"
    r"(?:知道|了解|解释|介绍|说明|分析|讨论|评价|请教|讲讲|讲解|告诉我)"
)
_GENERIC_TOPIC_RE = re.compile(
    r"^(?:(?:一些|一点|点|些|几个|几部|一部|相关|好看|好看的|随便|任意|推荐|的)\s*)*"
    r"(?:(?:视频|影片|电影|电视剧|番剧|教程|课程|内容|东西)(?:看看|看一下|推荐)?|看看)?$"
)


@dataclass(frozen=True, slots=True)
class BrowserActionDecision:
    """A routing result with optional verbatim topic evidence, not tool args."""

    kind: Literal["search", "needs_clarification", "knowledge", "semantic_candidate"]
    reason_code: str
    clarification_question: str = ""
    topic: str = ""


def _topic_text(value: str) -> str:
    text = value.strip().strip("。.!！?？")
    text = _POLITE_END_RE.sub("", text).strip()
    return text.strip(" \t\r\n:：,，。.!！?？;；\"'“”‘’「」『』`")


def _matched_topic(match: re.Match[str]) -> str:
    """Trim only request framing and leave the topic as a literal substring."""
    raw_topic = match.group("topic").strip()
    if match.group("action").startswith(("找", "看")) or match.group("action") == "观看":
        raw_topic = _QUANTITY_PREFIX_RE.sub("", raw_topic, count=1)
    return _topic_text(raw_topic)


def is_explicit_bilibili_topic(topic: str) -> bool:
    """Require a named topic from this input, not history or generic video words."""
    if not isinstance(topic, str):
        return False
    value = _topic_text(topic)
    try:
        validate_search_arguments({"query": value})
    except (TypeError, ValueError):
        return False
    return bool(
        value
        and _EMPTY_TOPIC_RE.fullmatch(value) is None
        and _REFERENTIAL_TOPIC_RE.search(value) is None
        and _GENERIC_TOPIC_RE.fullmatch(value) is None
        and _SITE_RE.fullmatch(value) is None
    )


def _unknown_bilibili_request(text: str) -> BrowserActionDecision:
    """Separate genuinely unknown wording from blocked quotes and discussions.

    This gate never authorizes a tool. It admits only one current, explicit
    request to a pure classifier, which must still establish a named topic.
    Quoted titles can be data, but the destination and request must be unquoted.
    """
    discussion = BrowserActionDecision("knowledge", "browser_bilibili_discussion")
    unquoted = _QUOTED_RE.sub(" ", text).strip().rstrip("。.!！?？")
    if (
        _SITE_RE.search(unquoted) is None
        or _CURRENT_REQUEST_RE.search(unquoted) is None
        or _REQUEST_START_RE.search(unquoted) is None
        or _CONSULTATION_RE.search(unquoted) is not None
        or _NON_REQUEST_RE.search(unquoted) is not None
        or _META_REFERENCE_RE.search(unquoted) is not None
    ):
        return discussion
    if _EXTRA_OPERATION_RE.search(unquoted):
        return BrowserActionDecision(
            "needs_clarification", "browser_bilibili_single_action_only",
            "请明确本次 B 站搜索的主题；组合任务中也需要分别写明每一步。",
        )
    return BrowserActionDecision("semantic_candidate", "browser_bilibili_semantic_candidate")


def classify_bilibili_request(question: str) -> BrowserActionDecision | None:
    """Recognize only a complete site-specific command in the current turn.

    None leaves unrelated requests to existing intent handling. Unknown wording
    with an explicit current request can be classified once in hybrid mode.
    Definite discussions, negations and quoted commands never enter that path.
    Topic evidence alone is not an execution grant; the executor checks provenance.
    A missing-topic result is specific to a single positive search request;
    generic semantic ambiguity must not be promoted to this result by callers.
    """
    if not isinstance(question, str) or not question.strip():
        return None
    text = question.strip()
    if _SITE_RE.search(text) is None:
        return None
    discussion = BrowserActionDecision("knowledge", "browser_bilibili_discussion")
    if "```" in text or "~~~" in text:
        return discussion
    match = _SEARCH_RE.fullmatch(text)
    if match is None:
        if _OPEN_RE.fullmatch(text):
            return BrowserActionDecision(
                "needs_clarification", "browser_bilibili_missing_topic",
                "你想在 B 站搜索什么内容？请写出主题或作品名称。",
            )
        return _unknown_bilibili_request(text)

    raw_topic = match.group("topic")
    # Do not treat learning/viewing questions or a discussion of the sentence as
    # an instruction to navigate. Search topics may legitimately contain words
    # such as 教程、实现、代码、工具, so those are not global execution blockers.
    unquoted_topic = _QUOTED_RE.sub(" ", raw_topic).strip().rstrip("。.!！?？")
    if _META_REFERENCE_RE.search(unquoted_topic):
        return discussion
    action = match.group("action")
    if action.startswith(("找", "看")) or action == "观看":
        # 找不到/看不了 describes a problem, and question wording may omit
        # punctuation. Quoted work titles remain ordinary search topics.
        if re.search(r"^不(?:了|到|见|成)|怎么|如何|是否|为什么|是什么意思|什么意思", unquoted_topic):
            return discussion
    if action.startswith(("学", "看", "找")) or action == "观看":
        # Viewing intent means searching for the named content, not selecting
        # or playing a video. Keep the same explicit-request boundary as 学习.
        if (
            not action.startswith("找")
            and not match.group("browser")
            and not _CURRENT_REQUEST_RE.search(match.group("prefix"))
        ):
            return discussion
        ending = raw_topic.rstrip("。.!！")
        if re.search(r"吗[?？]?$|么[?？]?$|[?？]$", ending) and not re.search(
            r"(?:好吗|可以吗|行吗|好不好)[?？]?$", ending,
        ):
            return discussion
    if _EXTRA_OPERATION_RE.search(unquoted_topic):
        return BrowserActionDecision(
            "needs_clarification", "browser_bilibili_single_action_only",
            "请将 B 站搜索单独发送，并明确本次要搜索的主题。",
        )
    topic = _matched_topic(match)
    if not is_explicit_bilibili_topic(topic):
        return BrowserActionDecision(
            "needs_clarification", "browser_bilibili_missing_topic",
            "你想在 B 站搜索什么内容？请补充主题或作品名称。",
        )
    return BrowserActionDecision("search", "browser_bilibili_search", topic=topic)


def extract_bilibili_topic(question: str) -> str | None:
    """Return a literal topic only after the complete request gate accepts it.

    Use this helper for slot provenance instead of independently searching for
    a verb with a looser regex. It does not extract from quoted commands,
    negations, conditions, discussions, compound work or semantic candidates.
    A bare reply such as 火影忍者 needs an already established pending task;
    this helper intentionally does not authorize such a reply on its own.
    """
    decision = classify_bilibili_request(question)
    if decision is None or decision.kind != "search":
        return None
    topic = decision.topic
    return topic if topic in question and is_explicit_bilibili_topic(topic) else None


def is_bilibili_search_request(question: str) -> bool:
    """Whether this turn authorizes the single natural-language search action."""
    decision = classify_bilibili_request(question)
    return decision is not None and decision.kind == "search"
