"""Recognize one current Bilibili search request without creating tool arguments.

The browser MCP tool owns navigation. This module only supplies a narrow routing
boundary; the existing MCP model still produces and validates the final query.
Quoted commands, explanations, hypothetical requests and compound work do not
authorize browser actions. A quoted search topic is ordinary input data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from browser_mcp.contracts import SEARCH_BILIBILI_TOOL


_SITE = r"(?<![A-Za-z0-9_])(?:bilibili|bilili|bili|b\s*站|哔哩哔哩)(?![A-Za-z0-9_])"
_SITE_RE = re.compile(_SITE, re.I)
_ACTION_RE = re.compile(r"打开|进入|访问|搜索|搜|查找|查询|学习|学一下|学", re.I)
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
    r"(?P<action>搜索|搜一下|搜一搜|搜|查找|查询|查一下|学习|学一下|学)"
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
    r"[，,；;。\r\n]|然后|随后|接着|并且|同时|另外|之后|以及|但是|不过|"
    r"如果|假如|假设|要是|除非|否则|不要|不用|不需要|不想|别(?:打开|搜索|执行)|"
    r"(?:并|再|且)(?:打开|进入|搜索|调用|执行|运行|解释|总结|开始)|"
    r"\b(?:then|and\s+then|if|unless|do\s+not|don't)\b",
    re.I,
)
_CURRENT_REQUEST_RE = re.compile(r"我想|我要|我需要|请|麻烦|帮我|帮忙|替我|为我|给我|能不能|能否|可以")
_REFERENTIAL_TOPIC_RE = re.compile(r"^(?:它|这个|那个|这些|那些|上述|上面|前面|刚才)")
_EMPTY_TOPIC_RE = re.compile(r"(?:一下|下|一搜|点|一些)?")
_POLITE_END_RE = re.compile(r"(?:好吗|好不好|可以吗|行吗|吧|吗)$")


@dataclass(frozen=True, slots=True)
class BrowserActionDecision:
    """A routing result, deliberately without executable arguments."""

    kind: Literal["search", "needs_clarification", "knowledge"]
    reason_code: str
    clarification_question: str = ""


def _topic_text(value: str) -> str:
    text = value.strip().strip("。.!！?？")
    text = _POLITE_END_RE.sub("", text).strip()
    return text.strip(" \t\r\n:：,，。.!！?？;；\"'“”‘’「」『』`")


def classify_bilibili_request(question: str) -> BrowserActionDecision | None:
    """Recognize only a complete site-specific command in the current turn.

    None leaves unrelated requests to existing intent handling. Mentions of a
    browser action that fail this command grammar remain knowledge requests;
    they cannot be promoted to execution by the optional intent classifier.
    Topic inspection checks completeness only and never supplies tool arguments.
    """
    if not isinstance(question, str) or not question.strip():
        return None
    text = question.strip()
    if _SITE_RE.search(text) is None or _ACTION_RE.search(text) is None:
        return None
    discussion = BrowserActionDecision("knowledge", "browser_bilibili_discussion")
    if "```" in text or "~~~" in text:
        return discussion
    match = _SEARCH_RE.fullmatch(text)
    if match is None:
        if _OPEN_RE.fullmatch(text):
            return BrowserActionDecision(
                "needs_clarification", "browser_bilibili_missing_topic",
                "你想在 B 站搜索或学习什么主题？",
            )
        return discussion

    raw_topic = match.group("topic")
    # Do not treat a learning-method question or a discussion of the sentence as
    # an instruction to navigate. Search topics may legitimately contain words
    # such as 教程、实现、代码、工具, so those are not global execution blockers.
    unquoted_topic = _QUOTED_RE.sub(" ", raw_topic).strip().rstrip("。.!！?？")
    if _META_REFERENCE_RE.search(unquoted_topic):
        return discussion
    if match.group("action").startswith("学"):
        if not match.group("browser") and not _CURRENT_REQUEST_RE.search(match.group("prefix")):
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
    topic = _topic_text(raw_topic)
    if not topic or _EMPTY_TOPIC_RE.fullmatch(topic) or _REFERENTIAL_TOPIC_RE.search(topic):
        return BrowserActionDecision(
            "needs_clarification", "browser_bilibili_missing_topic",
            "你想在 B 站搜索或学习什么主题？请在本条消息中写出主题。",
        )
    return BrowserActionDecision("search", "browser_bilibili_search")


def is_bilibili_search_request(question: str) -> bool:
    """Whether this turn authorizes the single natural-language search action."""
    decision = classify_bilibili_request(question)
    return decision is not None and decision.kind == "search"
