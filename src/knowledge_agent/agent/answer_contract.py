"""Parse a small answer-status protocol without trusting self-reported evidence.

This is a formatting contract, not a factual entailment checker. A status label
never exempts a substantive answer from the current-turn citation checks. Only
short, entirely non-factual refusals/questions can be published without sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


AnswerKind = Literal[
    "supported", "partial", "insufficient_evidence", "needs_clarification", "out_of_scope",
]
RefusalKind = Literal["insufficient_evidence", "needs_clarification", "out_of_scope"]

ANSWER_VALIDATION_REASONS = frozenset({
    "missing_citations", "unknown_citation", "unverified_citation_metadata",
    "answer_contract_invalid", "answer_status_conflict",
})

ANSWER_CONTRACT_INSTRUCTION = """
最终知识回答必须以单独一行状态开头（工具调用时不要输出该行）：
ANSWER_STATUS: supported
把 supported 替换为以下恰当的值，下一行开始写给用户的正文：
- supported：资料支持当前问题，事实结论旁必须引用本轮 [S编号]。
- partial：只回答了有证据的部分；该部分必须有引用，另用独立句说明缺失的信息。
- insufficient_evidence：资料不相关或不足以提供任何有依据的回答。
- needs_clarification：必须先补充问题的对象、参数或上下文才能回答。
- out_of_scope：问题超出 Python / LangChain 学习及相关知识库的服务范围。
后三种状态只写简短的拒答或澄清（最多 3 句），不要继续讲事实、贴代码、附引用。
资料不足可写“当前提供的资料未涉及量子纠缠，无法依据这些资料回答这个问题。”；
超范围可写“这个问题超出了当前助手的服务范围。我可以帮助你学习 Python 和 LangChain。”；
澄清可写“请明确你想了解的对象或补充相关上下文。”。主题按实际问题替换。
不要把资料缺失说成服务错误，也不要为了补齐引用给不相关资料加来源标记。
状态行是程序协议，不在正文中重复，不要加代码围栏。它不代替真实的证据判断。
""".strip()

_HEADER = re.compile(
    r"ANSWER_STATUS:[ \t]*(supported|partial|insufficient_evidence|needs_clarification|out_of_scope)[ \t]*"
)
_HEADER_START = re.compile(r"^ANSWER_STATUS\b", re.IGNORECASE | re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[。！？!?])\s*|\n+|(?<=[.])\s+(?=[A-Z])")
_DECORATION = re.compile(r"^\s*(?:[-*]\s+|\*\*|#+\s*)|\*\*$")

# Deliberately anchored to statements about this response/material, rather than
# tutorial words such as “部分”, “没有” or “请说明” appearing anywhere in prose.
_EVIDENCE_REFUSAL = re.compile(
    r"^(?:(?:抱歉|很抱歉|对不起)[，,：:]?\s*)?(?:"
    r"(?:当前|本轮|现有|上述|这些|所给|提供的|检索到的|当前提供的|目前提供的)?"
    r"(?:知识库(?:中|里)?(?:的?资料)?|参考资料|资料|材料|文档|检索结果|证据)"
    r"(?:中|里)?(?:并)?(?:暂无|没有|未|尚未|不|不足|缺乏|缺少|无法|不能)"
    r"[^。！？!?]{0,220}|"
    r"(?:根据|基于|仅凭)(?:当前|本轮|这些|上述|所给|提供的|检索到的)?"
    r"(?:知识库资料|参考资料|资料|材料|文档|检索结果)[，,]?"
    r"(?:我|我们)?(?:无法|不能|尚不能|还不能|未找到|没有找到|没有)[^。！？!?]{0,160}|"
    r"(?:我|我们)(?:无法|不能|尚不能|还不能)(?:依据|根据|从)"
    r"(?:当前|这些|提供的|本轮)?(?:资料|材料|文档|检索结果)[^。！？!?]{0,160}|"
    r"(?:there is (?:not enough|insufficient)|there isn't enough) (?:relevant )?"
    r"(?:information|evidence)(?: in (?:the |these |provided )?(?:sources|materials|documents))?"
    r"(?: to answer (?:this|your|the) question)?[.!]?|"
    r"(?:the (?:provided |retrieved )?(?:sources|materials|documents)) "
    r"(?:do not|don't) (?:cover|contain information (?:on|about)|support) [^.\n]{1,160}[.]?"
    r")[。.!！]?$",
    re.IGNORECASE,
)
_SCOPE_REFUSAL = re.compile(
    r"^(?:(?:抱歉|很抱歉|对不起)[，,：:]?\s*)?(?:"
    r"(?:这个|该|当前|你的|您的|你提出的|您提出的|你所问的)(?:问题|请求|主题)"
    r"(?:已经|已|明显)?(?:超出|超出了|不在|不属于)"
    r"(?:了)?(?:我|我们|本助手|当前助手|该助手|当前知识库|本知识库|当前服务)"
    r"(?:的)?[^。！？!?]{0,100}(?:范围|领域)|"
    r"(?:this|your) (?:question|request|topic) is (?:outside|beyond) "
    r"(?:my|our|this assistant's) (?:scope|supported scope)"
    r")[。.!！]?$",
    re.IGNORECASE,
)
_CLARIFICATION = re.compile(
    r"^(?:请(?:说明|明确|提供|补充|确认|告诉我)|"
    r"(?:你|您)指的是|(?:could you|please) (?:clarify|specify|provide|confirm))"
    r"[^。！？!?]{1,220}[。.!！？?]?$",
    re.IGNORECASE,
)
_SCOPE_HELP = re.compile(
    r"^(?:我|本助手|当前助手)(?:主要|只能|仅|可以|仅能)?"
    r"(?:帮助你(?:学习|理解|了解)|帮助您(?:学习|理解|了解)|支持|回答|提供)\s*"
    r"(?:与|有关)?\s*(?:Python|LangChain)"
    r"(?:\s*(?:和|与|/|、|及|或)\s*(?:Python|LangChain))*\s*"
    r"(?:相关的?|领域的?)?(?:学习帮助|学习问题|技术问题|学习|帮助|问题|知识)?[。.!！]?$",
    re.IGNORECASE,
)
_NEUTRAL = re.compile(r"^(?:抱歉|很抱歉|对不起|sorry)[。.!！]?$", re.IGNORECASE)
_UNSAFE_REFUSAL_CONTENT = re.compile(
    r"```|~~~|[：:]|(?:^|\n)\s*(?:#{1,6}\s|[-*]\s|\d+[.)、]\s)|"
    r"(?:但(?:是)?|不过|然而|实际上|事实上|一般来说|通常来说)[，,]?|"
    r"\b(?:however|but|actually|in fact|for example)\b",
    re.IGNORECASE,
)
_REFUSAL_CONTINUATION = re.compile(
    r"^(?:无法|不能|不足|没有|未|尚未|缺少|缺乏|不支持|不包含|不涵盖|不涉及|"
    r"并(?:未|不|没有)|也(?:未|不|没有)|(?:因而|因此|所以|故)(?:我|我们)?(?:无法|不能)|"
    r"(?:我|我们)(?:无法|不能)|请(?:说明|明确|提供|补充|确认))"
)


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    body: str
    kind: AnswerKind | None = None
    error_code: str | None = None


def parse_answer_contract(text: str) -> ParsedAnswer:
    """Read one leading header; retain legacy unmarked answers for validation."""
    cleaned = text.lstrip("\ufeff").strip()
    first, _, remainder = cleaned.partition("\n")
    header = _HEADER.fullmatch(first.strip())
    if header is not None:
        body = remainder.strip()
        if not body or _HEADER_START.search(body):
            return ParsedAnswer(body=body, error_code="answer_contract_invalid")
        return ParsedAnswer(body=body, kind=header.group(1))
    if _HEADER_START.search(cleaned):
        return ParsedAnswer(body="", error_code="answer_contract_invalid")
    return ParsedAnswer(body=cleaned)


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in _SENTENCE_END.split(text) if part.strip())


def _sentence_kind(sentence: str) -> RefusalKind | None:
    if _SCOPE_REFUSAL.fullmatch(sentence):
        return "out_of_scope"
    if _EVIDENCE_REFUSAL.fullmatch(sentence):
        return "insufficient_evidence"
    if _CLARIFICATION.fullmatch(sentence):
        return "needs_clarification"
    return None


def plain_refusal_kind(text: str) -> RefusalKind | None:
    """Accept only a bounded refusal/question, never a labelled factual essay.

    Every sentence must describe a refusal, request needed input, or state the
    assistant's scope. Unknown extra prose makes the entire response ineligible
    for the citation exemption. This intentionally favors a safe explanation
    over publishing an unverified answer from a nonconforming model.
    """
    if len(text) > 600 or _UNSAFE_REFUSAL_CONTENT.search(text):
        return None
    sentences = _sentences(text)
    if not sentences or len(sentences) > 4:
        return None
    kinds: set[RefusalKind] = set()
    for sentence in sentences:
        kind = _sentence_kind(sentence)
        if kind is not None:
            # A refusal followed by “，RAG 是……” is still a factual answer;
            # do not let its opening clause grant a citation exemption.
            if kind != "needs_clarification":
                clauses = re.split(r"[，,；;]", sentence)
                for clause in clauses[1:]:
                    clause = clause.strip()
                    if clause and not (
                        _sentence_kind(clause) is not None
                        or _REFUSAL_CONTINUATION.match(clause)
                    ):
                        return None
            kinds.add(kind)
        elif not (_SCOPE_HELP.fullmatch(sentence) or _NEUTRAL.fullmatch(sentence)):
            return None
    for kind in ("out_of_scope", "insufficient_evidence", "needs_clarification"):
        if kind in kinds:
            return kind
    return None


def has_evidence_gap(text: str) -> bool:
    """Recognize a dedicated gap sentence, not incidental tutorial vocabulary."""
    return any(
        _EVIDENCE_REFUSAL.fullmatch(_DECORATION.sub("", sentence)) is not None
        for sentence in _sentences(text)
    )


def has_scope_refusal(text: str) -> bool:
    """A scope refusal mixed with substantive prose must not become success."""
    return any(
        _SCOPE_REFUSAL.fullmatch(clause.strip()) is not None
        for sentence in _sentences(text)
        for clause in (sentence, *re.split(r"[，,；;]", sentence))
    )
