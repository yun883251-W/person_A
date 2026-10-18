"""Bounded task planning; execution and budgets belong to the owning Agent.

The planner may describe two or three requested tasks, but cannot call a tool or
grant permission to one. MCP steps copy their operational goal from the current
user request. Questionnaire controls remain in the deterministic MBTI session.
Only the public execution summary may be persisted; it may include a literal
user-topic binding, but no full tool payload, intermediate answers, model
reasoning, or questionnaire answers.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from browser_mcp.contracts import SEARCH_BILIBILI_TOOL
from knowledge_agent.agent.answering.diagnostics import validate_public_answer_diagnostics
from knowledge_agent.agent.planning.topic_references import TopicBinding, TopicReferenceError, resolve_topic_reference, validate_topic_binding

if TYPE_CHECKING:
    from knowledge_agent.agent.dialogue.intent_types import IntentDecision


@dataclass(frozen=True, slots=True)
class PlanStep:
    step_id: str
    goal: str
    kind: Literal["knowledge", "mcp"]
    depends_on: tuple[str, ...] = ()
    allowed_tool_names: tuple[str, ...] = ()
    # Host-owned classification of the original clause; never model JSON or
    # persisted plan data. Reusing it avoids a second semantic classification.
    intent: IntentDecision | None = None
    # A topic copied from another user clause is not an answer dependency.
    topic_binding: TopicBinding | None = None


@dataclass(frozen=True, slots=True)
class TaskPlan:
    steps: tuple[PlanStep, ...]
    # Runtime provenance only: public/stored plan summaries keep version 1.
    decision_source: Literal["rules", "model", "fallback"] = "rules"
    reason_code: str = "plan_explicit_clauses"


class PlanValidationError(ValueError):
    """Expose a fixed failure code, never a raw model response."""

    def __init__(
        self, code: str, *, reason_code: str | None = None,
        decision_source: Literal["rules", "model", "fallback"] = "rules",
    ) -> None:
        super().__init__(code)
        self.code = code
        self.reason_code = reason_code or code
        self.decision_source = decision_source


_TOOL_NAME = re.compile(r"^mcp__[A-Za-z0-9_-]+__[A-Za-z0-9_-]+$")
_CODE_BLOCK = re.compile(r"```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)")
_QUOTED = re.compile(
    r'''`[^`\n]*`|“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』|"(?:\\.|[^"\\\n])*"|(?<!\w)'(?:\\.|[^'\\\n])*'(?!\w)''',
)
_CLAUSE_SEPARATOR = re.compile(
    r"[，,；;。\n]+|(?:然后|随后|接着|最后|其次|再)(?=\s*\S)|"
    r"\b(?:and\s+then|then|finally|next)\b", re.I,
)
_RAW_NUMBER_PREFIX = re.compile(
    r"^\s*(?:[（(](?:\d+|[一二三四五六七八九十])[）)]|"
    r"\d+[).、．](?!\d)|[一二三四五六七八九十][、.．])\s*",
)
_RAW_ORDER_PREFIX = re.compile(
    r"^\s*(?:首先|先|然后|随后|接着|最后|其次|再|"
    r"(?:first|then|finally|next)\b)\s*", re.I,
)
_PREFIX = re.compile(
    r"^\s*(?:(?:[（(]?\d+[）).、]|[一二三][、.])\s*)?"
    r"(?:(?:请帮我|请你|请|麻烦你|麻烦|帮我|为我|首先|先)\s*|"
    r"(?:please|first)\b\s*)*", re.I,
)
_TASK_VERB = re.compile(
    r"^(?:解释|说明|介绍|讲讲|讲解|回答|比较|对比|分析|检索|搜索|查询|查找|总结|归纳|"
    r"制定|安排|整理|列出|给出|生成|编写|写出|调用|执行|运行|使用|"
    r"给我|给一个|给几个|提供|展示|演示|举例|举个例|举一个例|告诉我|用|帮我|"
    r"根据|基于|结合|参照|参考|对照|沿用|利用|扩展|优化|补充|改进|"
    r"不参考|不依赖|不使用|不基于|不要参考|不要依赖|无需|不用|不必|"
    r"开始|暂停|继续|取消|恢复|退出|重测|重做|重新|"
    r"(?:explain|compare|analy[sz]e|search|find|summari[sz]e|list|"
    r"create|write|call|invoke|execute|run|use|start|pause|resume|cancel)\b)", re.I,
)
_NUMBERED_TASK = re.compile(
    r"(?<![^\s；;，,。:：])(?:[（(](?P<parenthesized>\d+|[一二三四五六七八九十])[）)]|"
    r"(?P<arabic>\d+)[).、．](?!\d)|(?P<chinese>[一二三四五六七八九十])[、.．])\s*",
)
_PLAN_INTRO = re.compile(
    r"^(?:(?:请帮我|请你|请|麻烦你|麻烦|帮我)\s*)?"
    r"(?:完成|执行|处理)(?:以下|下面|这些)(?:任务|步骤|请求)\s*[:：]\s*",
)
_TOOL_VERB = re.compile(
    r"^(?:调用|执行|运行|使用|用|(?:call|invoke|execute|run|use)\b)"
    r"\s*(?:(?:MCP|the)\s+)?(?:(?:工具|tool)\s*)?", re.I,
)
_STATEFUL_TOOLS = frozenset({"get_mbti_questionnaire", "score_mbti_answers"})
_ASSESSMENT_TARGET = re.compile(r"mbti|人格测试|性格测试|人格问卷|性格问卷", re.I)
_STATEFUL_ACTION = re.compile(
    r"^(?:开始|开启|启动|重新|重做|重测|暂停|继续|恢复|退出|结束|取消|"
    r"作答|答题|提交答案|选[择]?\s*[1-5]|(?:我)?(?:要|想)(?:做|测)|"
    r"做(?:个|一下)?\s*(?:mbti|人格测试|性格测试)|测一下|帮我测|"
    r"(?:start|restart|pause|resume|cancel|answer)\b)", re.I,
)
_DISCUSSION_PREFIX = re.compile(
    r"^(?:解释|说明|介绍)(?:一下)?(?:为什么|如何|怎样)|"
    r"^(?:how|why|what)\b", re.I,
)
_BROWSER_SITE = re.compile(
    r"(?<![A-Za-z0-9_])(?:bilibili|bilili|bili|b\s*站|哔哩哔哩)(?![A-Za-z0-9_])", re.I,
)
_BROWSER_TASK_LEAD = re.compile(
    r"^(?:在|去|到|上|打开|进入|访问|启动|用浏览器|"
    r"我想|我需要|我要|请|麻烦|帮|替我|为我|给我|能不能|能否|可不可以|可以|"
    r"搜|查找|查询|找|学习|学|观看|看)", re.I,
)
_SUMMARY_STATUSES = frozenset({"completed", "partial", "failed", "needs_clarification"})
_STEP_STATUSES = frozenset({
    "pending", "success", "insufficient_evidence", "needs_clarification",
    "service_error", "limit_reached", "out_of_scope", "skipped",
})

# A temporal order is not a data dependency. Only these explicit references to
# this request's earlier steps can introduce edges. Quoted/code examples never
# enter this parser; history and a model response are not dependency sources.
_STEP_REFERENCE = re.compile(
    r"(?P<previous>上一步|前一步|前一项|前一个步骤|前一个任务|\bprevious\s+step\b)|"
    r"(?P<count>(?:上述|以上|前面|上面|前)(?:的)?\s*(?P<count_number>[123一二两三])\s*(?:步|个步骤|个任务|项任务))|"
    r"(?P<all>(?:上述|以上|前面|上面)(?:的)?(?:所有|全部|各个|各)(?:步骤|任务|步))|"
    r"(?P<ordinal>第\s*(?P<ordinal_number>[1-9一二三四五六七八九十])\s*(?:个步骤|个任务|个回答|个结果|步|项任务|项)|"
    r"(?<![A-Za-z0-9_])P(?P<plan_number>[1-9])(?![A-Za-z0-9_])|\bstep\s+(?P<english_number>[1-9])\b)",
    re.I,
)
_REFERENCE_USE = re.compile(
    r"(?:根据|基于|结合|沿用|利用|参照|参考|依照|依赖|使用|用|对比|对照|比较|"
    r"总结|归纳|分析|扩展|改进|解释|说明|介绍|评价|优化|补充|把|将)\s*(?:一下)?\s*$|"
    r"\b(?:based\s+on|using|use|compare|summarize|explain|extend|from)\s*$", re.I,
)
_REFERENCE_RESULT = re.compile(
    r"^\s*(?:的)?(?:回答|答案|结果|输出|结论|内容|资料|解释|代码|方案)|"
    r"^\s*(?:'s\s+)?(?:answer|result|output|conclusion)\b", re.I,
)
_REFERENCE_JOIN = re.compile(r"(?:和|与|及|、|以及|\band\b)\s*$", re.I)
_REFERENCE_NEGATION = re.compile(
    r"(?:不要|无需|不用|不必|不需要|不能|不应|别|不|避免|禁止)\s*"
    r"(?:(?:再|继续|直接|重复)\s*)?(?:根据|基于|结合|引用|沿用|利用|参照|参考|依照|依赖|使用|用|总结|比较)?(?:于)?\s*$|"
    r"(?:独立于|脱离|忽略|跳过)\s*$|"
    r"\b(?:without|do\s+not|don't|not)\s*(?:using|use|rely(?:ing)?\s+on|refer(?:ring)?\s+to)?\s*$",
    re.I,
)
_REFERENCE_NEGATIVE_SUFFIX = re.compile(r"^\s*(?:无关|没有关系|不相关)")
_AMBIGUOUS_REFERENCE = re.compile(
    r"上述|上文|前文|前面(?:的)?(?:内容|回答|结果|结论)|上面(?:的)?(?:内容|回答|结果|结论)|"
    r"(?:这个|那个|这些|那些|该)(?:概念|内容|结果|答案|回答|结论|方案)|"
    r"(?:根据|基于|结合|使用|用|总结|比较|分析|解释|说明|介绍)\s*(?:它们|它|这两者|两者)|"
    r"\b(?:those\s+results|that\s+answer|the\s+above|using\s+it)\b", re.I,
)


def _operational_text(question: str) -> str:
    """Ignore quoted examples while retaining a formatted single identifier.

    A quoted tool name or parameter such as `INTJ` is data in an outer command;
    an entire quoted command never contributes an action or task separator.
    """
    def unquote_identifier(match: re.Match[str]) -> str:
        inner = match.group(0)[1:-1]
        return inner if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", inner) else " "

    return _QUOTED.sub(unquote_identifier, _CODE_BLOCK.sub(" ", question))


def _is_browser_task_clause(clause: str) -> bool:
    """Select a possible site action for classification, without authorizing it."""
    text = _PREFIX.sub("", _operational_text(clause)).strip()
    return bool(_BROWSER_SITE.search(text) and _BROWSER_TASK_LEAD.match(text))


def _strip_plan_lead(clause: str) -> str:
    text = _RAW_NUMBER_PREFIX.sub("", clause.strip())
    return _RAW_ORDER_PREFIX.sub("", text).strip()


def _starts_task(clause: str) -> bool:
    text = _PREFIX.sub("", _operational_text(_strip_plan_lead(clause))).strip()
    return bool(_TASK_VERB.match(text)) or _is_browser_task_clause(text)


def _masked_plan_text(question: str) -> str:
    masked = _CODE_BLOCK.sub(lambda match: " " * len(match.group(0)), question)
    return _QUOTED.sub(lambda match: " " * len(match.group(0)), masked)


def _numbered_parts(question: str) -> tuple[tuple[int, ...], tuple[str, ...]] | None:
    """Read a whole numbered request, never a numbered example inside a question."""
    matches = list(_NUMBERED_TASK.finditer(_masked_plan_text(question)))
    if len(matches) < 2:
        return None
    lead = question[:matches[0].start()].strip()
    if lead and _PLAN_INTRO.fullmatch(lead) is None:
        return None
    ordinals: list[int] = []
    clauses: list[str] = []
    for index, match in enumerate(matches):
        raw = match.group("parenthesized") or match.group("arabic") or match.group("chinese")
        ordinal = int(raw) if raw.isdecimal() else "一二三四五六七八九十".index(raw) + 1
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        clause = question[match.end():end].strip().rstrip("；;，,。\n").strip()
        ordinals.append(ordinal)
        clauses.append(_strip_plan_lead(clause))
    return tuple(ordinals), tuple(clauses)


def raw_plan_clauses(question: str) -> tuple[str, ...]:
    """Split task boundaries while retaining exact quoted titles and arguments.

    Quotes and code are masked with equal-length spaces only while locating
    separators. The returned strings always come from the user's original
    input. Commas inside an argument description do not create a new task.
    """
    if not isinstance(question, str) or not question.strip():
        return ()
    numbered = _numbered_parts(question)
    if numbered is not None:
        return numbered[1]
    masked = _masked_plan_text(question)
    result: list[str] = []
    start = 0
    for match in _CLAUSE_SEPARATOR.finditer(masked):
        if not _starts_task(question[match.end():]):
            continue
        part = _strip_plan_lead(question[start:match.start()])
        if part and _operational_text(part).strip():
            result.append(part)
        start = match.end()
    last = _strip_plan_lead(question[start:])
    if last and _operational_text(last).strip():
        result.append(last)
    return tuple(result)


def _clauses(question: str) -> tuple[str, ...]:
    return tuple(
        cleaned for part in raw_plan_clauses(question)
        if (cleaned := _PREFIX.sub("", _operational_text(part)).strip())
    )


def should_plan(question: str) -> bool:
    """Choose planning only for explicitly separated, imperative task requests."""
    if not isinstance(question, str) or not question.strip():
        return False
    text = _operational_text(question).strip()
    numbered = _numbered_parts(question)
    if numbered is not None:
        return all(_starts_task(clause) for clause in numbered[1])
    starts_ordered = re.match(r"^(?:(?:请帮我|请你|请|麻烦|帮我)\s*)?(?:首先|先|first\b)", text, re.I)
    if not starts_ordered and _DISCUSSION_PREFIX.match(_PREFIX.sub("", text)):
        return False
    if not re.search(r"然后|随后|接着|最后|其次|再|\b(?:then|finally|next)\b", text, re.I):
        return False
    if not starts_ordered and not re.search(r"[，,；;。\n]", text):
        return False
    clauses = raw_plan_clauses(question)
    return len(clauses) >= 2 and all(_starts_task(clause) for clause in clauses)


def _reference_is_negated(prefix: str) -> bool:
    """Recognize a local negative qualifier, never a negation in another clause."""
    segment = re.split(r"[，,；;。\n]", prefix)[-1]
    if re.search(r"(?:不要|不能|不应|别)\s*(?:不|忽略|跳过)", segment):
        raise PlanValidationError("plan_ambiguous_dependency")
    return _REFERENCE_NEGATION.search(segment) is not None


def _reference_number(value: str) -> int:
    if value.isdecimal():
        return int(value)
    return 2 if value == "两" else "一二三四五六七八九十".index(value) + 1


def _explicit_dependencies(clause: str, index: int) -> tuple[str, ...]:
    """Build a stable graph from affirmative references in the current clause.

    Generic references such as "它" or "上述内容" are not silently attached to
    the previous task. An ordinal describing a topic ("检索的第一步") is also
    not a plan edge unless it is used as an earlier task/result reference.
    """
    text = _masked_plan_text(clause)
    masked = list(text)
    dependencies: set[int] = set()
    previous_reference_end: int | None = None
    previous_reference_negated = False
    for match in _STEP_REFERENCE.finditer(text):
        prefix = text[:match.start()]
        joined = (
            previous_reference_end is not None
            and _REFERENCE_JOIN.fullmatch(text[previous_reference_end:match.start()].strip()) is not None
        )
        is_reference = (
            match.group("ordinal") is None
            or _REFERENCE_USE.search(prefix) is not None
            or _REFERENCE_RESULT.match(text[match.end():]) is not None
            or joined
        )
        if not is_reference:
            continue
        masked[match.start():match.end()] = " " * (match.end() - match.start())
        negated = (
            _reference_is_negated(prefix)
            or _REFERENCE_NEGATIVE_SUFFIX.match(text[match.end():]) is not None
            or (joined and previous_reference_negated)
        )
        previous_reference_end = match.end()
        previous_reference_negated = negated
        if negated:
            continue
        if match.group("previous"):
            numbers = (index - 1,)
        elif match.group("count"):
            count = _reference_number(match.group("count_number"))
            # "前两步" means the preceding two tasks, not an arbitrary pair.
            numbers = tuple(range(index - count, index))
        elif match.group("all"):
            numbers = tuple(range(1, index))
        else:
            raw = match.group("ordinal_number") or match.group("plan_number") or match.group("english_number")
            numbers = (_reference_number(raw),)
        if not numbers or any(number < 1 or number >= index for number in numbers):
            raise PlanValidationError("plan_invalid_dependency_reference")
        dependencies.update(numbers)
    remaining = "".join(masked)
    for match in _AMBIGUOUS_REFERENCE.finditer(remaining):
        if not _reference_is_negated(remaining[:match.start()]):
            raise PlanValidationError("plan_ambiguous_dependency")
    return tuple(f"P{number}" for number in sorted(dependencies))


def _value(item: object, field: str) -> object | None:
    return item.get(field) if isinstance(item, Mapping) else getattr(item, field, None)


def _capabilities(available_tools: Sequence[object]) -> tuple[dict[str, str], ...]:
    if isinstance(available_tools, (str, bytes)) or not isinstance(available_tools, Sequence):
        raise PlanValidationError("plan_invalid_capabilities")
    result: dict[str, dict[str, str]] = {}
    for tool in available_tools:
        name = _value(tool, "name")
        if not isinstance(name, str) or len(name) > 160 or not _TOOL_NAME.fullmatch(name):
            continue
        description = _value(tool, "description")
        if not isinstance(description, str):
            description = _value(_value(tool, "definition"), "description")
        result[name] = {"name": name, "description": description[:800] if isinstance(description, str) else ""}
    return tuple(result.values())


def authorized_tool_names(question: str, available_tools: Sequence[object]) -> tuple[str, ...]:
    """Require an explicit invocation verb and an exact, unambiguous tool name.

    For example ``调用 get_mbti_description`` authorizes a unique short name;
    ``如何调用 get_mbti_description`` and a name mentioned in code authorize none.
    A model, a history message, or a tool description cannot expand this set.
    """
    if not isinstance(question, str) or not question.strip():
        return ()
    capabilities = _capabilities(available_tools)
    short_names: dict[str, list[str]] = {}
    for capability in capabilities:
        name = capability["name"]
        short_names.setdefault(name.split("__", 2)[2], []).append(name)
    requested: set[str] = set()
    for clause in _clauses(question):
        verb = _TOOL_VERB.match(clause)
        if verb is None:
            continue
        target = clause[verb.end():].lstrip(" `\"'“‘")
        for capability in capabilities:
            name = capability["name"]
            short = name.split("__", 2)[2]
            aliases = (name, short) if len(short_names[short]) == 1 else (name,)
            for alias in aliases:
                matched = re.match(re.escape(alias) + r"(?![A-Za-z0-9_-])", target)
                if matched is None:
                    continue
                suffix = target[matched.end():].strip(" `\"'“”‘’。？！?!")
                if re.fullmatch(r"(?:工具)?(?:的)?(?:怎么用|如何使用|用法|原理是什么|是什么|是怎么实现的)", suffix):
                    continue
                requested.add(name)
    return tuple(capability["name"] for capability in capabilities if capability["name"] in requested)


def _has_stateful_assessment(question: str) -> bool:
    for clause in _clauses(question):
        if clause.casefold().startswith("mbti答题 "):
            return True
        if _ASSESSMENT_TARGET.search(clause) and _STATEFUL_ACTION.match(clause):
            return True
        if (_ASSESSMENT_TARGET.search(question) and _STATEFUL_ACTION.match(clause)
                and (re.search(r"问卷|答题", clause)
                     or re.fullmatch(r"(?:暂停|继续|恢复|退出|取消)(?:测试)?[。.!！]?", clause))):
            return True
    return False


def plan_response_schema() -> dict[str, Any]:
    """Retain the previous public schema helper for external integrations.

    The bounded Agent path now constructs its plan from explicit user clauses
    and does not generate a dependency graph with a model.
    """
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "step_id": {"type": "string", "enum": ["P1", "P2", "P3"]},
                        "depends_on": {
                            "type": "array", "maxItems": 2,
                            "items": {"type": "string", "enum": ["P1", "P2"]},
                        },
                    },
                    "required": ["step_id", "depends_on"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["steps"],
        "additionalProperties": False,
    }


def _validate_goal(goal: object) -> str:
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 2000:
        raise PlanValidationError("plan_invalid_goal")
    if any(ord(char) < 32 and char not in "\n\t" for char in goal):
        raise PlanValidationError("plan_invalid_goal")
    return goal.strip()


def _validate_identity(item: dict[str, Any], index: int) -> tuple[str, str, str, tuple[str, ...]]:
    if item.get("step_id") != f"P{index}":
        raise PlanValidationError("plan_invalid_step_id")
    goal = _validate_goal(item.get("goal"))
    kind = item.get("kind")
    if not isinstance(kind, str) or kind not in {"knowledge", "mcp"}:
        raise PlanValidationError("plan_invalid_kind")
    dependencies = item.get("depends_on")
    previous = {f"P{number}" for number in range(1, index)}
    if (not isinstance(dependencies, list) or any(not isinstance(dep, str) for dep in dependencies)
            or len(dependencies) != len(set(dependencies)) or not set(dependencies) <= previous):
        raise PlanValidationError("plan_invalid_dependencies")
    return item["step_id"], goal, kind, tuple(dependencies)


def _clause_authorizations(
    clauses: tuple[str, ...], capabilities: tuple[dict[str, str], ...],
    resolver: Callable[[str], IntentDecision] | None,
    topic_bindings: Mapping[str, TopicBinding] | None = None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, IntentDecision]]:
    """Classify only possible browser actions, never ordinary knowledge clauses.

    Exact tool-name invocations retain their deterministic authorization path.
    The optional resolver belongs to runner, including all classification costs.
    Neither a planner response nor an earlier answer can produce authorization.
    """
    from knowledge_agent.agent.dialogue.intent_types import IntentDecision, IntentKind

    available = {item["name"] for item in capabilities}
    authorized: dict[str, tuple[str, ...]] = {}
    resolved: dict[str, IntentDecision] = {}
    for clause in clauses:
        names = authorized_tool_names(clause, capabilities)
        binding = (topic_bindings or {}).get(clause)
        if binding is not None or _is_browser_task_clause(clause):
            if binding is not None:
                if not validate_topic_binding(binding, clauses):
                    raise PlanValidationError("plan_topic_reference_invalid")
                if SEARCH_BILIBILI_TOOL not in available:
                    raise PlanValidationError("browser_bilibili_unavailable")
                decision = IntentDecision(
                    kind=IntentKind.MCP_TOOL,
                    allowed_tool_names=(SEARCH_BILIBILI_TOOL,),
                    authorized_user_input=clause,
                    reason_code="browser_bilibili_topic_reference",
                )
            else:
                if resolver is None:
                    raise PlanValidationError("plan_clause_intent_unavailable")
                decision = resolver(clause)
            if not isinstance(decision, IntentDecision):
                raise PlanValidationError("plan_invalid_clause_intent")
            if decision.kind is IntentKind.NEEDS_CLARIFICATION:
                if decision.decision_source == "fallback" or decision.reason_code == "browser_bilibili_unavailable":
                    # A timeout, invalid response or service failure is not an
                    # unclear user request. Preserve the actual diagnostic.
                    raise PlanValidationError(
                        decision.reason_code or "plan_clause_intent_failed",
                        decision_source=decision.decision_source,
                    )
                raise PlanValidationError(
                    "plan_clause_needs_clarification",
                    reason_code=decision.reason_code,
                    decision_source=decision.decision_source,
                )
            if decision.kind is IntentKind.KNOWLEDGE:
                # Mentioning a site in an explanatory request grants no tool
                # permission, but should not reject the other knowledge tasks.
                names = ()
            elif decision.kind is not IntentKind.MCP_TOOL:
                # A discussion, negative command or unresolved target must not
                # be promoted to execution by the later planning model.
                raise PlanValidationError("plan_unconfirmed_clause_action")
            else:
                names = decision.allowed_tool_names
                if (names != (SEARCH_BILIBILI_TOOL,)
                        or decision.authorized_user_input != clause
                        or not set(names) <= available):
                    raise PlanValidationError("plan_unauthorized_clause_tool")
            resolved[clause] = decision
        text = _PREFIX.sub("", _operational_text(clause)).strip()
        if (re.match(r"^(?:调用|执行|运行|(?:call|invoke|execute|run)\b)", text, re.I)
                and _TOOL_VERB.match(text) is not None and not names):
            raise PlanValidationError("plan_unavailable_tool")
        if any(name.split("__", 2)[2] in _STATEFUL_TOOLS for name in names):
            raise PlanValidationError("plan_stateful_assessment")
        authorized[clause] = names
    return authorized, resolved


def build_plan(
    question: str, *, context: Sequence[object] = (),
    available_tools: Sequence[object] = (), model_invoker: Callable[..., object] | None = None,
    max_steps: int = 3, max_context_chars: int = 24000,
    clause_intent_resolver: Callable[[str], IntentDecision] | None = None,
) -> TaskPlan:
    """Construct an explicit bounded plan without a model-generated graph.

    The compatibility parameters ``context`` and ``model_invoker`` cannot
    influence this graph. Browser clause classification can still call its
    resolver; that caller owns the exact model budget and error provenance.
    """
    if not isinstance(question, str) or not question.strip():
        raise PlanValidationError("plan_invalid_question")
    if type(max_steps) is not int or not 2 <= max_steps <= 3:
        raise PlanValidationError("plan_invalid_step_limit")
    if type(max_context_chars) is not int or not 1 <= max_context_chars <= 200000:
        raise PlanValidationError("plan_invalid_context_limit")
    if model_invoker is not None and not callable(model_invoker):
        raise PlanValidationError("plan_invalid_invoker")
    if clause_intent_resolver is not None and not callable(clause_intent_resolver):
        raise PlanValidationError("plan_invalid_clause_resolver")
    if not should_plan(question):
        raise PlanValidationError("plan_not_requested")
    capabilities = _capabilities(available_tools)
    clauses = raw_plan_clauses(question)
    numbered = _numbered_parts(question)
    if numbered is not None and numbered[0] != tuple(range(1, len(clauses) + 1)):
        raise PlanValidationError("plan_invalid_numbering")
    if not 2 <= len(clauses) <= max_steps or len(set(clauses)) != len(clauses):
        raise PlanValidationError("plan_invalid_step_count")
    for clause in clauses:
        try:
            _validate_goal(clause)
        except PlanValidationError:
            raise PlanValidationError("plan_invalid_request_goal") from None
    if _has_stateful_assessment(question):
        raise PlanValidationError("plan_stateful_assessment")
    if len(question) > max_context_chars:
        raise PlanValidationError("plan_context_too_long")
    topic_bindings: dict[str, TopicBinding] = {}
    for index, clause in enumerate(clauses):
        try:
            binding = resolve_topic_reference(clause, clauses[:index])
        except TopicReferenceError as exc:
            raise PlanValidationError(exc.code) from exc
        if binding is not None:
            topic_bindings[clause] = binding
    dependencies_by_step = tuple(
        () if clause in topic_bindings else _explicit_dependencies(clause, index)
        for index, clause in enumerate(clauses, 1)
    )
    clause_tools, clause_intents = _clause_authorizations(
        clauses, capabilities, clause_intent_resolver, topic_bindings,
    )
    # Reject duplicate tool execution before any planned action is dispatched.
    requested_names = [name for names in clause_tools.values() for name in names]
    if len(requested_names) != len(set(requested_names)):
        raise PlanValidationError("plan_repeated_tool")
    steps: list[PlanStep] = []
    for index, goal in enumerate(clauses, 1):
        names = clause_tools[goal]
        dependencies = dependencies_by_step[index - 1]
        if names and dependencies:
            # Tool arguments may use a bound user topic, never another answer.
            raise PlanValidationError("plan_tool_dependency_unsupported")
        if not names and any(steps[int(key[1:]) - 1].kind != "knowledge" for key in dependencies):
            # A browser receipt / MCP answer is not a source document. Refuse
            # this unsupported handoff before executing the external action.
            raise PlanValidationError("plan_knowledge_tool_dependency_unsupported")
        steps.append(PlanStep(
            f"P{index}", goal, "mcp" if names else "knowledge",
            dependencies, tuple(names), clause_intents.get(goal), topic_bindings.get(goal),
        ))
    return TaskPlan(tuple(steps))


def validate_plan_summary(value: object) -> dict[str, Any]:
    """Validate and copy the sole plan shape exposed by API, CLI and storage."""
    if not isinstance(value, dict) or set(value) != {"version", "status", "steps"}:
        raise PlanValidationError("plan_invalid_summary")
    if type(value["version"]) is not int or value["version"] != 1:
        raise PlanValidationError("plan_invalid_summary_version")
    status = value["status"]
    if not isinstance(status, str) or status not in _SUMMARY_STATUSES:
        raise PlanValidationError("plan_invalid_summary_status")
    items = value["steps"]
    if (not isinstance(items, list) or len(items) > 3
            or not items and status not in {"failed", "needs_clarification"}):
        raise PlanValidationError("plan_invalid_step_count")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        required_fields = {
            "step_id", "goal", "kind", "depends_on", "status", "model_calls", "tool_calls", "stop_reason",
        }
        if (not isinstance(item, dict) or not required_fields <= set(item)
                or set(item) - required_fields - {"answer_diagnostics", "topic_binding"}):
            raise PlanValidationError("plan_invalid_summary_step")
        step_id, goal, kind, dependencies = _validate_identity(item, index)
        step_status = item["status"]
        if not isinstance(step_status, str) or step_status not in _STEP_STATUSES:
            raise PlanValidationError("plan_invalid_summary_status")
        for count in (item["model_calls"], item["tool_calls"]):
            if type(count) is not int or count < 0:
                raise PlanValidationError("plan_invalid_summary_count")
        reason = item["stop_reason"]
        if reason is not None and (not isinstance(reason, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,159}", reason)):
            raise PlanValidationError("plan_invalid_summary_reason")
        result.append({
            "step_id": step_id, "goal": goal, "kind": kind, "depends_on": list(dependencies),
            "status": step_status, "model_calls": item["model_calls"], "tool_calls": item["tool_calls"],
            "stop_reason": reason,
        })
        if "answer_diagnostics" in item:
            try:
                result[-1]["answer_diagnostics"] = validate_public_answer_diagnostics(item["answer_diagnostics"])
            except (TypeError, ValueError) as exc:
                raise PlanValidationError("plan_invalid_summary_diagnostics") from exc
        if item.get("topic_binding") is not None:
            binding = item["topic_binding"]
            if (kind != "mcp" or dependencies or not isinstance(binding, dict)
                    or set(binding) != {"topic", "source_step_id"}
                    or not isinstance(binding["topic"], str)
                    or not isinstance(binding["source_step_id"], str)):
                raise PlanValidationError("plan_invalid_summary_topic_binding")
            source = next((prior for prior in result[:-1] if prior["step_id"] == binding["source_step_id"]), None)
            if source is None or source["kind"] != "knowledge":
                raise PlanValidationError("plan_invalid_summary_topic_binding")
            candidate = TopicBinding(goal, binding["topic"], binding["source_step_id"], source["goal"])
            if not validate_topic_binding(candidate, tuple(prior["goal"] for prior in result)):
                raise PlanValidationError("plan_invalid_summary_topic_binding")
            result[-1]["topic_binding"] = dict(binding)
    return {"version": 1, "status": status, "steps": result}
