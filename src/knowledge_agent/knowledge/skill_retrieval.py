"""Build and execute bounded, skill-aware knowledge retrieval strategies.

This module performs deterministic text analysis only. It does not execute user
code, load executable behavior from skill text, alter retriever configuration,
or generate a final answer. The owning runner supplies turn-scoped executors and
state so strategy searches share the normal tool, follow-up, and context budget.
"""

from __future__ import annotations

import ast
import re
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from ..schemas import RetrievalResult
from .code_analysis import extract_visible_code


RetrievalOperation = Literal["search", "read_section"]
RetrievalStepStatus = Literal["executed", "skipped", "error"]
RetrievalStrategy = Literal["default", "concept", "comparison", "code"]

SKILL_STRATEGIES: Mapping[str, RetrievalStrategy] = {
    "concept-explanation": "concept",
    "concept-comparison": "comparison",
    "code-explanation": "code",
}


@dataclass(frozen=True, slots=True)
class RetrievalSubject:
    """A bounded retrieval object and the visible input it came from."""

    text: str
    source: Literal["question", "visible_context"]
    kind: Literal["concept", "comparison", "code"]


@dataclass(frozen=True, slots=True)
class RetrievalAnalysis:
    """Deterministic analysis used to build a strategy plan."""

    subjects: tuple[RetrievalSubject, ...]
    focus: str
    code_identifiers: tuple[str, ...] = ()
    clarification_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalStep:
    """One planned knowledge-tool operation."""

    step_id: str
    operation: RetrievalOperation
    target: str | tuple[str, ...]
    required: bool
    query: str | None = None
    anchor_id: str | None = None

    @property
    def targets(self) -> tuple[str, ...]:
        return (self.target,) if isinstance(self.target, str) else self.target

    def __post_init__(self) -> None:
        if (
            not self.step_id.strip()
            or not self.targets
            or any(not value.strip() for value in self.targets)
        ):
            raise ValueError("retrieval steps require an ID and non-empty targets")
        if self.operation == "search":
            if not self.query or self.anchor_id is not None:
                raise ValueError("search steps require query and no anchor_id")
        elif self.operation == "read_section":
            if not self.anchor_id or self.query is not None:
                raise ValueError("read_section steps require anchor_id and no query")
        else:
            raise ValueError("unsupported retrieval operation")


@dataclass(frozen=True, slots=True)
class SkillRetrievalPlan:
    """Immutable retrieval strategy selected for one knowledge step."""

    strategy_id: RetrievalStrategy
    selected_skill_id: str | None
    steps: tuple[RetrievalStep, ...]
    coverage_targets: tuple[str, ...]
    reason_code: str


@dataclass(frozen=True, slots=True)
class RetrievalStepResult:
    """Execution result for one planned operation."""

    status: RetrievalStepStatus
    returned_chunk_ids: tuple[str, ...] = ()
    retained_chunk_ids: tuple[str, ...] = ()
    stop_reason: str | None = None
    step_id: str = ""


@dataclass(frozen=True, slots=True)
class SkillRetrievalOutcome:
    """Aggregate execution, coverage, and truncation information."""

    executed: int = 0
    skipped: int = 0
    errors: int = 0
    missing_targets: tuple[str, ...] = ()
    budget_limited: bool = False
    strategy_id: RetrievalStrategy = "default"
    step_results: tuple[RetrievalStepResult, ...] = ()
    index_version: str | None = None
    retained_chunk_ids: tuple[str, ...] = ()
    target_chunk_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    query_attempted_targets: tuple[str, ...] = ()

    @property
    def error(self) -> int:
        return self.errors

    @property
    def coverage_missing(self) -> tuple[str, ...]:
        return self.missing_targets


class RetrievalExecutionState(Protocol):
    """Minimal mutable turn state needed by the strategy executor."""

    evidence: MutableMapping[str, RetrievalResult]
    tool_calls: int
    followup_searches: int


ToolResult = Mapping[str, Any]
SearchExecutor = Callable[[str], ToolResult]
SectionExecutor = Callable[[str], ToolResult]

_PATH_ONLY = re.compile(r"^\s*(?:[A-Za-z]:)?[^\n]+\.(?:py|pyi)\s*$", re.I)
_COMPARISON_PATTERNS = (
    re.compile(
        r"(?:比较|对比)\s*([^，。；?？]{1,80}?)\s*"
        r"(?:和|与|及|、|跟)\s*([^，。；?？]{1,80})"
    ),
    re.compile(
        r"([^，。；?？]{1,80}?)\s*(?:和|与|跟|相比于)\s*"
        r"([^，。；?？]{1,80}?)\s*(?:的)?"
        r"(?:区别|差异|异同|哪个好|如何选择)"
    ),
    re.compile(
        r"\b([A-Za-z_][\w. -]{0,60}?)\s+(?:vs\.?|versus)\s+"
        r"([A-Za-z_][\w. -]{0,60}?)(?:[?.!,;]|$)",
        re.I,
    ),
)
_QUESTION_NOISE = re.compile(
    r"^(?:请|帮我|解释|说明|介绍|讲解|什么是|如何理解|"
    r"what is|explain|describe)\s*",
    re.I,
)
_FOCUS_RULES = (
    (re.compile(r"为什么|用途|作用|场景|why|use case", re.I), "用途与适用场景"),
    (re.compile(r"怎么|如何|流程|过程|工作原理|how|process", re.I), "工作过程"),
    (
        re.compile(r"区别|差异|比较|选择|vs\.?|versus", re.I),
        "核心差异与适用场景",
    ),
    (
        re.compile(r"参数|返回|输入|输出|调用|parameter|return", re.I),
        "调用、输入与输出",
    ),
)


def strategy_for_skill(selected_skill_id: str | None) -> RetrievalStrategy:
    """Map only built-in skill IDs to specialized strategies."""

    return SKILL_STRATEGIES.get(selected_skill_id or "", "default")


def _clean_subject(value: str) -> str:
    value = value.strip(" \t\r\n`'\"“”‘’()（）[]【】,，.。?？:：;；")
    return re.sub(r"\s+", " ", value)[:120]


def _focus(question: str) -> str:
    for pattern, label in _FOCUS_RULES:
        if pattern.search(question):
            return label
    return "定义与核心机制"


def _comparison_subjects(question: str) -> tuple[str, ...]:
    for pattern in _COMPARISON_PATTERNS:
        match = pattern.search(question)
        if match is None:
            continue
        values = tuple(_clean_subject(re.sub(
            r"(?:的)?(?:区别|差异|异同|优缺点|适用场景|如何选择|哪个好)(?:是什么|有哪些|是什么呢)?$",
            "", _clean_subject(value),
        )) for value in match.groups())
        if (
            len(values) == 2
            and all(values)
            and values[0].casefold() != values[1].casefold()
        ):
            return values
    return ()


def _visible_code(question: str, visible_context: str) -> tuple[str, str]:
    return extract_visible_code(question, visible_context)


def _code_identifiers(code: str, question: str = "") -> tuple[str, ...]:
    if not code:
        return ()
    identifiers: list[str] = []

    def add(value: str) -> None:
        if value and value not in identifiers and not value.startswith("_"):
            identifiers.append(value[:120])

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        candidates = re.findall(
            r"\b[a-zA-Z_]\w*(?:\.\w+)*(?=\s*\()|\b[A-Z][A-Za-z0-9_]*\b|\b[a-zA-Z_]\w*(?:\.\w+)+",
            code,
        )
        for value in candidates:
            add(value)
            if len(identifiers) >= 6:
                break
        return tuple(identifiers)

    def dotted(node: ast.AST) -> str:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    nodes = list(ast.walk(tree))
    aliases: dict[str, str] = {}
    imported: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
                imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name != "*":
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                    imported.append(f"{node.module}.{alias.name}")
    # Actual calls come before unrelated imports. A bounded visible question can
    # then prioritize a named call without treating source text as instructions.
    for node in nodes:
        if isinstance(node, ast.Call):
            name = dotted(node.func)
            root, separator, rest = name.partition(".")
            add(aliases.get(root, root) + (separator + rest if separator else ""))
    for node in nodes:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name)
    for name in imported:
        add(name)
    focus_lines = [line for line in question.splitlines() if re.search(r"解释|解读|分析|为什么|如何|explain|why|how", line, re.I)]
    focus = " ".join(focus_lines)[:2000]
    identifiers.sort(key=lambda value: 0 if re.search(r"(?<![\w])" + re.escape(value.split(".")[-1]) + r"(?![\w])", focus, re.I) else 1)
    return tuple(identifiers[:6])


def analyze_retrieval_question(
    question: str,
    *,
    strategy_id: RetrievalStrategy,
    visible_context: str = "",
) -> RetrievalAnalysis:
    """Extract bounded subjects from a disambiguated question and visible code."""

    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be blank")
    focus = _focus(cleaned)

    if strategy_id == "comparison":
        objects = _comparison_subjects(cleaned)
        if len(objects) != 2:
            return RetrievalAnalysis(
                (), focus, clarification_reason="comparison_objects_unclear"
            )
        return RetrievalAnalysis(
            tuple(
                RetrievalSubject(value, "question", "comparison")
                for value in objects
            ),
            focus,
        )

    if strategy_id == "code":
        code, source = _visible_code(cleaned, visible_context)
        if not code:
            reason = (
                "code_path_without_visible_code"
                if _PATH_ONLY.fullmatch(cleaned)
                else "visible_code_missing"
            )
            return RetrievalAnalysis((), focus, clarification_reason=reason)
        identifiers = _code_identifiers(code, cleaned)
        if not identifiers:
            return RetrievalAnalysis(
                (), focus, clarification_reason="code_identifiers_missing"
            )
        return RetrievalAnalysis(
            (RetrievalSubject(identifiers[0], source, "code"),),
            focus,
            identifiers,
        )

    concept = _clean_subject(_QUESTION_NOISE.sub("", cleaned))
    if not concept:
        return RetrievalAnalysis((), focus, clarification_reason="concept_unclear")
    return RetrievalAnalysis(
        (RetrievalSubject(concept, "question", "concept"),),
        focus,
    )


def build_retrieval_plan(
    retrieval_question: str,
    *,
    selected_skill_id: str | None,
    remaining_searches: int,
    visible_context: str = "",
) -> SkillRetrievalPlan:
    """Create a deterministic plan without performing retrieval."""

    if remaining_searches < 0:
        raise ValueError("remaining_searches must not be negative")
    question = retrieval_question.strip()
    if not question:
        raise ValueError("retrieval_question must not be blank")

    strategy = strategy_for_skill(selected_skill_id)
    if strategy == "default":
        return SkillRetrievalPlan(
            strategy,
            selected_skill_id,
            (RetrievalStep("search-1", "search", "question", True, query=question),),
            ("question",),
            "default_search",
        )

    analysis = analyze_retrieval_question(
        question,
        strategy_id=strategy,
        visible_context=visible_context,
    )
    if analysis.clarification_reason:
        return SkillRetrievalPlan(
            strategy,
            selected_skill_id,
            (),
            (),
            analysis.clarification_reason,
        )

    if strategy == "concept":
        concept = analysis.subjects[0].text
        steps = [
            RetrievalStep(
                "search-concept",
                "search",
                "concept_core",
                True,
                query=f"{concept} {analysis.focus}",
            )
        ]
        if remaining_searches > 1 and analysis.focus == "定义与核心机制":
            steps.append(
                RetrievalStep(
                    "search-concept-application",
                    "search",
                    "concept_application",
                    False,
                    query=f"{concept} 用途 适用场景 示例",
                )
            )
        targets = tuple(target for step in steps for target in step.targets)
        return SkillRetrievalPlan(
            strategy,
            selected_skill_id,
            tuple(steps),
            targets,
            "concept_queries",
        )

    if strategy == "comparison":
        left, right = (subject.text for subject in analysis.subjects)
        targets = (f"object:{left}", f"object:{right}")
        if remaining_searches < 2:
            steps = (
                RetrievalStep(
                    "search-comparison-joint",
                    "search",
                    targets,
                    True,
                    query=f"{left} {right} {analysis.focus}",
                ),
            )
            reason = "comparison_joint_budget_limited"
        else:
            steps = (
                RetrievalStep(
                    "search-comparison-a",
                    "search",
                    targets[0],
                    True,
                    query=f"{left} {analysis.focus}",
                ),
                RetrievalStep(
                    "search-comparison-b",
                    "search",
                    targets[1],
                    True,
                    query=f"{right} {analysis.focus}",
                ),
            )
            reason = "comparison_separate_queries"
        return SkillRetrievalPlan(
            strategy,
            selected_skill_id,
            steps,
            targets,
            reason,
        )

    identifiers = analysis.code_identifiers[:2]
    allowed = max(1, remaining_searches)
    steps = tuple(
        RetrievalStep(
            f"search-code-{number}",
            "search",
            f"api:{identifier}",
            number == 1,
            query=f"{identifier} {analysis.focus}",
        )
        for number, identifier in enumerate(identifiers[:allowed], start=1)
    )
    return SkillRetrievalPlan(
        strategy,
        selected_skill_id,
        steps,
        tuple(target for step in steps for target in step.targets),
        "code_identifier_queries",
    )


def _chunk_ids(result: ToolResult, key: str) -> tuple[str, ...]:
    raw = result.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    found: list[str] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        chunk_id = item.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id not in found:
            found.append(chunk_id)
    return tuple(found)


def _error_code(result: ToolResult) -> str:
    error = result.get("error")
    if isinstance(error, Mapping) and isinstance(error.get("code"), str):
        return str(error["code"])
    return "knowledge_tool_failed"


def _retained_ids(
    target_hits: Mapping[str, Sequence[str]],
    evidence: Mapping[str, RetrievalResult],
    *,
    max_context_chars: int,
) -> tuple[str, ...]:
    """Round-robin targets while keeping only complete chunks that fit."""

    queues = [deque(chunk_ids) for chunk_ids in target_hits.values() if chunk_ids]
    ordered: list[str] = []
    seen: set[str] = set()
    while queues:
        remaining: list[deque[str]] = []
        for queue in queues:
            while queue and queue[0] in seen:
                queue.popleft()
            if queue:
                chunk_id = queue.popleft()
                seen.add(chunk_id)
                ordered.append(chunk_id)
            if queue:
                remaining.append(queue)
        queues = remaining

    for chunk_id in evidence:
        if chunk_id not in seen:
            seen.add(chunk_id)
            ordered.append(chunk_id)

    retained: list[str] = []
    used_chars = 320
    for chunk_id in ordered:
        result = evidence.get(chunk_id)
        if result is None:
            continue
        chunk = result.chunk
        estimated_chars = (
            len(chunk.text)
            + len(chunk.source_path)
            + sum(map(len, chunk.heading_path))
            + 220
        )
        if used_chars + estimated_chars <= max_context_chars:
            retained.append(chunk_id)
            used_chars += estimated_chars
    return tuple(retained)


def execute_retrieval_plan(
    plan: SkillRetrievalPlan,
    *,
    state: RetrievalExecutionState,
    search: SearchExecutor,
    read_section: SectionExecutor,
    max_tool_calls: int,
    max_followup_searches: int,
    max_context_chars: int,
) -> SkillRetrievalOutcome:
    """Run plan steps sequentially against the owning turn's shared budgets."""

    if max_tool_calls < 1 or max_context_chars < 1 or max_followup_searches < 0:
        raise ValueError("retrieval budgets are invalid")

    step_results: list[RetrievalStepResult] = []
    target_hits: dict[str, list[str]] = {
        target: [] for target in plan.coverage_targets
    }
    expected_version: str | None = None
    search_count = 0
    attempted_targets: list[str] = []
    budget_limited = "budget_limited" in plan.reason_code

    for step in plan.steps:
        if state.tool_calls >= max_tool_calls:
            budget_limited = True
            step_results.append(
                RetrievalStepResult(
                    "skipped",
                    stop_reason="tool_budget_exhausted",
                    step_id=step.step_id,
                )
            )
            continue
        if (
            step.operation == "search"
            and search_count > 0
            and state.followup_searches >= max_followup_searches
        ):
            budget_limited = True
            step_results.append(
                RetrievalStepResult(
                    "skipped",
                    stop_reason="followup_search_budget_exhausted",
                    step_id=step.step_id,
                )
            )
            continue

        previous_evidence = dict(state.evidence)
        state.tool_calls += 1
        if step.operation == "search":
            if search_count > 0:
                state.followup_searches += 1
            search_count += 1
            attempted_targets.extend(target for target in step.targets if target not in attempted_targets)
            output = search(step.query or "")
            returned = _chunk_ids(output, "results")
        else:
            output = read_section(step.anchor_id or "")
            returned = _chunk_ids(output, "chunks")

        raw_version = output.get("index_version")
        if isinstance(raw_version, str):
            if expected_version is None:
                expected_version = raw_version
            elif raw_version != expected_version:
                state.evidence.clear()
                state.evidence.update(previous_evidence)
                step_results.append(
                    RetrievalStepResult(
                        "error",
                        stop_reason="index_version_changed",
                        step_id=step.step_id,
                    )
                )
                continue

        status = output.get("status")
        if status == "error":
            step_results.append(
                RetrievalStepResult(
                    "error",
                    returned,
                    stop_reason=_error_code(output),
                    step_id=step.step_id,
                )
            )
            continue

        stop_reason = (
            output.get("reason_code", "retrieval_empty")
            if status == "empty"
            else "section_limited"
            if status == "limited"
            else None
        )
        step_results.append(
            RetrievalStepResult(
                "executed",
                returned,
                stop_reason=stop_reason,
                step_id=step.step_id,
            )
        )
        for target in step.targets:
            hits = target_hits.setdefault(target, [])
            for chunk_id in returned:
                result = state.evidence.get(chunk_id)
                if chunk_id in hits or result is None:
                    continue
                if target.startswith("object:"):
                    subject = target.removeprefix("object:")
                    visible_text = "\n".join((*result.chunk.heading_path, result.chunk.text))
                    # Running a query for A/B is not evidence for both objects.
                    # Conservative lexical coverage, never a semantic guarantee.
                    pattern = re.escape(subject)
                    if subject.isascii():
                        pattern = r"(?<![A-Za-z0-9_])" + pattern + r"(?![A-Za-z0-9_])"
                    if re.search(pattern, visible_text, re.I) is None:
                        continue
                hits.append(chunk_id)

    retained = _retained_ids(
        target_hits,
        state.evidence,
        max_context_chars=max_context_chars,
    )
    retained_set = set(retained)
    finalized = tuple(
        RetrievalStepResult(
            result.status,
            result.returned_chunk_ids,
            tuple(
                chunk_id
                for chunk_id in result.returned_chunk_ids
                if chunk_id in retained_set
            ),
            result.stop_reason,
            result.step_id,
        )
        for result in step_results
    )
    missing_targets = tuple(
        target for target in plan.coverage_targets if not retained_set.intersection(target_hits.get(target, ()))
    )
    return SkillRetrievalOutcome(
        executed=sum(result.status == "executed" for result in finalized),
        skipped=sum(result.status == "skipped" for result in finalized),
        errors=sum(result.status == "error" for result in finalized),
        missing_targets=missing_targets,
        budget_limited=budget_limited or len(retained) < len(state.evidence),
        strategy_id=plan.strategy_id,
        step_results=finalized,
        index_version=expected_version,
        retained_chunk_ids=retained,
        target_chunk_ids=tuple((target, tuple(ids)) for target, ids in target_hits.items()),
        query_attempted_targets=tuple(attempted_targets),
    )


def reconcile_retrieval_outcome(
    outcome: SkillRetrievalOutcome, *, available_chunk_ids: Sequence[str],
) -> SkillRetrievalOutcome:
    """Use the exact final source map after serialized-context trimming.

    A successful search and a retained candidate do not imply that the answering
    model received that candidate. Keep this distinction for every target.
    """
    available = set(available_chunk_ids)
    retained = tuple(chunk_id for chunk_id in outcome.retained_chunk_ids if chunk_id in available)
    retained_set = set(retained)
    return replace(
        outcome,
        retained_chunk_ids=retained,
        missing_targets=tuple(target for target, ids in outcome.target_chunk_ids if not retained_set.intersection(ids)),
        budget_limited=outcome.budget_limited or len(retained) < len(outcome.retained_chunk_ids),
        step_results=tuple(replace(item, retained_chunk_ids=tuple(
            chunk_id for chunk_id in item.retained_chunk_ids if chunk_id in retained_set
        )) for item in outcome.step_results),
    )


def section_followup_step(
    outcome: SkillRetrievalOutcome,
    evidence: Mapping[str, RetrievalResult],
) -> RetrievalStep | None:
    """Propose one registered-anchor read only for a visibly incomplete chunk."""

    for chunk_id in outcome.retained_chunk_ids:
        result = evidence.get(chunk_id)
        if result is None:
            continue
        text = result.chunk.text.rstrip()
        incomplete_code = text.count("```") % 2 == 1
        incomplete_sentence = bool(text) and text[-1] in {
            ":",
            "：",
            "\\",
            ",",
            "，",
        }
        if incomplete_code or incomplete_sentence:
            return RetrievalStep(
                "read-section-1",
                "read_section",
                "adjacent_context",
                False,
                anchor_id=chunk_id,
            )
    return None


def retrieval_summary(
    plan: SkillRetrievalPlan,
    outcome: SkillRetrievalOutcome,
) -> dict[str, Any]:
    """Return a persistence-safe summary without queries or evidence text."""

    pairs = tuple(zip(plan.steps, outcome.step_results))
    skipped_reason = next(
        (
            result.stop_reason
            for result in outcome.step_results
            if result.status == "skipped"
        ),
        None,
    )
    return {
        "strategy_id": plan.strategy_id,
        "coverage_targets": list(plan.coverage_targets),
        "planned_searches": sum(
            step.operation == "search" for step in plan.steps
        ),
        "executed_searches": sum(
            result.status == "executed" and step.operation == "search"
            for step, result in pairs
        ),
        "section_reads": sum(
            result.status == "executed" and step.operation == "read_section"
            for step, result in pairs
        ),
        "retained_chunks": len(outcome.retained_chunk_ids),
        "skipped_reason": skipped_reason,
        "coverage_missing": list(outcome.missing_targets),
        "coverage_is_retrieval_only": True,
        "query_attempted_targets": list(outcome.query_attempted_targets),
        "coverage_basis": "provided_evidence_lexical" if plan.strategy_id == "comparison" else "provided_evidence_from_query",
        "budget_limited": outcome.budget_limited,
    }


# Validation is intentionally deferred; this file contains implementation only.
# TODO 15: validate query order, counters, empty results, and missing visible code.
# TODO 16: validate provenance, version changes, deduplication, and shared budgets.
# TODO 17: compare off/manual/auto behavior using the evaluation layer.
