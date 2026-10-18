"""Pass verified answers and immutable source snapshots between plan steps.

Previous prose is background, never a substitute for source text. Each receiving
step assigns new local citation IDs to the complete snapshots it actually sends
to its model. No tools, tool receipts, or old citation IDs cross this boundary.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from knowledge_agent.schemas import AnswerResult, AnswerSource, RetrievalResult
from knowledge_agent.agent.answering.citations import historical_answer_text


@dataclass(frozen=True, slots=True)
class DependencyStep:
    step_id: str
    goal: str
    answer: str
    evidence: tuple[RetrievalResult, ...] = ()


@dataclass(frozen=True, slots=True)
class DependencyContext:
    steps: tuple[DependencyStep, ...] = ()

    @property
    def evidence(self) -> tuple[RetrievalResult, ...]:
        return unique_evidence(item for step in self.steps for item in step.evidence)


def source_identity(source: AnswerSource) -> tuple[object, ...]:
    """A citation's complete snapshot identity, excluding its ranking score."""
    return (
        source.index_version, source.chunk_id, source.content_hash,
        source.source_path, source.heading_path, source.start_line, source.end_line,
    )


def evidence_identity(result: RetrievalResult) -> tuple[object, ...]:
    # Include the actual bytes as well as the declared hash. A malformed or
    # stale snapshot must not alias different text under a shared chunk ID.
    digest = hashlib.sha256(result.chunk.text.encode("utf-8", errors="surrogatepass")).hexdigest()
    return (*source_identity(AnswerSource.from_retrieval_result(result)), digest)


def unique_evidence(results: Iterable[RetrievalResult]) -> tuple[RetrievalResult, ...]:
    """Deduplicate full snapshots; discard conflicting records within a version."""
    retained: dict[tuple[object, ...], RetrievalResult] = {}
    snapshot_ids: dict[tuple[str, str], tuple[object, ...]] = {}
    conflicts: set[tuple[str, str]] = set()
    for result in results:
        key = evidence_identity(result)
        snapshot_id = (result.index_version, result.chunk_id)
        previous = snapshot_ids.setdefault(snapshot_id, key)
        if previous != key:
            conflicts.add(snapshot_id)
        retained.setdefault(key, result)
    return tuple(
        result for result in retained.values()
        if (result.index_version, result.chunk_id) not in conflicts
    )


def cited_evidence(
    result: AnswerResult, supplied: Mapping[str, RetrievalResult],
) -> tuple[RetrievalResult, ...]:
    """Export only source snapshots actually supplied and accepted as citations."""
    if result.status != "success" or (result.routing is not None and result.routing.route != "knowledge"):
        return ()
    cited = {source_identity(source) for source in result.sources}
    return unique_evidence(
        item for item in supplied.values()
        if source_identity(AnswerSource.from_retrieval_result(item)) in cited
    )


def dependency_step(
    step_id: str, goal: str, result: AnswerResult,
    evidence: tuple[RetrievalResult, ...],
) -> DependencyStep:
    """Create the read-only handoff after the executor checks step success."""
    if result.status != "success" or (result.routing is not None and result.routing.route != "knowledge"):
        raise ValueError("Only completed knowledge answers may supply dependency context")
    snapshots = cited_evidence(result, {str(index): item for index, item in enumerate(evidence)})
    return DependencyStep(
        step_id=step_id, goal=goal,
        answer=historical_answer_text(result.answer, result.sources), evidence=snapshots,
    )


_TRANSFORM = re.compile(
    r"总结|归纳|概括|整理|改写|重写|压缩|精简|翻译|提炼|"
    r"(?:列出|给出|制定|生成|写出|做成|转换为|转成).{0,24}"
    r"(?:要点|学习步骤|学习计划|学习路线|学习清单|大纲|表格|摘要)|"
    r"\b(?:summarize|summarise|rewrite|rephrase|translate|outline)\b|"
    r"\b(?:create|give|make|list)\b.{0,30}\b(?:study plan|learning steps|outline|summary)\b",
    re.IGNORECASE,
)
_NEW_FACTS = re.compile(
    r"搜索|检索|查找|查询|补充|扩展|举例|示例|最新|额外|其他资料|更多资料|"
    r"解释|介绍|讲解|说明|分析|证明|推导|工作原理|底层原理|如何实现|编写代码|实现代码|对比|比较|"
    r"\b(?:search|research|look up|additional|new facts|examples?|compare|implement|explain|introduce|analyze|analyse|prove)\b",
    re.IGNORECASE,
)


def is_dependency_transform(goal: str, has_dependencies: bool = True) -> bool:
    """Conservatively identify reshaping of explicitly referenced prior answers.

    This rule only decides whether a new search is mandatory. It never grants
    an operation, changes a dependency, or exempts the answer from citations.
    Requests for new facts retain the ordinary retrieval path.
    """
    return bool(has_dependencies and _TRANSFORM.search(goal) and not _NEW_FACTS.search(goal))
