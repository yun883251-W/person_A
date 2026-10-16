"""Agent 会话记录、单轮状态和阶段间传递的数据类型。

不负责调用模型、执行工具或管理应用生命周期，供各执行阶段共同使用。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..knowledge.skill_retrieval import SkillRetrievalOutcome, SkillRetrievalPlan
from ..schemas import AnswerResult, AnswerSource, AnswerStatus, RetrievalResult
from ..session_store import SessionStoreError
from ..tools.registry import InMemoryToolRegistry
from .citations import historical_answer_text
from .router import RouteKind
from .skill_selector import SelectedSkillSnapshot
from .trace import TraceRecorder


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One committed, complete conversation turn."""

    question: str
    retrieval_question: str
    messages: tuple[BaseMessage, ...]
    result: AnswerResult

    @property
    def route(self) -> str:
        return self.result.routing.route if self.result.routing is not None else "knowledge"

    @property
    def intent_kind(self) -> str:
        return self.result.routing.intent if self.result.routing is not None else "knowledge"


class TurnPersistenceError(SessionStoreError):
    """Report that a completed answer could not be saved."""

    def __init__(self, result: AnswerResult) -> None:
        super().__init__("The answer was generated, but the conversation turn was not saved")
        self.result = result


class ConversationHistory:
    """Keep complete turns isolated by an explicit session identifier."""

    def __init__(self, max_turns: int) -> None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        self._max_turns = max_turns
        self._sessions: dict[str, list[ConversationTurn]] = {}

    @staticmethod
    def _session_key(session_id: str) -> str:
        if not isinstance(session_id, str):
            raise TypeError("session_id must be a string")
        key = session_id.strip()
        if not key:
            raise ValueError("session_id must not be blank")
        return key

    def turns(self, session_id: str) -> tuple[ConversationTurn, ...]:
        return tuple(self._sessions.get(self._session_key(session_id), ()))

    def messages(
        self, session_id: str, *, route: RouteKind | None = None
    ) -> tuple[BaseMessage, ...]:
        """Project committed turns into model context without old knowledge IDs.

        Keep raw messages and rendered results on each turn for persistence and
        display. Replaying raw knowledge exchanges would also replay retrieved
        chunks, tool payloads and rejected model answers as if they were current.
        This read-time projection covers restored sessions without a migration.
        """
        messages: list[BaseMessage] = []
        for turn in self.turns(session_id):
            if route is not None and turn.route != route.value:
                continue
            if turn.route == RouteKind.KNOWLEDGE.value:
                answer = historical_answer_text(turn.result.answer, turn.result.sources)
                messages.extend((
                    HumanMessage(content=turn.question),
                    AIMessage(content=answer or "该轮回答移除历史引用后没有可供回顾的正文。"),
                ))
            else:
                messages.extend(turn.messages)
        return tuple(messages)

    def append(self, session_id: str, turn: ConversationTurn) -> None:
        key = self._session_key(session_id)
        turns = self._sessions.setdefault(key, [])
        turns.append(turn)
        if len(turns) > self._max_turns:
            del turns[: len(turns) - self._max_turns]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(self._session_key(session_id), None)


def _complete_history_messages(messages: Sequence[object]) -> tuple[BaseMessage, ...] | None:
    """Return a system-free history only when every tool exchange is complete."""
    history = tuple(
        message
        for message in messages
        if isinstance(message, BaseMessage) and not isinstance(message, SystemMessage)
    )
    requested: set[str] = set()
    pending: set[str] = set()
    for message in history:
        if pending and not isinstance(message, ToolMessage):
            return None
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                call_id = str(call.get("id", "")).strip()
                if not call_id or call_id in requested:
                    return None
                requested.add(call_id)
                pending.add(call_id)
        elif isinstance(message, ToolMessage):
            call_id = str(message.tool_call_id).strip()
            if call_id not in pending:
                return None
            pending.remove(call_id)
    if pending:
        return None
    return history


RetrievalStatus = Literal["not_started", "empty", "results_found", "failed"]


@dataclass(slots=True)
class TurnState:
    """Mutable execution state owned by exactly one question-answer turn."""

    original_question: str
    retrieval_question: str | None = None
    messages: list[object] = field(default_factory=list)
    evidence: dict[str, RetrievalResult] = field(default_factory=dict)
    source_map: dict[str, AnswerSource] = field(default_factory=dict)
    # Prior-step snapshots are citation material only, not current tool read
    # permissions. Keep them out of the retriever's chunk-ID namespace.
    dependency_evidence: tuple[RetrievalResult, ...] = ()
    dependency_transform: bool = False
    # Exactly the full source text sent under each currently registered ID.
    supplied_evidence: dict[str, RetrievalResult] = field(default_factory=dict)
    retrieval_status: RetrievalStatus = "not_started"
    model_calls: int = 0
    tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    last_tool_error_code: str | None = None
    followup_searches: int = 0
    context_chars: int = 0
    completed_tool_call_ids: set[str] = field(default_factory=set)
    status: AnswerStatus | None = None
    stop_reason: str | None = None
    retrieval_error_code: str | None = None
    skill_retrieval_plan: SkillRetrievalPlan | None = None
    skill_retrieval_outcome: SkillRetrievalOutcome | None = None
    skill_retrieval_summary: dict[str, Any] | None = None
    selected_skill: SelectedSkillSnapshot | None = None
    step_id: str | None = None
    trace_recorder: TraceRecorder | None = None
    # Browser actions have a program-authored receipt, not an LLM page summary.
    browser_action_result: AnswerResult | None = None
    # Host-bound topic from this request's clauses; never inferred from history
    # or another model answer. Browser dispatch must match it exactly.
    browser_search_topic: str | None = None

    def __post_init__(self) -> None:
        if self.retrieval_question is None:
            self.retrieval_question = self.original_question


ToolExecutor = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class KnowledgeTools:
    """Model-visible read-only tools and their closed execution registry."""

    definitions: tuple[BaseTool, ...]
    executors: Mapping[str, ToolExecutor]
    registry: InMemoryToolRegistry[TurnState] | None = None


QuestionKind = Literal[
    "knowledge",
    "greeting",
    "scope",
    "needs_clarification",
    "out_of_scope",
]


@dataclass(frozen=True, slots=True)
class PreparedQuestion:
    """Result of validating, classifying, and (when needed) retrieving a query."""

    state: TurnState
    kind: QuestionKind
    evidence_context: str = ""
    terminal_result: AnswerResult | None = None


@dataclass(frozen=True, slots=True)
class PreparedModelTurn:
    """Messages, tools, and retry policy needed for one model invocation."""

    model: Any
    messages: tuple[BaseMessage, ...]
    tools: KnowledgeTools
    max_retries: int
    skill_id: str | None = None
    route: RouteKind = RouteKind.KNOWLEDGE
    # Planned knowledge leaves can reserve their final invocation for an answer
    # without exposing tools that would require another model round afterwards.
    answer_model: Any | None = None


ModelTurnKind = Literal["tool_requests", "final_answer", "empty_response"]


@dataclass(frozen=True, slots=True)
class ModelTurnResult:
    """A complete model message classified for the agent's next step."""

    kind: ModelTurnKind
    message: AIMessage
    answer: str | None = None


ToolCallStatus = Literal["success", "error"]


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """The outcome of one model-requested tool call, in request order."""

    tool_call_id: str | None
    name: str | None
    status: ToolCallStatus
    output: Any
