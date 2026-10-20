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
from knowledge_agent.session.store import SessionStoreError
from ..tools.registry import InMemoryToolRegistry
from knowledge_agent.agent.answering.citations import historical_answer_text
from .router import RouteKind
from knowledge_agent.skills.selector import SelectedSkillSnapshot
from knowledge_agent.observability.trace import TraceRecorder


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """一个已提交、完整的对话轮次。"""

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
    """报告已完成的回答无法保存。"""

    def __init__(self, result: AnswerResult) -> None:
        super().__init__("The answer was generated, but the conversation turn was not saved")
        self.result = result


class ConversationHistory:
    """通过显式会话标识保持完整轮次相互隔离。"""

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
        """把已提交轮次投影为模型上下文，不含旧知识 ID。

        在每个轮次上保留原始消息和渲染结果，用于持久化和展示。
        重放原始知识交换也会重放检索片段、工具载荷和被拒绝的模型回答，
        仿佛它们仍属于当前轮次。
        这个读取时投影覆盖已恢复的会话，无需迁移。
        """
        messages: list[BaseMessage] = []
        recent_chat = {id(turn) for turn in self.turns(session_id)[-2:] if turn.route == "chat"}
        for turn in self.turns(session_id):
            if route is not None and turn.route != route.value:
                if not (route is RouteKind.KNOWLEDGE and id(turn) in recent_chat):
                    continue
            if turn.route in {"knowledge", "chat"}:
                if turn.result.assessment_state is not None or turn.result.status in {"service_error", "limit_reached"}:
                    continue
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
    """仅当所有工具交换都完整时，返回不含系统消息的历史。"""
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
    """由恰好一个问答轮次持有的可变执行状态。"""

    original_question: str
    retrieval_question: str | None = None
    messages: list[object] = field(default_factory=list)
    evidence: dict[str, RetrievalResult] = field(default_factory=dict)
    source_map: dict[str, AnswerSource] = field(default_factory=dict)
    # 先前步骤的快照只是引用材料，不是当前工具读取
    # 权限。将它们排除在检索器的 chunk-ID 命名空间之外。
    dependency_evidence: tuple[RetrievalResult, ...] = ()
    dependency_transform: bool = False
    # 恰好是当前每个已登记 ID 所对应的完整源文本。
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
    # 浏览器操作有程序生成的回执，而不是 LLM 页面摘要。
    browser_action_result: AnswerResult | None = None
    # 来自本请求从句、由宿主绑定的主题；绝不从历史
    # 或另一个模型回答推断。浏览器派发必须精确匹配它。
    browser_search_topic: str | None = None
    browser_search_limit: int | None = None

    def __post_init__(self) -> None:
        if self.retrieval_question is None:
            self.retrieval_question = self.original_question


ToolExecutor = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class KnowledgeTools:
    """模型可见的只读工具及其封闭的执行注册表。"""

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
    """查询经过校验、分类以及（必要时）检索后的结果。"""

    state: TurnState
    kind: QuestionKind
    evidence_context: str = ""
    terminal_result: AnswerResult | None = None


@dataclass(frozen=True, slots=True)
class PreparedModelTurn:
    """一次模型调用所需的消息、工具和重试策略。"""

    model: Any
    messages: tuple[BaseMessage, ...]
    tools: KnowledgeTools
    max_retries: int
    skill_id: str | None = None
    route: RouteKind = RouteKind.KNOWLEDGE
    # 计划中的知识叶子可以把最后一次调用留给回答，
    # 而不暴露那些之后还需要再一轮模型的工具。
    answer_model: Any | None = None


ModelTurnKind = Literal["tool_requests", "final_answer", "empty_response"]


@dataclass(frozen=True, slots=True)
class ModelTurnResult:
    """一条已分类、供智能体判断下一步的完整模型消息。"""

    kind: ModelTurnKind
    message: AIMessage
    answer: str | None = None


ToolCallStatus = Literal["success", "error"]


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """一次模型请求的工具调用结果，按请求顺序排列。"""

    tool_call_id: str | None
    name: str | None
    status: ToolCallStatus
    output: Any
