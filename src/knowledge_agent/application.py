"""CLI 与 HTTP 共用的应用服务：会话生命周期、互斥访问和资源释放。

Agent 的问答执行位于 agent.runner；本模块管理应用级服务与持久化边界。
"""

from __future__ import annotations

import uuid
from threading import Lock
from typing import Any

from .agent import assessment
from .agent.runner import KnowledgeAgent
from .agent.state import ConversationHistory, ConversationTurn, TurnPersistenceError, TurnState
from .config import AgentConfig
from .knowledge.retrieval import KnowledgeRetriever
from .schemas import AnswerResult
from .session_store import SessionNotFoundError, SessionStore
from .tools.registry import InMemoryToolRegistry


class AgentApplicationError(RuntimeError):
    """Base error intended for presentation by an application adapter or CLI."""


class SessionNotStartedError(AgentApplicationError):
    """Raised when a session operation is requested before starting a session."""


class ApplicationBusyError(AgentApplicationError):
    """Raised when overlapping synchronous application operations are attempted."""


class KnowledgeAgentApplication:
    """Own the reusable services and one synchronous conversation lifecycle.

    A CLI only needs to call :meth:`start_session`, :meth:`ask`,
    :meth:`clear_session`, and :meth:`refresh_index`.  The operation lock keeps
    an index refresh from overlapping a model/tool loop, so each answer observes
    one index snapshot.  Conversation state remains owned by this instance's
    ``KnowledgeAgent`` rather than by a shared model client or retriever.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        model: object | None = None,
        retriever: KnowledgeRetriever | None = None,
        session_store: SessionStore | None = None,
        mcp_registry: InMemoryToolRegistry[TurnState] | None = None,
        owned_mcp_runtime: Any | None = None,
    ) -> None:
        self.agent = KnowledgeAgent(
            config,
            model=model,
            retriever=retriever,
            session_store=session_store,
            mcp_registry=mcp_registry,
        )
        self._session_id: str | None = None
        self._owned_mcp_runtime = owned_mcp_runtime
        self._operation: str | None = None
        self._operation_lock = Lock()
        self._closed = False

    @property
    def config(self) -> AgentConfig:
        """Expose the effective limits without duplicating configuration in a CLI."""
        return self.agent.config

    @property
    def session_id(self) -> str | None:
        """Return the active session identifier, if a session has been started."""
        return self._session_id

    def _begin_operation(self, operation: str) -> None:
        if not self._operation_lock.acquire(blocking=False):
            active = self._operation or "another operation"
            raise ApplicationBusyError(
                f"cannot start {operation} while {active} is in progress"
            )
        self._operation = operation
        if self._closed:
            self._finish_operation()
            raise AgentApplicationError("the application is closed")

    def _finish_operation(self) -> None:
        self._operation = None
        self._operation_lock.release()

    def _require_session(self) -> str:
        if self._session_id is None:
            raise SessionNotStartedError("start_session() must be called first")
        return self._session_id

    def start_session(self, session_id: str | None = None) -> str:
        """Create and activate a new, empty session."""
        self._begin_operation("start_session")
        try:
            if self._session_id is not None:
                raise AgentApplicationError("a session is already active")
            store = self.agent.session_store
            if store is not None:
                store.initialize()
                if session_id is None:
                    key = store.create_session().session_id
                else:
                    key = store.create_session(
                        ConversationHistory._session_key(session_id)
                    ).session_id
            else:
                key = (
                    ConversationHistory._session_key(session_id)
                    if session_id is not None
                    else uuid.uuid4().hex
                )
            self.agent.clear_session(key)
            self._session_id = key
            return key
        finally:
            self._finish_operation()

    def resume_session(self, session_id: str) -> str:
        """Restore the newest complete turns of an existing session."""
        self._begin_operation("resume_session")
        try:
            if self._session_id is not None:
                raise AgentApplicationError("a session is already active")
            key = ConversationHistory._session_key(session_id)
            store = self.agent.session_store
            if store is None:
                raise AgentApplicationError("session persistence is not configured")
            store.initialize()
            if store.get_session(key) is None:
                raise SessionNotFoundError("Session does not exist")
            stored_turns = store.load_context_turns(
                key, max_history_turns=self.config.max_history_turns
            )
            # 答题状态独立于模型历史窗口；中途问了多轮知识问题后仍可继续。
            loader = getattr(store, "load_latest_assessment", None)
            snapshot = loader(key) if callable(loader) else next(
                (
                    turn.result.assessment_state for turn in reversed(stored_turns)
                    if turn.result.assessment_state is not None
                ),
                None,
            )
            restored_assessment = (
                assessment.validate_snapshot(snapshot) if snapshot is not None else None
            )
            self.agent.clear_session(key)
            for turn in stored_turns:
                self.agent.history.append(
                    key,
                    ConversationTurn(
                        question=turn.question,
                        retrieval_question=turn.retrieval_question,
                        messages=turn.messages,
                        result=turn.result,
                    ),
                )
            if restored_assessment is not None:
                self.agent.assessment_states[key] = restored_assessment
            self._session_id = key
            return key
        finally:
            self._finish_operation()

    def ask(
        self, question: str, *, skill_id: str | None = None, skill_mode: str | None = None,
    ) -> AnswerResult:
        """Run one complete turn in the active session."""
        self._begin_operation("ask")
        try:
            if self._owned_mcp_runtime is not None:
                # 每轮只更新能力快照，不重复启动进程；断连工具不会继续被宣称可用。
                if self._owned_mcp_runtime.list_tools():
                    from .mcp.adapter import create_mcp_tool_registry

                    self.agent.mcp_registry = create_mcp_tool_registry(self._owned_mcp_runtime)
                else:
                    self.agent.mcp_registry = InMemoryToolRegistry()
            return self.agent.ask(
                question,
                session_id=self._require_session(),
                skill_id=skill_id,
                skill_mode=skill_mode,
            )
        finally:
            self._finish_operation()

    def clear_session(self) -> None:
        """Delete persisted and retained turns while keeping the session active."""
        self._begin_operation("clear_session")
        try:
            session_id = self._require_session()
            store = self.agent.session_store
            if store is not None:
                store.initialize()
                if not store.clear_session(session_id):
                    raise SessionNotFoundError("Session does not exist")
            self.agent.clear_session(session_id)
        finally:
            self._finish_operation()

    def end_session(self) -> None:
        """Release in-memory state while retaining all committed records."""
        self._begin_operation("end_session")
        try:
            session_id = self._require_session()
            self.agent.clear_session(session_id)
            self._session_id = None
        finally:
            self._finish_operation()

    def delete_session(self, session_id: str) -> bool:
        """Delete a persisted session and release it if it is currently active."""
        self._begin_operation("delete_session")
        try:
            key = ConversationHistory._session_key(session_id)
            store = self.agent.session_store
            if store is None:
                raise AgentApplicationError("session persistence is not configured")
            store.initialize()
            deleted = store.delete_session(key)
            if deleted and self._session_id == key:
                self.agent.clear_session(key)
                self._session_id = None
            return deleted
        finally:
            self._finish_operation()

    def close(self) -> None:
        """Release the active session, storage, and explicitly owned MCP runtime."""
        if self._closed:
            return
        self._begin_operation("close")
        try:
            try:
                if self._session_id is not None:
                    self.agent.clear_session(self._session_id)
                    self._session_id = None
            finally:
                try:
                    if self.agent.session_store is not None:
                        self.agent.session_store.close()
                finally:
                    try:
                        if self._owned_mcp_runtime is not None:
                            self._owned_mcp_runtime.close()
                    finally:
                        self._closed = True
        finally:
            self._finish_operation()

    def __enter__(self) -> KnowledgeAgentApplication:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def get_session_history(self) -> tuple[ConversationTurn, ...]:
        """Return an immutable snapshot of the active session history."""
        self._begin_operation("get_session_history")
        try:
            return self.agent.get_session_history(self._require_session())
        finally:
            self._finish_operation()

    def refresh_index(self) -> Any:
        """Refresh the shared retriever strictly between complete answer turns."""
        self._begin_operation("refresh_index")
        try:
            return self.agent.retriever.refresh()
        finally:
            self._finish_operation()


def create_agent_application(
    config: AgentConfig | None = None,
    *,
    model: object | None = None,
    retriever: KnowledgeRetriever | None = None,
    session_store: SessionStore | None = None,
    mcp_registry: InMemoryToolRegistry[TurnState] | None = None,
    owned_mcp_runtime: Any | None = None,
) -> KnowledgeAgentApplication:
    """Create the shared CLI/API boundary with an already prepared tool registry.

    Injected registries remain caller-owned. Bootstrap can transfer its runtime
    explicitly so the application refreshes snapshots and closes it on shutdown.
    """
    return KnowledgeAgentApplication(
        config,
        model=model,
        retriever=retriever,
        session_store=session_store,
        mcp_registry=mcp_registry,
        owned_mcp_runtime=owned_mcp_runtime,
    )
