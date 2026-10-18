"""CLI 与 HTTP 共用的应用服务：会话生命周期、互斥访问和资源释放。

Agent 的问答执行位于 agent.runner；本模块管理应用级服务与持久化边界。
"""

from __future__ import annotations

import uuid
from threading import Lock
from typing import Any

from .agent import assessment
from knowledge_agent.session.memory import MemorySettings, MemoryUpdate, safe_summary_text, update_session_summary
from .agent.runner import KnowledgeAgent
from .agent.state import ConversationHistory, ConversationTurn, TurnPersistenceError, TurnState
from .config import AgentConfig, get_memory_settings
from .knowledge.retrieval import KnowledgeRetriever
from knowledge_agent.models.provider import invoke_chat_model
from .schemas import AnswerResult
from knowledge_agent.session.store import SessionNotFoundError, SessionStore
from .tools.registry import InMemoryToolRegistry


class AgentApplicationError(RuntimeError):
    """应用适配器或 CLI 用于展示的错误基类。"""


class SessionNotStartedError(AgentApplicationError):
    """尚未开始会话就请求会话操作时抛出。"""


class ApplicationBusyError(AgentApplicationError):
    """尝试重叠执行同步应用操作时抛出。"""


class KnowledgeAgentApplication:
    """持有可复用服务，以及一条同步对话生命周期。

    CLI 只需调用 :meth:`start_session`、:meth:`ask`、:meth:`clear_session`
    和 :meth:`refresh_index`。操作锁确保索引刷新不会与模型/工具循环重叠，
    因此每个回答都基于同一份索引快照。对话状态由本实例的
    ``KnowledgeAgent`` 持有，而不是由共享的模型客户端或检索器持有。
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
        memory_settings: MemorySettings | None = None,
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
        self.memory_settings = memory_settings if memory_settings is not None else get_memory_settings()
        # 摘要工作是辅助性的：它绝不增加已完成回答的 model_calls/tool_calls，
        # 也不会重写已成功提交的结果。
        self.last_memory_update = MemoryUpdate("not_run")

    @property
    def config(self) -> AgentConfig:
        """直接暴露生效的上限，避免在 CLI 中重复配置。"""
        return self.agent.config

    @property
    def session_id(self) -> str | None:
        """返回当前活动的会话标识符；尚未开始会话时返回 ``None``。"""
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

    def _release_session_memory(self, session_id: str) -> None:
        self.agent.session_summaries.pop(session_id, None)
        self.last_memory_update = MemoryUpdate("not_run")

    def _restore_session_memory(self, session_id: str) -> None:
        """可选的记忆功能失败不得妨碍恢复已保存的回答。"""
        self._release_session_memory(session_id)
        if not self.memory_settings.enabled:
            self.last_memory_update = MemoryUpdate("disabled")
            return
        loader = getattr(self.agent.session_store, "load_session_summary", None)
        if not callable(loader):
            self.last_memory_update = MemoryUpdate("unavailable", reason_code="memory_store_unavailable")
            return
        try:
            record = loader(session_id)
            if record is not None:
                summary = safe_summary_text(record.summary, max_chars=self.memory_settings.max_chars)
                if summary:
                    self.agent.session_summaries[session_id] = summary
            self.last_memory_update = MemoryUpdate("unchanged", record=record)
        except Exception:
            self.last_memory_update = MemoryUpdate("failed", reason_code="memory_restore_failed")

    def _maintain_session_memory(self, session_id: str) -> None:
        update = MemoryUpdate("not_run")
        try:
            update = update_session_summary(
                self.agent.session_store, session_id, settings=self.memory_settings,
                max_history_turns=self.config.max_history_turns,
                # 每批只尝试一次；提供方自身的重试不得放大该次数。
                model_invoker=lambda messages: invoke_chat_model(self.agent.model, messages, max_retries=0),
            )
            self.last_memory_update = update
            if update.status in {"updated", "unchanged", "conflict"}:
                summary = (
                    safe_summary_text(update.record.summary, max_chars=self.memory_settings.max_chars)
                    if update.record is not None else ""
                )
                if summary:
                    self.agent.session_summaries[session_id] = summary
                else:
                    self.agent.session_summaries.pop(session_id, None)
        except Exception:
            # 即使是自定义存储出现意外的记忆故障，也不能把已持久化的
            # 回答变成应用故障。中断信号仍会向上传播。
            self.last_memory_update = MemoryUpdate(
                "failed", update.model_calls, "memory_maintenance_failed",
            )

    def start_session(self, session_id: str | None = None) -> str:
        """创建并激活一个新的空会话。"""
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
            self._restore_session_memory(key)
            self._session_id = key
            return key
        finally:
            self._finish_operation()

    def resume_session(self, session_id: str) -> str:
        """恢复现有会话中最近的完整轮次。"""
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
            business_loader = getattr(store, "load_business_state", None)
            restored_business = business_loader(key) if callable(business_loader) else None
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
            if restored_business is not None:
                self.agent.restore_business_state(key, restored_business)
            self._restore_session_memory(key)
            self._session_id = key
            return key
        finally:
            self._finish_operation()

    def ask(
        self, question: str, *, skill_id: str | None = None, skill_mode: str | None = None,
    ) -> AnswerResult:
        """在活动会话中执行一个完整轮次。"""
        self._begin_operation("ask")
        try:
            if self._owned_mcp_runtime is not None:
                # 每轮只更新能力快照，不重复启动进程；断连工具不会继续被宣称可用。
                if self._owned_mcp_runtime.list_tools():
                    from .mcp.adapter import create_mcp_tool_registry

                    self.agent.mcp_registry = create_mcp_tool_registry(self._owned_mcp_runtime)
                else:
                    self.agent.mcp_registry = InMemoryToolRegistry()
            session_id = self._require_session()
            result = self.agent.ask(
                question,
                session_id=session_id,
                skill_id=skill_id,
                skill_mode=skill_mode,
            )
            if result.status not in {"service_error", "limit_reached"}:
                self._maintain_session_memory(session_id)
            else:
                self.last_memory_update = MemoryUpdate("not_run")
            return result
        finally:
            self._finish_operation()

    def clear_session(self) -> None:
        """删除已持久化和保留的轮次，同时保持会话处于活动状态。"""
        self._begin_operation("clear_session")
        try:
            session_id = self._require_session()
            store = self.agent.session_store
            if store is not None:
                store.initialize()
                if not store.clear_session(session_id):
                    raise SessionNotFoundError("Session does not exist")
            self.agent.clear_session(session_id)
            self._release_session_memory(session_id)
        finally:
            self._finish_operation()

    def end_session(self) -> None:
        """释放内存中的状态，同时保留所有已提交记录。"""
        self._begin_operation("end_session")
        try:
            session_id = self._require_session()
            self.agent.clear_session(session_id)
            self._release_session_memory(session_id)
            self._session_id = None
        finally:
            self._finish_operation()

    def delete_session(self, session_id: str) -> bool:
        """删除已持久化的会话；若它当前处于活动状态，则一并释放。"""
        self._begin_operation("delete_session")
        try:
            key = ConversationHistory._session_key(session_id)
            store = self.agent.session_store
            if store is None:
                raise AgentApplicationError("session persistence is not configured")
            store.initialize()
            deleted = store.delete_session(key)
            if deleted:
                self.agent.session_summaries.pop(key, None)
            if deleted and self._session_id == key:
                self.agent.clear_session(key)
                self._release_session_memory(key)
                self._session_id = None
            return deleted
        finally:
            self._finish_operation()

    def close(self) -> None:
        """释放活动会话、存储以及显式持有的 MCP 运行时。"""
        if self._closed:
            return
        self._begin_operation("close")
        try:
            try:
                if self._session_id is not None:
                    self.agent.clear_session(self._session_id)
                    self._release_session_memory(self._session_id)
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
        """返回活动会话历史的不可变快照。"""
        self._begin_operation("get_session_history")
        try:
            return self.agent.get_session_history(self._require_session())
        finally:
            self._finish_operation()

    def refresh_index(self) -> Any:
        """仅在完整回答轮次之间刷新共享检索器。"""
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
    memory_settings: MemorySettings | None = None,
) -> KnowledgeAgentApplication:
    """使用已准备好的工具注册器创建 CLI/API 共享边界。

    注入的注册器仍归调用方所有。Bootstrap 可以显式移交其运行时，
    以便应用刷新快照并在关闭时释放它。
    """
    return KnowledgeAgentApplication(
        config,
        model=model,
        retriever=retriever,
        session_store=session_store,
        mcp_registry=mcp_registry,
        owned_mcp_runtime=owned_mcp_runtime,
        memory_settings=memory_settings,
    )
