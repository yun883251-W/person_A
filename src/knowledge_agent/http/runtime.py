"""有界协程调度与隔离 Agent 通道；管理操作独占，只读查询独立。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock, get_ident
from typing import Any

from knowledge_agent.observability.trace import TraceSkillMode, observe_trace_progress
from ..knowledge.manager import KnowledgeBaseService, IndexStatus, load_knowledge_base_config

RuntimeFactory = Callable[[], Any]
StoreFactory = Callable[[], Any]
ApplicationFactory = Callable[[Any], Any]


class ServiceBusyError(RuntimeError):
    """已有业务操作正在运行时抛出。"""

    code = "service_busy"
    status_code = 503

    def __init__(self) -> None:
        super().__init__("The service is busy; try again after the current request finishes")


class ServiceRuntime:
    """每个通道持有自己的线程资源，外层调度器维护会话顺序与容量。"""

    def __init__(
        self,
        store_factory: StoreFactory | None = None,
        application_factory: ApplicationFactory | None = None,
        *, concurrent_sessions: bool | None = None,
    ) -> None:
        self._concurrent_sessions = (
            store_factory is None and application_factory is None
            if concurrent_sessions is None else concurrent_sessions
        )
        self._scheduler = None
        self._shared_mcp = None
        self._shared_mcp_error = None
        self._uses_default_application = application_factory is None
        self._lanes = []
        self._store_factory = store_factory or self._default_store_factory
        self._application_factory = (
            application_factory or self._default_application_factory
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="knowledge-agent-service",
        )
        self._read_pending = 0
        self._read_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="knowledge-agent-reader")
        self._reader_factory: StoreFactory | None = None
        self._state_lock = Lock()
        self._worker_thread_id: int | None = None
        self._store: Any = None
        self._application: Any = None
        self._knowledge_service: KnowledgeBaseService | None = None
        self._initialized = False
        self._busy = False
        self._closing = False
        self._closed = False

    @staticmethod
    def _default_store_factory() -> Any:
        from ..config import get_session_store_settings
        from knowledge_agent.session.store import SQLiteSessionStore

        settings = get_session_store_settings()
        return SQLiteSessionStore(
            settings.db_path,
            lock_timeout=settings.lock_timeout,
        )

    @staticmethod
    def _default_application_factory(store: Any) -> Any:
        from ..mcp.bootstrap import create_configured_application

        return create_configured_application(session_store=store)

    def initialize(self) -> None:
        """在专用工作线程中构造并初始化存储。"""
        with self._state_lock:
            if self._closed or self._closing:
                raise RuntimeError("service runtime is closed")
            if self._initialized:
                return

        try:
            self._executor.submit(self._initialize_on_worker).result()
            if self._uses_default_application:
                self._initialize_shared_mcp()
            if self._concurrent_sessions:
                from .scheduling import RequestScheduler, setting
                count = setting("API_SESSION_CONCURRENCY", 4, 1, 128)
                for _ in range(count):
                    lane = ServiceRuntime(self._store_factory, self._application_factory, concurrent_sessions=False)
                    self._lanes.append(lane)
                    lane.initialize()
                self._scheduler = RequestScheduler(self._lanes, self._submit_serial)
        except BaseException:
            for lane in self._lanes:
                lane.close()
            self._executor.submit(self._close_on_worker).result()
            self._close_shared_mcp()
            with self._state_lock:
                self._closing = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._read_executor.shutdown(wait=True, cancel_futures=True)
            with self._state_lock:
                self._closed = True
                self._closing = False
            raise

        with self._state_lock:
            self._initialized = True

    def _initialize_shared_mcp(self) -> None:
        """在接受请求前只启动一套 MCP；失败不阻断聊天和知识问答。"""
        from ..config import get_mcp_settings
        from ..mcp.client import McpRuntime
        from ..mcp.bootstrap import create_configured_application

        runtime = None
        try:
            runtime = McpRuntime(get_mcp_settings().config_path)
            runtime.start()
        except Exception:
            self._shared_mcp_error = "mcp_startup_failed"
            logging.getLogger(__name__).exception("Shared MCP startup failed; tools unavailable")
            if runtime is not None:
                runtime.close()
            runtime = McpRuntime(None)
            runtime.start()
        self._shared_mcp = runtime
        self._application_factory = lambda store: create_configured_application(
            session_store=store, mcp_runtime=runtime,
        )

    def _close_shared_mcp(self) -> None:
        runtime, self._shared_mcp = self._shared_mcp, None
        if runtime is not None:
            runtime.close()

    def _initialize_on_worker(self) -> None:
        self._worker_thread_id = get_ident()
        store = self._store_factory()
        try:
            store.initialize()
        except BaseException:
            store.close()
            raise
        self._store = store
        from knowledge_agent.session.store import SQLiteSessionStore

        if isinstance(store, SQLiteSessionStore):
            path, timeout = store.db_path.resolve(), store.lock_timeout
            self._reader_factory = lambda: SQLiteSessionStore(path, lock_timeout=timeout, read_only=True)

    def _require_worker_thread(self) -> None:
        if get_ident() != self._worker_thread_id:
            raise RuntimeError("runtime resources may only be used by the worker thread")

    @property
    def store(self) -> Any:
        """在执行已提交任务时返回归工作线程所有的存储实例。"""
        self._require_worker_thread()
        if self._store is None:
            raise RuntimeError("service runtime is not initialized")
        return self._store

    @property
    def application(self) -> Any:
        """在所属工作线程上延迟创建 Agent 应用。"""
        self._require_worker_thread()
        if self._store is None:
            raise RuntimeError("service runtime is not initialized")
        if self._application is None:
            self._application = self._application_factory(self._store)
        return self._application

    @property
    def knowledge_service(self) -> KnowledgeBaseService:
        """在所属工作线程上延迟构造文件系统服务。"""
        self._require_worker_thread()
        if self._knowledge_service is None:
            self._knowledge_service = KnowledgeBaseService(
                load_knowledge_base_config(),
                refresh_callback=self._refresh_index_on_worker,
            )
        return self._knowledge_service

    def _refresh_index_on_worker(self) -> Any:
        self._require_worker_thread()
        result = self.application.refresh_index()
        # 调用方处于独占管理屏障，所有问答通道此时空闲。
        failure = None
        for lane in self._lanes:
            try:
                lane._executor.submit(lane._refresh_existing_index).result()
            except Exception as exc:
                failure = exc
        if failure is not None:
            raise failure
        return result

    def _refresh_existing_index(self):
        self._require_worker_thread()
        if self._application is not None:
            try:
                self._application.refresh_index()
            except Exception:
                # 发布成功而刷新失败时，不能继续复用旧索引的应用。
                self._close_on_worker()
                self._initialize_on_worker()
                raise

    def submit(self, function, /, *args, **kwargs):
        if self._scheduler is not None:
            with self._state_lock:
                if self._closing or self._closed:
                    raise RuntimeError("service runtime is closed")
            # 管理/删除/清空等操作作为全局屏障，不与任何问答并行。
            return self._scheduler.submit(None, lambda _lane: function(*args, **kwargs))
        return self._submit_serial(function, *args, **kwargs)

    def _submit_serial(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        """向专用工作线程提交一个完整的业务操作。"""
        with self._state_lock:
            if not self._initialized:
                raise RuntimeError("service runtime is not initialized")
            if self._closed or (self._closing and self._scheduler is None):
                raise RuntimeError("service runtime is closed")
            if self._busy:
                raise ServiceBusyError()
            self._busy = True

            try:
                future = self._executor.submit(function, *args, **kwargs)
            except BaseException:
                self._busy = False
                raise

        # 极短的操作此时可能已经完成。在状态锁之外注册回调，
        # 因为 Future 随后会立即调用回调。
        future.add_done_callback(self._operation_finished)
        return future

    def _operation_finished(self, _future: Future[Any]) -> None:
        with self._state_lock:
            self._busy = False

    @property
    def busy(self) -> bool:
        """返回已提交的业务操作是否仍在运行。"""
        with self._state_lock:
            return self._busy or (self._scheduler is not None and self._scheduler.snapshot()["outstanding"] > 0)

    @property
    def closing(self) -> bool:
        """返回运行时是否已开始关闭。"""
        with self._state_lock:
            return self._closing

    async def run(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """提交一个完整的业务操作并异步等待其完成。"""
        future = self.submit(function, *args, **kwargs)
        return await self._wait_for_future(future)

    async def run_auxiliary(self, function, /, *args, **kwargs):
        """无会话写入的辅助模型操作共享通道和模型预算，不作为管理屏障。"""
        if self._scheduler is None:
            return await self.run(function, *args, **kwargs)
        with self._state_lock:
            if self._closing or self._closed:
                raise RuntimeError("service runtime is closed")
        from uuid import uuid4
        future = self._scheduler.submit(("auxiliary", uuid4().hex),
                                        lambda _lane: function(*args, **kwargs))
        return await self._wait_for_future(future)

    @staticmethod
    async def _wait_for_future(future: Future[Any]) -> Any:
        wrapped = asyncio.wrap_future(future)
        try:
            # 取消 HTTP 等待方不得取消已经交给专用工作线程的任务。
            # 忙碌状态由任务的完成回调负责切换。
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            # 当 HTTP 等待方已不存在时，稍后取回异常。
            wrapped.add_done_callback(ServiceRuntime._consume_future_exception)
            raise

    @staticmethod
    def _consume_future_exception(future: asyncio.Future[Any]) -> None:
        if not future.cancelled():
            future.exception()

    def submit_store(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        """提交一个首个参数为会话存储的操作。"""
        return self.submit(self._call_with_store, function, args, kwargs)

    async def run_store(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """执行会话存储操作并异步等待其完成。"""
        return await self._wait_for_future(
            self.submit_store(function, *args, **kwargs)
        )

    async def read_store(self, function: Callable[..., Any]) -> Any:
        """读取已提交的会话快照，使用独立只读连接，不占用 Agent 工作线程。"""
        if self._reader_factory is None:
            # 自定义存储仍遵守原线程协议，不假设它支持并发。
            return await self.run_store(function)
        with self._state_lock:
            if not self._initialized or self._closing or self._closed:
                raise RuntimeError("service runtime is not available")
            if self._read_pending >= 128:
                from .scheduling import QueueFullError
                raise QueueFullError("Read request queue is full")
            self._read_pending += 1
            try:
                future = self._read_executor.submit(self._read_store_snapshot, function)
            except BaseException:
                self._read_pending -= 1
                raise
        future.add_done_callback(self._read_finished)
        return await self._wait_for_future(future)

    def _read_finished(self, _future):
        with self._state_lock:
            self._read_pending -= 1

    def _read_store_snapshot(self, function: Callable[..., Any]) -> Any:
        store = self._reader_factory()
        try:
            store.initialize()
            return function(store)
        finally:
            store.close()

    def _call_with_store(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return function(self.store, *args, **kwargs)

    def submit_application(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        """提交操作，并按需传入 Agent 应用。"""
        return self.submit(self._call_with_application, function, args, kwargs)

    async def run_application(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """执行 Agent 应用操作并异步等待其完成。"""
        return await self._wait_for_future(
            self.submit_application(function, *args, **kwargs)
        )

    def _call_with_application(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return function(self.application, *args, **kwargs)

    async def run_knowledge(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """在共享服务工作线程上执行一次知识库操作。"""
        return await self.run(self._call_with_knowledge, function, args, kwargs)

    def _call_with_knowledge(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return function(self.knowledge_service, *args, **kwargs)

    async def get_knowledge_status(self) -> IndexStatus:
        return await self.run(self._get_knowledge_status_on_worker)

    def _get_knowledge_status_on_worker(self) -> IndexStatus:
        from dataclasses import replace

        result = self.knowledge_service.get_index_status()
        application = self._application
        if application is None:
            return result
        try:
            active_version = application.agent.retriever.index_version
        except Exception:
            return result
        return replace(result, active_version=active_version)

    async def ask_session(
        self,
        session_id: str,
        question: str,
        *,
        skill_id: str | None = None,
        skill_mode: TraceSkillMode | None = None,
        response_mode: str = "auto",
    ) -> Any | None:
        """在已有的持久化会话中提出一个问题。"""
        return await self._wait_for_future(self.submit_ask_session(
            session_id, question, skill_id=skill_id, skill_mode=skill_mode, response_mode=response_mode,
        ))

    def submit_ask_session(
        self, session_id: str, question: str, *,
        skill_id: str | None = None,
        skill_mode: TraceSkillMode | None = None,
        response_mode: str = "auto",
        progress_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> Future[Any]:
        """只提交一次；JSON 与 SSE 共用有界队列及会话 FIFO。"""
        if self._scheduler is not None:
            with self._state_lock:
                if self._closing or self._closed:
                    raise RuntimeError("service runtime is closed")
            return self._scheduler.submit(("session", session_id), lambda lane: lane._ask_session_on_worker(
                session_id, question, skill_id, skill_mode, response_mode, progress_observer,
            ))
        return self.submit(
            self._ask_session_on_worker, session_id, question, skill_id, skill_mode,
            response_mode, progress_observer,
        )

    def _ask_session_on_worker(
        self, session_id: str, question: str, skill_id: str | None = None,
        skill_mode: TraceSkillMode | None = None,
        response_mode: str = "auto",
        progress_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> Any | None:
        self._require_worker_thread()
        if self.store.get_session(session_id) is None:
            return None

        application = self.application
        with observe_trace_progress(progress_observer):
            application.resume_session(session_id)
            try:
                options: dict[str, Any] = {"skill_id": skill_id, "response_mode": response_mode}
                if skill_mode is not None:
                    options["skill_mode"] = skill_mode
                return application.ask(question, **options)
            finally:
                application.end_session()

    async def load_turns(
        self, session_id: str, *, limit: int, offset: int
    ) -> Any | None:
        """按时间顺序加载一页；会话不存在时返回 ``None``。"""
        def read_turns(store):
            if store.get_session(session_id) is None:
                return None
            return store.load_turns(session_id, limit=limit, offset=offset)

        return await self.read_store(read_turns)

    async def replace_memory(self, session_id: str, **values) -> None:
        """与问答串行执行；成功后释放缓存，下次从持久化边界恢复。"""
        await self.run(self._replace_memory_on_worker, session_id, values)

    def _replace_memory_on_worker(self, session_id: str, values: dict) -> None:
        self._require_worker_thread()
        self.store.replace_memory(session_id, **values)
        application = self._application
        if application is not None and application.session_id == session_id:
            application.end_session()

    async def clear_session(self, session_id: str) -> bool:
        """清空已持久化的问答轮，并释放对应的内存历史。"""
        return await self.run(self._clear_session_on_worker, session_id)

    def _clear_session_on_worker(self, session_id: str) -> bool:
        self._require_worker_thread()
        cleared = self.store.clear_session(session_id)
        application = self._application
        if cleared and application is not None and application.session_id == session_id:
            application.end_session()
        return cleared

    async def delete_session(self, session_id: str) -> bool:
        """删除已持久化的会话，并释放对应的内存历史。"""
        return await self.run(self._delete_session_on_worker, session_id)

    def _delete_session_on_worker(self, session_id: str) -> bool:
        self._require_worker_thread()
        deleted = self.store.delete_session(session_id)
        application = self._application
        if deleted and application is not None and application.session_id == session_id:
            application.end_session()
        return deleted

    def concurrency_status(self):
        if self._scheduler is None:
            result = {"mode": "serial", "worker_lanes": 1, "outstanding": int(self.busy)}
        else:
            result = self._scheduler.snapshot()
        if self._shared_mcp is not None:
            result["mcp_startup_error"] = self._shared_mcp_error
            result["mcp_servers"] = [
                {"name": item.server_name, "state": item.state.value}
                for item in self._shared_mcp.list_server_statuses()
            ]
        return result

    def close(self) -> None:
        """等待在途任务结束、关闭自有资源并停止工作线程。"""
        with self._state_lock:
            if self._closed or self._closing:
                return
            self._closing = True
            initialized = self._initialized

        failures = []
        try:
            if self._scheduler is not None:
                self._scheduler.close()
            for lane in self._lanes:
                try:
                    lane.close()
                except Exception as exc:
                    failures.append(exc)
            if initialized:
                try:
                    self._executor.submit(self._close_on_worker).result()
                except Exception as exc:
                    failures.append(exc)
        finally:
            try:
                self._close_shared_mcp()
            except Exception as exc:
                failures.append(exc)
            self._read_executor.shutdown(wait=True, cancel_futures=False)
            self._executor.shutdown(wait=True, cancel_futures=False)
            with self._state_lock:
                self._initialized = False
                self._closed = True
                self._closing = False
        if failures:
            raise failures[0]

    def _close_on_worker(self) -> None:
        self._require_worker_thread()
        application = self._application
        store = self._store
        self._application = None
        self._knowledge_service = None
        self._store = None
        if application is not None:
            # 应用持有注入的存储实例，并由其自行关闭。
            application.close()
        elif store is not None:
            store.close()

    def __enter__(self) -> ServiceRuntime:
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
