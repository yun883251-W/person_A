"""单工作线程及资源生命周期：SQLite、Agent 与知识库操作共享线程归属。"""

from __future__ import annotations

import asyncio
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
    """Raised when a business operation is already running."""

    code = "service_busy"
    status_code = 503

    def __init__(self) -> None:
        super().__init__("The service is busy; try again after the current request finishes")


class ServiceRuntime:
    """Run all stateful service work on one long-lived worker thread."""

    def __init__(
        self,
        store_factory: StoreFactory | None = None,
        application_factory: ApplicationFactory | None = None,
    ) -> None:
        self._store_factory = store_factory or self._default_store_factory
        self._application_factory = (
            application_factory or self._default_application_factory
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="knowledge-agent-service",
        )
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
        """Construct and initialize storage in the dedicated worker thread."""
        with self._state_lock:
            if self._closed or self._closing:
                raise RuntimeError("service runtime is closed")
            if self._initialized:
                return

        try:
            self._executor.submit(self._initialize_on_worker).result()
        except BaseException:
            with self._state_lock:
                self._closing = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            with self._state_lock:
                self._closed = True
                self._closing = False
            raise

        with self._state_lock:
            self._initialized = True

    def _initialize_on_worker(self) -> None:
        self._worker_thread_id = get_ident()
        store = self._store_factory()
        try:
            store.initialize()
        except BaseException:
            store.close()
            raise
        self._store = store

    def _require_worker_thread(self) -> None:
        if get_ident() != self._worker_thread_id:
            raise RuntimeError("runtime resources may only be used by the worker thread")

    @property
    def store(self) -> Any:
        """Return worker-owned storage while executing a submitted task."""
        self._require_worker_thread()
        if self._store is None:
            raise RuntimeError("service runtime is not initialized")
        return self._store

    @property
    def application(self) -> Any:
        """Lazily create the agent application on its owning worker thread."""
        self._require_worker_thread()
        if self._store is None:
            raise RuntimeError("service runtime is not initialized")
        if self._application is None:
            self._application = self._application_factory(self._store)
        return self._application

    @property
    def knowledge_service(self) -> KnowledgeBaseService:
        """Lazily construct the filesystem service on its owning worker."""
        self._require_worker_thread()
        if self._knowledge_service is None:
            self._knowledge_service = KnowledgeBaseService(
                load_knowledge_base_config(),
                refresh_callback=self._refresh_index_on_worker,
            )
        return self._knowledge_service

    def _refresh_index_on_worker(self) -> Any:
        self._require_worker_thread()
        return self.application.refresh_index()

    def submit(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        """Submit one complete business operation to the dedicated worker."""
        with self._state_lock:
            if not self._initialized:
                raise RuntimeError("service runtime is not initialized")
            if self._closing or self._closed:
                raise RuntimeError("service runtime is closed")
            if self._busy:
                raise ServiceBusyError()
            self._busy = True

            try:
                future = self._executor.submit(function, *args, **kwargs)
            except BaseException:
                self._busy = False
                raise

        # A very short operation may already be complete here. Register outside
        # the state lock because Future then invokes the callback immediately.
        future.add_done_callback(self._operation_finished)
        return future

    def _operation_finished(self, _future: Future[Any]) -> None:
        with self._state_lock:
            self._busy = False

    @property
    def busy(self) -> bool:
        """Return whether a submitted business operation is still running."""
        with self._state_lock:
            return self._busy

    @property
    def closing(self) -> bool:
        """Return whether runtime shutdown has started."""
        with self._state_lock:
            return self._closing

    async def run(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Submit and asynchronously wait for one complete business operation."""
        future = self.submit(function, *args, **kwargs)
        return await self._wait_for_future(future)

    @staticmethod
    async def _wait_for_future(future: Future[Any]) -> Any:
        wrapped = asyncio.wrap_future(future)
        try:
            # Cancelling an HTTP waiter must not cancel work already handed to the
            # dedicated worker. Its completion callback owns the busy transition.
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            # Retrieve a later exception when there is no longer an HTTP waiter.
            wrapped.add_done_callback(ServiceRuntime._consume_future_exception)
            raise

    @staticmethod
    def _consume_future_exception(future: asyncio.Future[Any]) -> None:
        if not future.cancelled():
            future.exception()

    def submit_store(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        """Submit an operation whose first argument is the session store."""
        return self.submit(self._call_with_store, function, args, kwargs)

    async def run_store(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Run and asynchronously wait for a session-store operation."""
        return await self._wait_for_future(
            self.submit_store(function, *args, **kwargs)
        )

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
        """Submit an operation and lazily supply the agent application."""
        return self.submit(self._call_with_application, function, args, kwargs)

    async def run_application(
        self, function: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Run and asynchronously wait for an agent-application operation."""
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
        """Run one knowledge-base operation on the shared service worker."""
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
    ) -> Any | None:
        """Ask one question in an existing persisted session."""
        return await self._wait_for_future(self.submit_ask_session(
            session_id, question, skill_id=skill_id, skill_mode=skill_mode,
        ))

    def submit_ask_session(
        self, session_id: str, question: str, *,
        skill_id: str | None = None,
        skill_mode: TraceSkillMode | None = None,
        progress_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> Future[Any]:
        """Submit exactly once; JSON and SSE share worker ownership and locking."""
        return self.submit(
            self._ask_session_on_worker, session_id, question, skill_id, skill_mode,
            progress_observer,
        )

    def _ask_session_on_worker(
        self, session_id: str, question: str, skill_id: str | None = None,
        skill_mode: TraceSkillMode | None = None,
        progress_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> Any | None:
        self._require_worker_thread()
        if self.store.get_session(session_id) is None:
            return None

        application = self.application
        with observe_trace_progress(progress_observer):
            application.resume_session(session_id)
            try:
                options: dict[str, Any] = {"skill_id": skill_id}
                if skill_mode is not None:
                    options["skill_mode"] = skill_mode
                return application.ask(question, **options)
            finally:
                application.end_session()

    async def load_turns(
        self, session_id: str, *, limit: int, offset: int
    ) -> Any | None:
        """Load one chronological page, or return ``None`` if absent."""
        return await self.run(
            self._load_turns_on_worker,
            session_id,
            limit,
            offset,
        )

    def _load_turns_on_worker(
        self, session_id: str, limit: int, offset: int
    ) -> Any | None:
        self._require_worker_thread()
        if self.store.get_session(session_id) is None:
            return None
        return self.store.load_turns(session_id, limit=limit, offset=offset)

    async def clear_session(self, session_id: str) -> bool:
        """Clear persisted turns and release matching in-memory history."""
        return await self.run(self._clear_session_on_worker, session_id)

    def _clear_session_on_worker(self, session_id: str) -> bool:
        self._require_worker_thread()
        cleared = self.store.clear_session(session_id)
        application = self._application
        if cleared and application is not None and application.session_id == session_id:
            application.end_session()
        return cleared

    async def delete_session(self, session_id: str) -> bool:
        """Delete a persisted session and release matching in-memory history."""
        return await self.run(self._delete_session_on_worker, session_id)

    def _delete_session_on_worker(self, session_id: str) -> bool:
        self._require_worker_thread()
        deleted = self.store.delete_session(session_id)
        application = self._application
        if deleted and application is not None and application.session_id == session_id:
            application.end_session()
        return deleted

    def close(self) -> None:
        """Drain pending work, close owned resources, and stop the worker."""
        with self._state_lock:
            if self._closed or self._closing:
                return
            self._closing = True
            initialized = self._initialized

        try:
            if initialized:
                self._executor.submit(self._close_on_worker).result()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
            with self._state_lock:
                self._initialized = False
                self._closed = True
                self._closing = False

    def _close_on_worker(self) -> None:
        self._require_worker_thread()
        application = self._application
        store = self._store
        self._application = None
        self._knowledge_service = None
        self._store = None
        if application is not None:
            # The application owns the injected store and closes it itself.
            application.close()
        elif store is not None:
            store.close()

    def __enter__(self) -> ServiceRuntime:
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
