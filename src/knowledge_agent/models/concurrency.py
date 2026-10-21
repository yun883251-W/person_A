"""HTTP 工作通道的异步模型网关；CLI 无网关时沿用同步调用。"""
from contextlib import contextmanager
from contextvars import ContextVar
import asyncio
from threading import BoundedSemaphore, Lock
import inspect
import logging

_GATEWAY = ContextVar("chat_gateway", default=None)


def current_gateway():
    return _GATEWAY.get()


@contextmanager
def use_gateway(gateway):
    token = _GATEWAY.set(gateway)
    try:
        yield
    finally:
        _GATEWAY.reset(token)


def invoke_with_gateway(model, messages):
    gateway = _GATEWAY.get()
    return model.invoke(messages) if gateway is None else gateway.invoke(model, messages)


@contextmanager
def tool_slot():
    gateway = _GATEWAY.get()
    if gateway is None:
        yield
        return
    if not gateway.tools.acquire(timeout=gateway.wait_seconds):
        raise TimeoutError("MCP capacity wait expired before dispatch")
    try:
        yield
    finally:
        gateway.tools.release()


class ModelGateway:
    def __init__(self, loop, limit, wait_seconds, tool_limit=2):
        self.loop = loop
        self.limit = asyncio.Semaphore(limit)
        self.wait_seconds = wait_seconds
        self.tools = BoundedSemaphore(tool_limit)
        self.models = {}
        self.model_lock = Lock()

    def model_for(self, settings, factory):
        # HTTP 客户端只属于本网关事件循环，不借用 CLI 的跨运行时缓存。
        with self.model_lock:
            if settings not in self.models:
                self.models[settings] = factory(settings)
            return self.models[settings]

    async def close(self):
        seen = set()
        for model in self.models.values():
            candidates = [getattr(model, "root_async_client", None), getattr(model, "root_client", None)]
            # Ollama 适配器封装的 HTTP 客户端；缺少关闭接口时保持兼容。
            for name in ("_async_client", "_client"):
                candidates.append(getattr(getattr(model, name, None), "_client", None))
            for client in candidates:
                if client is None or id(client) in seen:
                    continue
                seen.add(id(client))
                close = getattr(client, "aclose", None) or getattr(client, "close", None)
                if callable(close):
                    try:
                        result = close()
                        if inspect.isawaitable(result):
                            await result
                    except Exception:
                        logging.getLogger(__name__).warning("model_client_close_failed")
        self.models.clear()

    async def _invoke(self, model, messages):
        await asyncio.wait_for(self.limit.acquire(), timeout=self.wait_seconds)
        try:
            # 原生异步客户端；不在调度循环内执行同步 invoke。
            return await model.ainvoke(messages)
        finally:
            self.limit.release()

    def invoke(self, model, messages):
        # 仅从同步工作通道调用，事件循环继续接收和推进其他模型请求。
        return asyncio.run_coroutine_threadsafe(self._invoke(model, messages), self.loop).result()
