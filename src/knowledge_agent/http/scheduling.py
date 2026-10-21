"""单进程有界协程调度；同会话 FIFO，管理操作为全局屏障。"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic

from knowledge_agent.models.concurrency import ModelGateway, use_gateway
from knowledge_agent.config import PROJECT_ROOT, _dotenv_values


class QueueFullError(RuntimeError):
    code = "request_queue_full"


class QueueTimeoutError(RuntimeError):
    code = "request_queue_timeout"


def setting(name, default, minimum, maximum):
    # 与项目其他设置保持一致：进程环境变量优先，其次项目 .env。
    dotenv = _dotenv_values(PROJECT_ROOT / ".env")
    raw_value = os.environ.get(name, dotenv.get(name, str(default)))
    # 并发配置都是整数；允许 `.env` 在数值后写说明性行内注释。
    value = int(raw_value.split("#", 1)[0].strip())
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass
class Job:
    key: object | None
    function: object
    future: Future
    submitted: float


class RequestScheduler:
    def __init__(self, lanes, admin_submit):
        self.lanes = lanes
        self.admin_submit = admin_submit
        self.queue_limit = setting("API_QUEUE_CAPACITY", 128, 1, 10000)
        self.queue_seconds = setting("API_QUEUE_TIMEOUT_SECONDS", 60, 1, 3600)
        self.model_limit = setting("API_MODEL_CONCURRENCY", 8, 1, 2500)
        self.model_wait = setting("API_MODEL_WAIT_SECONDS", 60, 1, 3600)
        self.tool_limit = setting("API_MCP_CONCURRENCY", 2, 1, 64)
        self.lock = Lock()
        self.outstanding = 0
        self.running = 0
        self.accepting = True
        self.ready = Event()
        self.thread = Thread(target=self._serve, name="agent-scheduler", daemon=True)
        self.thread.start()
        self.ready.wait()

    def _serve(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.pending = []
        self.active = set()
        self.keys = set()
        self.free = list(self.lanes)
        self.exclusive = False
        self.gateway = ModelGateway(self.loop, self.model_limit, self.model_wait, self.tool_limit)
        self.ready.set()
        try:
            self.loop.run_until_complete(self._pump())
            self.loop.run_until_complete(self.gateway.close())
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.run_until_complete(self.loop.shutdown_default_executor())
        finally:
            self.loop.close()

    def submit(self, key, function):
        future = Future()
        with self.lock:
            if not self.accepting:
                raise RuntimeError("service runtime is closed")
            if self.outstanding >= len(self.lanes) + self.queue_limit:
                raise QueueFullError("The request queue is full")
            self.outstanding += 1
            # 禁止调用方取消后提前释放会话顺序；HTTP 断开仍由服务收尾。
            future.set_running_or_notify_cancel()
            self.loop.call_soon_threadsafe(self.pending.append, Job(key, function, future, monotonic()))
        return future

    def _finish(self, job, *, result=None, error=None):
        with self.lock:
            self.outstanding -= 1
        if error is None:
            job.future.set_result(result)
        else:
            job.future.set_exception(error)

    async def _run(self, job, lane):
        def execute():
            with use_gateway(self.gateway):
                return job.function(lane)
        try:
            submitted = self.admin_submit(execute) if lane is None else lane.submit(execute)
            result = await asyncio.wrap_future(submitted)
            self._finish(job, result=result)
        except BaseException as exc:
            self._finish(job, error=exc)
        finally:
            with self.lock:
                self.running -= 1
            if lane is None:
                self.exclusive = False
            else:
                self.keys.remove(job.key)
                self.free.append(lane)

    async def _pump(self):
        while True:
            now = monotonic()
            for job in list(self.pending):
                if now - job.submitted >= self.queue_seconds:
                    self.pending.remove(job)
                    self._finish(job, error=QueueTimeoutError("Request expired before execution"))
            if not self.exclusive:
                for job in list(self.pending):
                    if job.key is None:
                        if self.active:
                            break
                        lane = None
                        self.exclusive = True
                    else:
                        if not self.free:
                            break
                        if job.key in self.keys:
                            continue
                        lane = self.free.pop()
                        self.keys.add(job.key)
                    self.pending.remove(job)
                    with self.lock:
                        self.running += 1
                    task = asyncio.create_task(self._run(job, lane))
                    self.active.add(task)
                    task.add_done_callback(self.active.discard)
                    if lane is None:
                        break
            with self.lock:
                done = not self.accepting and self.outstanding == 0
            if done and not self.active:
                return
            await asyncio.sleep(0.02)

    def snapshot(self):
        with self.lock:
            return {"mode": "bounded_async", "worker_lanes": len(self.lanes),
                    "outstanding": self.outstanding, "queue_capacity": self.queue_limit,
                    "running": self.running, "waiting": max(0, self.outstanding - self.running),
                    "queue_timeout_seconds": self.queue_seconds,
                    "model_concurrency": self.model_limit, "mcp_concurrency": self.tool_limit,
                    "admission_capacity": len(self.lanes) + self.queue_limit,
                    "accepting": self.accepting}

    def close(self):
        with self.lock:
            self.accepting = False
        self.thread.join()
