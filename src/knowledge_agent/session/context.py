"""统一可选记忆的预算管理；不裁剪当前请求、计划依赖或知识证据。"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass
class MemoryContext:
    """只接受公开消息投影；完整历史问答成对移除，保留最近追问语境。"""

    history: list[BaseMessage]
    profile: str
    profile_count: int
    profile_reason: str
    question: str = ""
    dropped_turns: int = 0
    summary_dropped: bool = False

    @property
    def history_turns(self) -> int:
        return sum(
            isinstance(left, HumanMessage) and isinstance(right, AIMessage)
            for left, right in zip(self.history, self.history[1:])
        )

    @property
    def summary_chars(self) -> int:
        return sum(len(str(m.content)) for m in self.history if self._is_summary(m))

    @staticmethod
    def _is_summary(message: BaseMessage) -> bool:
        return isinstance(message, HumanMessage) and str(message.content).startswith("SESSION_SUMMARY_DATA\n")

    def fit(self, build: Callable[[], Sequence[BaseMessage]], budget: int | None) -> None:
        """先画像、再较早问答、再摘要；未知消息（如计划依赖）必须保留。"""
        if budget is None:
            return
        if type(budget) is not int or budget < 1:
            raise ValueError("max_context_chars must be a positive integer")
        while sum(len(str(m.content)) for m in build()) > budget:
            if self.profile:
                self.profile, self.profile_count = "", 0
                self.profile_reason = "profile_budget_skipped"
                continue
            pair_indices = [
                i for i in range(len(self.history) - 1)
                if isinstance(self.history[i], HumanMessage)
                and not self._is_summary(self.history[i])
                and isinstance(self.history[i + 1], AIMessage)
                and not self.history[i + 1].tool_calls
            ]
            if pair_indices:
                # 最近两轮优先保留用于“它/刚才”等指代。更早轮次只在
                # 超预算时按词面相关性淘汰，仍按原顺序注入，不增加模型调用。
                candidates = pair_indices[:-2] or pair_indices
                terms = self._terms(self.question)
                pair_index = min(candidates, key=lambda i: (
                    len(terms & self._terms(str(self.history[i].content))), i,
                ))
                del self.history[pair_index:pair_index + 2]
                self.dropped_turns += 1
                continue
            summaries = [m for m in self.history if self._is_summary(m)]
            if summaries:
                self.history[:] = [m for m in self.history if not self._is_summary(m)]
                self.summary_dropped = True
                continue
            break

    @staticmethod
    def _terms(text: str) -> set[str]:
        words = set(re.findall(r"[a-z0-9_]+", text.lower()))
        for phrase in re.findall(r"[\u4e00-\u9fff]+", text):
            words.update(phrase[i:i + 2] for i in range(len(phrase) - 1))
        return words

    def record(self, trace, *, step_id=None) -> None:
        """只发布数量，不发布记忆正文。字符数包含注入的消息包装。"""
        if trace is None:
            return
        event = trace.start_event("memory", "program", step_id=step_id)
        trace.finish_event(
            event, "success" if self.history else "skipped",
            reason_code="memory_trimmed" if self.dropped_turns or self.summary_dropped else "memory_ready",
            history_turns=self.history_turns,
            dropped_turns=self.dropped_turns,
            summary_chars=self.summary_chars,
            memory_chars=sum(len(str(m.content)) for m in self.history),
        )
