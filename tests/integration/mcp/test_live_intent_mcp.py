"""Explicitly opted-in real-provider plus real-stdio MCP integration probe."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

pytestmark = pytest.mark.live
_OPT_IN = "PERSON_A_LIVE_INTENT_MCP_PROBE"


def test_live_knowledge_and_demo_tool_round_trip(tmp_path: Path) -> None:
    if os.environ.get(_OPT_IN, "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip(f"未设置 {_OPT_IN}=1，跳过真实模型与 MCP 联调")

    from knowledge_agent.mcp.adapter import create_mcp_tool_registry
    from knowledge_agent.mcp.client import McpRuntime
    from knowledge_agent.agent.intent import IntentKind, identify_intent
    from knowledge_agent.agent.router import RouteKind, resolve_route
    from knowledge_agent.config import IntentSettings
    from knowledge_agent.models.provider import bind_chat_tools, initialize_chat_model, invoke_chat_model

    config = tmp_path / "mcp-demo.json"
    config.write_text(json.dumps({
        "version": 1,
        "servers": {"demo": {
            "enabled": True, "command": sys.executable,
            "args": ["-u", "examples/mcp_demo_server.py"],
            "cwd": str(Path(__file__).resolve().parents[3]), "env_from": {},
            "allowed_tools": ["add_numbers"], "startup_timeout_seconds": 15,
            "call_timeout_seconds": 15, "max_output_chars": 4000,
        }},
    }), encoding="utf-8")

    model = initialize_chat_model()
    started = time.perf_counter()
    runtime = McpRuntime(config)
    runtime.start()
    try:
        capabilities = runtime.list_tools()
        classification_calls = 0

        def classify(messages):
            nonlocal classification_calls
            classification_calls += 1
            return invoke_chat_model(model, messages, max_retries=0)

        knowledge_started = time.perf_counter()
        knowledge_intent = identify_intent(
            "Explain what dependency injection means.",
            settings=IntentSettings(mode="hybrid", browser_mode="rules"),
            available_tools=capabilities,
            model_invoker=classify,
        )
        knowledge_route = resolve_route(knowledge_intent, capabilities)
        knowledge_elapsed = time.perf_counter() - knowledge_started

        tool_started = time.perf_counter()
        tool_intent = identify_intent(
            "Call mcp__demo__add_numbers with a=2 and b=3.",
            settings=IntentSettings(mode="hybrid", browser_mode="rules"),
            available_tools=capabilities,
            model_invoker=classify,
        )
        tool_route = resolve_route(tool_intent, capabilities)
        registrations = create_mcp_tool_registry(runtime).list_registrations()
        bound = bind_chat_tools(model, [item.definition for item in registrations])
        response = invoke_chat_model(
            bound,
            [HumanMessage(content="Call mcp__demo__add_numbers with a=2 and b=3. Do not calculate it yourself.")],
            max_retries=0,
        )
        calls = list(getattr(response, "tool_calls", ()))
        assert len(calls) == 1
        assert calls[0]["name"] == "mcp__demo__add_numbers"
        result = runtime.call_tool(calls[0]["name"], calls[0]["args"])
        assert result["status"] == "success"
        assert result["structured_content"]["sum"] == 5
        tool_elapsed = time.perf_counter() - tool_started
    finally:
        runtime.close()

    elapsed = time.perf_counter() - started
    misclassifications = []
    if knowledge_intent.kind is not IntentKind.KNOWLEDGE or knowledge_route.route is not RouteKind.KNOWLEDGE:
        misclassifications.append("knowledge question did not reach knowledge route")
    if tool_intent.kind is not IntentKind.MCP_TOOL or tool_route.route is not RouteKind.MCP:
        misclassifications.append("explicit demo request did not reach MCP route")

    report = {
        "cases": [
            {
                "name": "knowledge",
                "intent": knowledge_intent.kind.value,
                "route": knowledge_route.route.value,
                "model_calls": 1,
                "elapsed_seconds": round(knowledge_elapsed, 3),
            },
            {
                "name": "demo_tool",
                "intent": tool_intent.kind.value,
                "route": tool_route.route.value,
                "model_calls": 1,
                "tool_calls": 1,
                "elapsed_seconds": round(tool_elapsed, 3),
            },
        ],
        "classifier_calls": classification_calls,
        "misclassifications": misclassifications,
        "total_elapsed_seconds": round(elapsed, 3),
    }
    print(json.dumps(report, ensure_ascii=False))
    assert classification_calls == 1
    assert not misclassifications, report
    assert elapsed < 120
