from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "mcp-demo.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "servers": {
                    "demo": {
                        "enabled": True,
                        "command": sys.executable,
                        "args": ["-u", "examples/mcp_demo_server.py"],
                        "cwd": str(Path(__file__).resolve().parents[3]),
                        "env_from": {},
                        "allowed_tools": ["add_numbers"],
                        "startup_timeout_seconds": 15,
                        "call_timeout_seconds": 15,
                        "max_output_chars": 4000,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_real_stdio_demo_discovery_validation_call_and_close(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from knowledge_agent.mcp.client import McpRuntime

    runtime = McpRuntime(_config(tmp_path))
    runtime.start()
    try:
        tools = runtime.list_tools()
        assert [tool.name for tool in tools] == ["mcp__demo__add_numbers"]
        result = runtime.call_tool("mcp__demo__add_numbers", {"a": 2, "b": 3})
        assert result["status"] == "success"
        assert result["structured_content"]["sum"] == 5
        assert runtime.call_tool("mcp__demo__add_numbers", {"a": True, "b": 3})["status"] == "error"
    finally:
        runtime.close()
    assert runtime.call_tool("mcp__demo__add_numbers", {"a": 1, "b": 1})["status"] == "error"
