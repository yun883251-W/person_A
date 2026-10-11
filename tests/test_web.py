"""Isolated static routing checks; no real model, index, or session database."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from knowledge_agent import api


class _MissingSessionStore:
    def get_session(self, _session_id: str) -> None:
        return None


class _DummyRuntime:
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.store_calls = 0

    def initialize(self) -> None:
        self.initialized = True

    def close(self) -> None:
        self.closed = True

    async def run_store(self, operation: Any) -> Any:
        self.store_calls += 1
        return operation(_MissingSessionStore())


@pytest.fixture
def web_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, _DummyRuntime, Path]]:
    assets = tmp_path / "web" / "assets"
    assets.mkdir(parents=True)
    (tmp_path / "web" / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><title>本地问答</title></html>',
        encoding="utf-8",
    )
    (assets / "app.css").write_text("body { color: #123456; }", encoding="utf-8")
    (assets / "app.js").write_text('console.log("page loaded");', encoding="utf-8")
    (tmp_path / "web" / "private.txt").write_text("private-web-data", encoding="utf-8")
    (tmp_path / "private.txt").write_text("private-project-data", encoding="utf-8")
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    runtime = _DummyRuntime()
    with TestClient(api.create_app(runtime_factory=lambda: runtime)) as client:
        assert runtime.initialized
        yield client, runtime, tmp_path
    assert runtime.closed


def test_page_and_assets_do_not_access_runtime_store(web_client: Any) -> None:
    client, runtime, _ = web_client
    home = client.get("/")
    assert home.status_code == 200
    assert home.headers["content-type"].startswith("text/html")
    assert "本地问答" in home.text

    stylesheet = client.get("/assets/app.css")
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert "#123456" in stylesheet.text

    script = client.get("/assets/app.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert "page loaded" in script.text
    assert runtime.store_calls == 0


@pytest.mark.parametrize(
    "path",
    [
        "/not-an-api-route",
        "/private.txt",
        "/web/index.html",
        "/assets/missing.css",
        "/assets/%2e%2e/private.txt",
        "/assets/%2e%2e/%2e%2e/private.txt",
    ],
)
def test_static_routes_do_not_expose_other_directories(web_client: Any, path: str) -> None:
    client, runtime, _ = web_client
    response = client.get(path)
    assert response.status_code == 404
    assert "private-web-data" not in response.text
    assert "private-project-data" not in response.text
    assert runtime.store_calls == 0


def test_existing_health_docs_and_session_404_are_preserved(web_client: Any) -> None:
    client, runtime, _ = web_client
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200
    assert "/sessions" in client.get("/openapi.json").json()["paths"]
    missing = client.get("/sessions/missing-session")
    assert missing.status_code == 404
    assert missing.json()["code"] == "session_not_found"
    assert runtime.store_calls == 1


def test_missing_page_returns_404_without_breaking_api(web_client: Any) -> None:
    client, runtime, root = web_client
    (root / "web" / "index.html").unlink()
    missing = client.get("/")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Not Found"}
    assert client.get("/health").status_code == 200
    assert runtime.store_calls == 0
