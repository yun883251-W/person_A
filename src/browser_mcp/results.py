"""读取公开搜索页；不登录、不下载、不执行页面提供的指令。"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from .contracts import build_search_url, validate_search_arguments


class VideoSearchError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def canonical_video_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 2048:
        return None
    try:
        url = urlsplit("https:" + value if value.startswith("//") else value)
        if url.scheme != "https" or url.hostname != "www.bilibili.com" or url.username or url.password or url.port not in (None, 443):
            return None
        match = re.fullmatch(r"/video/(BV[0-9A-Za-z]{10}|av[0-9]+)/?", url.path)
        return "https://www.bilibili.com/video/" + match[1] if match else None
    except ValueError:
        return None


def clean_items(rows: list[Any], limit: int) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows[:100]:
        if not isinstance(row, dict) or not isinstance(row.get("title"), str):
            continue
        url = canonical_video_url(row.get("url"))
        title = " ".join(row["title"].split())
        if not url or url in seen or not title or len(title) > 200 or any(ord(c) < 32 or 0x7f <= ord(c) <= 0x9f or 0xd800 <= ord(c) <= 0xdfff for c in title):
            continue
        seen.add(url)
        items.append({"title": title, "url": url})
        if len(items) == limit:
            break
    return items


def read_video_results(query: str, limit: int) -> dict[str, Any]:
    checked = validate_search_arguments({"query": query, "limit": limit})
    query = checked["query"]
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError:
        raise VideoSearchError("browser_dependency_missing", "尚未安装浏览器读取依赖，请安装项目 browser 可选依赖及 Playwright Chromium。") from None
    # 独立临时上下文，不借用个人浏览器的登录信息；总等待小于 MCP 默认45秒预算。
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, timeout=10000)
            try:
                context = browser.new_context(locale="zh-CN")
                page = context.new_page()
                response = page.goto(build_search_url(query), wait_until="domcontentloaded", timeout=15000)
                if response is not None and response.status >= 400:
                    raise VideoSearchError("browser_search_blocked", "B站搜索页拒绝访问，未读取视频结果。")
                selector = '.bili-video-card a[href*="/video/"], .video-item a[href*="/video/"]'
                try:
                    page.locator(selector).first.wait_for(state="attached", timeout=10000)
                except PlaywrightTimeout:
                    raise VideoSearchError("browser_results_unavailable", "未读取到视频列表：可能暂无结果、需要登录或验证，或者页面结构已变化。") from None
                rows = page.locator(selector).evaluate_all("""links => links.slice(0,100).map(a => {
                    const card = a.closest('.bili-video-card, .video-item');
                    const heading = card?.querySelector('h3, .title');
                    return {title: heading?.getAttribute('title') || heading?.textContent || a.getAttribute('title') || a.textContent,
                            url: a.href};
                })""")
                items = clean_items(rows, limit)
                if not items:
                    raise VideoSearchError("browser_results_unavailable", "页面未提供可验证的视频标题和链接。")
                return {"status": "success", "action": "video_results_read", "site": "bilibili",
                        "query": query, "url": build_search_url(query), "limit": limit,
                        "page_verified": True, "items": items}
            finally:
                browser.close()
    except VideoSearchError:
        raise
    except PlaywrightTimeout:
        raise VideoSearchError("browser_search_timeout", "读取B站搜索页超时，未自动重试。") from None
    except Exception:
        raise VideoSearchError("browser_reader_failed", "浏览器读取失败；请确认已安装 Playwright Chromium。") from None


def verified_video_payload(payload: object, arguments: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or "limit" not in arguments:
        return None
    if payload.get("page_verified") is not True or type(payload.get("limit")) is not int:
        return None
    items = payload.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= arguments["limit"]:
        return None
    if items != clean_items(items, arguments["limit"]):
        return None
    expected = {"status": "success", "action": "video_results_read", "site": "bilibili",
                "query": arguments["query"], "url": build_search_url(arguments["query"]),
                "limit": arguments["limit"], "page_verified": True, "items": items}
    return expected if payload == expected else None
