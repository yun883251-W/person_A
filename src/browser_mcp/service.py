"""本机浏览器执行器：只请求打开 B 站搜索页，不抓取或读取视频。

浏览器运行在启动此 MCP 进程的电脑上。若宿主部署到远端服务器，
动作不会自动转移到用户访问前端的设备；当前功能面向本机部署。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .contracts import search_success_payload


class BrowserLaunchError(RuntimeError):
    """浏览器启动未获成功确认；不包含浏览器路径、环境或原始异常。"""

    def __init__(self, code: str) -> None:
        super().__init__("未能确认浏览器接受了打开请求；未自动重试，也未验证页面。")
        self.code = code


class BrowserService:
    """按需发出一次浏览器请求；opener 可注入，构造和导入阶段无启动动作。"""

    def __init__(self, opener: Callable[[str], bool] | None = None) -> None:
        if opener is not None and not callable(opener):
            raise TypeError("opener must be callable")
        self._opener = opener

    def search_bilibili(self, query: str) -> dict[str, Any]:
        """校验全部参数后调用一次 opener；不能据此声称页面已打开或视频已读取。"""
        payload = search_success_payload(query)
        url = payload["url"]

        try:
            opener = self._opener
            if opener is None:
                # 延迟导入和解析默认浏览器；校验失败不会触发此分支。
                import webbrowser

                opener = webbrowser.open_new_tab
            accepted = opener(url)
        except Exception:
            # 外部启动失败可能发生在动作已经提交之后；不自动重试。
            raise BrowserLaunchError("browser_launch_failed") from None
        if accepted is not True:
            raise BrowserLaunchError("browser_launch_not_accepted")

        return payload
