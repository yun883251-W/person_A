# 浏览器 MCP

浏览器能力只处理 B 站搜索，并通过固定工具 `search_bilibili` 暴露。它不接受任意 URL、JavaScript、Shell 命令或文件路径。

## 两种模式

- **open**：向后端所在电脑的默认浏览器发送打开搜索页请求。
- **results**：使用无头 Chromium 读取公开搜索结果，返回 1 至 10 条真实标题与链接。

普通“搜索 Python 装饰器”使用 open；明确要求数量、标题或链接时使用 results，未指定数量默认 5 条。

## 授权边界

- 当前输入必须明确 B 站和具体主题。
- 模型提取的主题必须是当前输入中的连续原文，不能根据历史或常识补写。
- 参数在 Agent、MCP 宿主和服务端三层校验。
- 单轮只派发一次，不自动重试外部操作。
- 否定、假设、引用示例、讨论命令本身以及组合歧义不会获得执行权限。
- 服务不登录、不绕过验证码、不播放、不下载，也不对视频内容作事实背书。

## 安装

open 模式使用 `.[api,mcp]`。results 模式额外需要：

```powershell
python -m pip install -e ".[api,browser]"
python -m playwright install chromium
```

站点不可访问、选择器失效、验证码或依赖缺失会返回结构化失败原因，而不是生成虚构结果。
