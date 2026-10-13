# B 站搜索 MCP 工具

Person_A 增加本地 `browser` MCP 服务。用户明确提出在 B 站搜索、学习某个主题或观看某个作品时，Agent 从问题中提取关键词，再调用工具向运行后端的电脑发出默认浏览器打开请求。工具构造固定的 B 站搜索链接，不读取网页、视频、搜索结果或浏览器账号。

**本次仅完成开发，没有运行测试、导入检查、编译、服务或模型联调，也没有安装依赖。** 下方命令和验证项用于后续统一验收，不是已执行记录。

## 使用方式

在现有网页会话或 CLI 中输入：

```text
我想在 bili 学习数据结构
帮我在 B 站搜索 Python 装饰器教程
我想在 B站看火影忍者
我想在 bilibili 观看三体
```

这些问题走 MCP 路线，模型产生类似 `{"query":"数据结构"}` 的工具参数，目标工具为 `mcp__browser__search_bilibili`。主题参数由模型根据本次用户请求提取；意图规则负责识别路线，不从问题中拼装可直接执行的参数。

“我想在 B站看火影忍者”会按观看意图进入搜索流程，模型应提取 `{"query":"火影忍者"}`。支持“看、看看、看一下、观看”，沿用学习请求的明确意图检查；当前动作止于打开搜索结果页，不自动选择集数或播放视频。

可以使用原 CLI 入口：

```powershell
person-a ask "我想在 bili 学习数据结构"
person-a chat
```

网页仍使用原 HTTP 服务；完整 JSON 和 SSE 提问接口复用同一 Agent 流程，不新增浏览器专用 HTTP 接口。浏览器工具需要聊天模型提取工具参数，因此应沿用现有 DeepSeek 配置。页面选择“普通问答”或“自动选择 Skill”均可按意图进入 MCP；手动指定知识 Skill 与 MCP 操作冲突时，沿用现有冲突提示。

普通知识问题，例如“Python 列表如何使用”，仍使用知识问答流程，不打开浏览器。仅咨询工具实现方式、否定操作或引用一段搜索指令，也不应被当作执行请求。缺少搜索主题时需要澄清，不替用户猜测主题。

打开的是**运行 Person_A 后端的电脑上的默认浏览器**。远程访问网页时，不会自动打开访问者电脑上的浏览器；没有可用桌面或默认浏览器时可能无法唤起。工具返回搜索链接和唤起请求的实际结果，不把该结果描述为“页面已经加载”或“找到了视频”。用户可自行打开返回链接查看内容。

## 启用与关闭

现有 `.[api,mcp]` 依赖已经覆盖此功能；浏览器唤起使用 Python 标准库，不增加第三方依赖。

默认运行配置 [`config/mcp_servers.json`](../config/mcp_servers.json) 保留原 MBTI 服务，并新增启用的 `browser`：

| 配置项 | 值与含义 |
| --- | --- |
| `enabled` | `true`，宿主启动时连接并发现工具；发现本身不会打开窗口 |
| `command` | `$PYTHON`，宿主替换为运行 Person_A 的 Python 解释器 |
| `args` | `-u examples/browser_mcp_server.py` |
| `cwd` | `..`，相对于 `config/` 定位项目根目录 |
| `allowed_tools` | 仅 `search_bilibili` |
| `startup_timeout_seconds` | `15` |
| `call_timeout_seconds` | `15` |
| `max_output_chars` | `4000` |

[`config/mcp_servers.browser.example.json`](../config/mcp_servers.browser.example.json) 是默认禁用的独立示例，供手动配置时参考；不要用它覆盖已有 MBTI 配置。

没有设置 `MCP_CONFIG_PATH` 时，宿主使用默认配置。如果现有配置将 `MCP_CONFIG_PATH` 显式设为空，所有 MCP 服务继续关闭，浏览器工具也不可用。只关闭本工具时，将默认配置中 `browser.enabled` 改为 `false`，其他服务可保持原样。配置调整后需按原流程重启 CLI 或 API。此次开发没有读取或修改实际 `.env`。

## 工具契约与执行边界

MCP 服务名称是 `browser`，服务端工具名称是 `search_bilibili`，宿主注册名称是 `mcp__browser__search_bilibili`。唯一参数：

```json
{"query": "数据结构"}
```

`query` 必须是 1—200 字符的非空字符串，不允许控制字符；不接受缺失参数、其他类型或额外字段。宿主在实际调用前校验，服务端收到请求后再次校验；无法通过校验的输入不打开浏览器。宿主与服务端保证“仅合法参数可执行”，不承诺模型首次生成的 JSON 一定正确。参数只表示搜索主题，不接受任意 URL、文件路径、命令或浏览器启动参数。

发布给模型的 Schema 与本地 Python 校验分别定义正则。根据用户提供的 DeepSeek 隔离测试，工具 Schema 的 `pattern` 包含 `\ud800-\udfff` 代理区范围会使模型请求在参数生成前被拒绝，即使搜索词是正常中文也会失败。因此发布的正则仅保留 C0/C1 控制字符范围；`allOf`、`not`、非空与额外字段约束保持不变。宿主和服务端仍通过 `validate_search_arguments()` 拒绝孤立代理字符。本次修复没有重新调用模型或运行测试；更新后需要重启 API 或 CLI，让 MCP 重新发布工具 Schema。

服务端将关键词进行 URL 查询编码，固定生成 `https://search.bilibili.com/all?keyword=...`，再调用 `webbrowser.open_new_tab()`。不会把模型参数交给 Shell，不支持切换网站、运行代码或访问任意本地文件。

专用浏览器执行分支关闭模型传输层重试，最多发送两次模型请求：第一次生成参数，必要时再修正一次；它不意味着执行失败后重试。只有模型给出一个有效的目标工具调用，宿主才允许执行。实际调用已发生后，无论返回成功、失败还是无法确认结果，本轮不再次打开。

**最终回答由程序根据工具的结构化结果生成，不再调用模型改写。** 模型产生的 `{"query":"数据结构"}` 是调用参数；MCP 返回的结构化 JSON 是实际执行结果，二者不能混为一谈。程序只依据执行结果报告请求状态和链接，不声称已看到网页内容。客户端断线或宿主超时也不能证明浏览器未打开；本功能没有新增请求 ID、断线恢复或自动重发。

## 文件职责

| 文件或模块 | 职责 |
| --- | --- |
| [`src/browser_mcp/`](../src/browser_mcp/) | 参数校验、固定搜索 URL、浏览器唤起和 stdio MCP 工具 |
| [`examples/browser_mcp_server.py`](../examples/browser_mcp_server.py) | 本地源码包的服务启动入口 |
| [`config/mcp_servers.json`](../config/mcp_servers.json) | 启用 browser，并保留 MBTI 配置 |
| [`config/mcp_servers.browser.example.json`](../config/mcp_servers.browser.example.json) | 默认禁用的浏览器工具示例 |
| [`agent/browser_actions.py`](../src/knowledge_agent/agent/browser_actions.py) | B 站搜索意图、操作请求与咨询、否定和引用的边界 |
| [`agent/intent.py`](../src/knowledge_agent/agent/intent.py)、[`agent/router.py`](../src/knowledge_agent/agent/router.py) | 接入搜索意图、工具可用性和 MCP 路线 |
| [`agent/browser_execution.py`](../src/knowledge_agent/agent/browser_execution.py) | 专用单动作分支、至多两次模型请求、一次实际调用和程序生成最终回答 |
| [`agent/model_loop.py`](../src/knowledge_agent/agent/model_loop.py) | 将浏览器操作交给专用执行分支，其他问答继续使用原模型循环 |
| [`mcp/`](../src/knowledge_agent/mcp/) 与 [`tools/registry.py`](../src/knowledge_agent/tools/registry.py) | 原有工具发现、白名单、参数校验、调用和资源清理 |

## 待统一验收

- [ ] 默认配置能发现 MBTI 与 browser 工具；browser 禁用或 MCP 全局关闭时不执行浏览器操作。
- [ ] 单次 ask、chat、完整 JSON 和 SSE 接口均能走同一自然语言到 MCP 的流程。
- [ ] “我想在 bili 学习数据结构”路由到指定工具，模型仅提取用户表达的主题。
- [ ] “我想在 B站看火影忍者”与“我想在 bilibili 观看三体”进入搜索；缺少作品名称时澄清，否定、引用、询问用法不触发，回答不宣称已播放。
- [ ] 普通知识问答、实现咨询、否定指令、引用示例和无搜索主题请求不会误开窗口。
- [ ] 中文、英文、空格与标点正确编码；宿主和服务端均拒绝缺失、空白、超长、控制字符、错误类型及额外字段。
- [ ] 无效参数至多修正一次；单次响应包含多个调用时不执行；重复工具请求、调用失败和调用超时不会导致本轮重复打开。
- [ ] 实际调用后的最终回答由程序依据结构化结果生成，不额外消耗模型调用或接受模型虚构的执行结果。
- [ ] 浏览器返回成功、返回失败或抛错时均如实报告；不误称页面已加载、获得搜索结果或读取视频。
- [ ] 断连、后台调用结束、MCP 退出及服务关闭仍沿用已有生命周期；不自动重发请求。
- [ ] 手动知识 Skill 冲突提示、知识检索和现有 MBTI 问卷流程保持可用。
- [ ] 最后由用户安排一次真实桌面唤起联调，观察实际打开次数与返回链接；在此之前不报告测试通过。

返回 [开发文档导航](README.md) 或 [项目 README](../README.md)。
