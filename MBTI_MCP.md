# MindCare MBTI 接入

Person_A 通过本地 stdio MCP 子进程提供问卷、评分和类型描述。核心代码源自 MindCare 的 `backend/mbti`，已复制到本项目 `src/mbti_mcp/core/`，并将包内导入适配为相对导入。用户在 Person_A 的网页或 CLI 会话中答题，宿主保存进度并调用工具；本项目可以独立运行，不再依赖原 MindCare 目录，也无需启动它的 Web 服务或业务数据库。

**本次仅完成开发。未运行测试、语法检查、构建、服务或模型联调，也未安装依赖；以下启动命令与验收清单供后续使用，不是已执行记录。**

## 启动与配置

在 Person_A 根目录、运行 Person_A 的同一个 Python 环境中安装：

```powershell
python -m pip install -e ".[api,mcp]"
```

应用现有模型、索引与会话配置仍按 README 设置。MBTI 答题分支本身不调用 DeepSeek，也不检索教材，但这不代表应用的既有初始化配置可以全部省略。

网页入口保持原命令：

```powershell
python -m uvicorn knowledge_agent.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

浏览器打开 [本地页面](http://127.0.0.1:8000/)，新建会话，发送“开始 MBTI 测试”。CLI 多轮入口为：

```powershell
person-a chat
```

CLI 恢复已有会话时使用 `person-a chat --session-id ID`；`ask` 是单次请求入口，不适合逐题完成同一份问卷。

运行配置位于 `config/mcp_servers.json`：

- `mbti.enabled` 为 `true`，只允许问卷、评分、类型描述三个工具。
- `command` 为宿主识别的 `$PYTHON` 占位符，启动时替换为当前 Person_A Python 解释器的绝对路径。它不是要求用户配置的环境变量。
- 参数调用 `examples/mbti_mcp_server.py`；该启动文件补充源码包路径，再进入 `mbti_mcp.server`。
- 问卷与评分直接从本地 `mbti_mcp.core` 导入，没有外部项目路径或 `--mindcare-root` 启动参数；移动整个 Person_A 目录时无需调整 MindCare 路径。
- `cwd: ".."` 相对于配置文件目录解析，在当前布局中得到 Person_A 根目录。

`MCP_CONFIG_PATH` 的含义：

| 配置情况 | 行为 |
| --- | --- |
| 环境变量和 `.env` 都没有该键 | 使用项目自带的 `config/mcp_servers.json` |
| `MCP_CONFIG_PATH=./config/mcp_servers.json` | 使用已启用 MBTI 的运行配置 |
| `MCP_CONFIG_PATH=`，即显式空值 | 关闭 MCP |
| 指向其他 JSON 配置 | 只使用该文件配置的服务，不自动合并默认配置 |

本次没有覆盖实际 `.env`。如果旧文件中存在空值，需要自行改成运行配置路径才能启用。`config/mcp_servers.mbti.example.json` 保留为默认禁用的示例；它不会自动启动。正常使用由 API/CLI 管理 MCP 子进程的启动和关闭，无需另开终端运行服务。若使用自行维护的旧 MCP 配置，应删除 MBTI 启动参数中的 `--mindcare-root` 及其路径值，只保留本项目启动文件；修改配置后重启宿主。

## 使用流程

```text
开始 MBTI 测试
  → 获取第一页题目（MCP）
  → 逐题接收用户 1—5 的明确答案
  → 保存进度，必要时获取下一页（MCP）
  → 48 题全部作答后评分（MCP）
  → 展示类型、四维得分比例和平局说明
  → 可继续查询类型描述（MCP）
```

每次展示一道题。网页可点击选项，CLI 可直接输入 `1` 至 `5`，也支持问卷中的选项文字。选项不是 A/B：1 为完全不符合，2 为不太符合，3 为中立/不确定，4 为比较符合，5 为完全符合。

| 用户输入 | 含义 |
| --- | --- |
| 开始 MBTI 测试 | 开始答题；已有进行中的测试按现有进度提示 |
| 暂停 MBTI 测试 | 暂停并保留已有答案 |
| 继续 MBTI 测试 | 恢复当前待答题目；不会替用户回答或跳题 |
| 退出 MBTI 测试 | 结束本次作答，历史保留 |
| 查看 MBTI 结果 | 查看当前测试的结果或完成状态 |
| 重新开始 MBTI 测试 | 开始新一份测试，旧问答仍保留在历史中 |
| 介绍 INTJ 的特点 | 查询指定四字母类型的描述，不据此认定用户属于该类型 |

历史答题卡只供回看，不能作为当前题目重复提交。答题卡携带测试与题目标识，由宿主校验；暂停、退出、切换会话或已提交后，旧卡不应继续改变当前进度。发送其他知识问题仍走原知识问答流程，测试进度保留。

## 模块与数据归属

| 模块 | 职责 |
| --- | --- |
| `agent/assessment.py` | 识别明确答题操作，生成下一轮状态，编排三个固定 MCP 工具 |
| `agent/runner.py` | 将答题分支与已有意图/路由、执行记录、问答提交整合 |
| `mcp/client.py`、`mcp/adapter.py` | 管理外部服务生命周期、白名单、参数校验、调用预算和结果转换 |
| `mcp/bootstrap.py` | API 与 CLI 共用的 MCP 运行时启动、注册器组装与关闭 |
| `mbti_mcp/core/` | 内置从 MindCare 复制并适配的题库、评分与类型描述，包内使用相对导入 |
| `mbti_mcp/service.py` | 直接导入本地核心，分页问卷、严格校验并调用内置评分函数 |
| `mbti_mcp/server.py` | 使用官方 MCP Python SDK 暴露 stdio 工具 |
| `session_store.py` | 在现有 SQLite 问答记录中保存与恢复测试状态 |
| `web/assets/app.js`、`app.css` | 展示答题卡、进度、选项与四维结果 |

相关目录如下：

```text
src/
├── knowledge_agent/mcp/
│   ├── __init__.py
│   ├── client.py
│   ├── adapter.py
│   └── bootstrap.py
└── mbti_mcp/
    ├── __init__.py
    ├── server.py
    ├── service.py
    └── core/
        ├── __init__.py
        ├── questions.py
        ├── scoring.py
        └── descriptions.py
```

`core/` 中四个文件的来源保留为 MindCare；本项目维护的是已复制的独立代码，原项目后续修改不会自动同步。题库、评分和类型描述均随 Person_A 一起保存和运行。

状态保存在现有 `turns.result_json.assessment_state` 中，包含题库版本、评分版本、已提交答案、已加载题目、测试状态和结果。没有新增数据库服务或答题表，也不会写入 MindCare 的数据库。

恢复会话时单独读取最新答题快照，不依赖最近几轮模型上下文，因此超过模型历史窗口的测试仍可续答。状态与该轮问答一起提交；SQLite 写入失败时不能把本轮答案当作已经保存。请求断连后应先重新读取会话历史，再决定是否重发。清空历史或删除会话也会删除其中的测试进度与结果。

普通答题、评分、类型描述采用程序调用 MCP，`model_calls` 不因这些步骤增加；实际执行的 MCP 调用纳入 `tool_calls` 和执行记录。仅保存答案、暂停或恢复已加载题目时，可能无需新的工具调用。知识问答仍按原规则调用检索工具与 DeepSeek。

## 三个 MCP 工具

宿主公开名加上 `mcp__mbti__` 前缀；服务端远程名如下：

| 工具 | 输入 | 主要输出 |
| --- | --- | --- |
| `get_mbti_questionnaire` | `offset` 为 0—47 的整数；`limit` 为 1—12 的整数，默认 12 | 题目 ID/文本、五个选项、总题数、下一页位置、题库与评分版本 |
| `score_mbti_answers` | 完整 `answers` 对象与 `questionnaire_version` | 类型、四维原始分/比例、各维是否平局、实际默认字母、版本 |
| `get_mbti_description` | 四字母 `type_code`，如 `INTJ` | 本地内置的类型名称、特点、优势等一般性描述，内容源自 MindCare |

评分严格要求字符串题号 `"0"` 至 `"47"`，以及整数 1—5。缺题、额外题号、前导零题号、布尔值、小数与字符串分值均被拒绝；不会像原宽松校验那样转换数值或对缺题补默认答案。问卷页不公开计分方向；版本由本地内置的数据或评分文件内容生成，分页和评分时校验版本一致。

计分使用从 MindCare 复制到 `core/scoring.py` 的函数。当前实际平局默认字母为 EI→I、SN→S、TF→F、JP→J，结果同时返回平局标记；不能把默认字母解释成明确倾向。维度百分比只是该量表得分比例，不是模型置信度或诊断概率。这是项目中的自评问卷，不宣称是官方 MBTI 测评或心理诊断。类型描述省略原 `overview` 中无出处的人口比例。

服务使用项目声明的 `mcp>=1.26,<2.0`，参考 [官方 SDK v1.26 文档](https://github.com/modelcontextprotocol/python-sdk/blob/v1.26.0/README.md)。业务错误通过 MCP 工具错误返回，成功结果提供结构化 JSON；stdout 专用于协议。模块导入不自动启动服务，业务核心固定使用本地 `mbti_mcp.core`，工具调用参数不能切换代码来源。

## 待统一验收

以下均未在本次执行：

- 在所用 Python 环境安装声明依赖，确认默认配置启用与显式空值关闭均符合预期；内置核心文件缺失、依赖缺失或服务不可用时给出可理解的失败提示。
- 在不提供原 MindCare 目录的环境中启动 Person_A，确认问卷、评分和描述均从本地 `core/` 运行；移动整个项目后，默认配置仍使用当前 Python 与本地启动文件。
- 完成工具发现和三工具调用，确认白名单、命名空间、严格参数校验及错误结果处理。
- 完成全部 48 题，与内置 `core/scoring.py` 函数对照评分；覆盖缺题、非法分值、无效类型、平局、分页及题库/评分版本变化。
- 验证网页与 CLI 的开始、暂停、继续、退出、重新开始、结果查询，以及答题中穿插知识问题。
- 验证页面刷新、宿主重启、超过模型历史窗口后的恢复；不同会话互不影响，清空/删除同步清除进度。
- 验证重复点击、旧题目卡、切换会话、请求超时与断连；存储失败不推进已保存状态。
- 验证 MCP 超时、进程退出与宿主关闭的资源清理；API/CLI 入口都能正常管理运行时。
- 回归原知识检索、Skill、DeepSeek 回答与引用校验；确认 MBTI 结果没有伪造教材来源引用，执行计数与实际调用一致。
