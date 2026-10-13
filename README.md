# Person_A：文档知识库问答 Agent

本项目使用本地 `docs/` 中的 Markdown 教材，通过 Ollama 生成向量，结合向量与关键词检索，再由 DeepSeek 回答 Python / LangChain 学习问题并给出来源引用。另通过本地 MCP 服务提供 MBTI 问卷与评分，支持在同一会话中答题、暂停、恢复和查看结果。MBTI 核心代码源自 MindCare，已复制并适配到本项目，可独立运行。

浏览器 MCP 工具支持“我想在 bili 学习数据结构”这类请求：模型提取搜索主题，程序向后端电脑的默认浏览器发出 B 站搜索页打开请求，并返回请求状态与链接。工具不读取页面或视频，也不确认搜索结果。

文档入库、检索、模型适配、Agent、意图识别、MCP 接入、MBTI 答题、命令行、HTTP API 和本地网页已有实现。**本次只开发，没有运行测试、语法检查、服务或模型联调，也没有安装依赖；新增接入仍待统一验收，不能据此认为已验证可用。**

本次进一步细化咨询与操作的意图边界，并加入本轮最多 3 步、共享调用预算的任务规划。执行结果随回答保存，页面可展开“任务执行步骤”查看完成和跳过情况；简单问题继续使用原流程。实现边界、示例与待验收事项见 [意图边界与多步骤任务规划](project_docs/INTENT_PLANNING.md)。**这两项同样只完成开发，尚未测试。**

## 快速开始

在项目根目录打开终端，以可编辑方式安装到准备使用的 Python 环境：

```powershell
python -m pip install -e .
```

当前配置根据源码目录定位项目根目录，先采用以上安装方式。之后编辑项目根目录的 `.env`；如果文件不存在，可手动从 `.env.example` 创建副本，不要覆盖已有配置或提交真实密钥。

至少确认以下配置：

| 配置 | 用途 |
| --- | --- |
| `KNOWLEDGE_DOCS_DIR` / `KNOWLEDGE_INDEX_DIR` | 文档与索引路径，默认 `./docs` 和 `./data/index`，相对项目根目录解析 |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | 默认 Ollama 与 `qwen3-embedding:0.6b`；本地需已准备好配置的模型 |
| `EMBEDDING_BASE_URL` | Ollama 服务地址，默认 `http://localhost:11434` |
| `CHAT_PROVIDER` / `CHAT_MODEL` | 使用 `deepseek`，模型名填写账号实际可用的 API 模型 |
| `CHAT_API_KEY` / `CHAT_BASE_URL` | 本地填写 DeepSeek 密钥，服务地址默认 `https://api.deepseek.com` |
| `SESSION_DB_PATH` / `SESSION_DB_LOCK_TIMEOUT` | 本地会话数据库路径（默认 `./data/sessions/chat.db`）与 SQLite 锁等待秒数 |
| `MCP_CONFIG_PATH` | MCP 服务配置路径；未设置该键时使用项目 `config/mcp_servers.json`，显式设为空字符串则关闭 MCP |

启动 Ollama 服务，并确认向量模型可用。超时、重试次数、生成参数和检索条数可参考 `.env.example` 中的其他字段调整。

首次使用先建库，再问答：

```powershell
person-a build
person-a ask "向量检索和关键词检索有什么区别？"
person-a chat
```

文档新增、修改或删除后同步索引：

```powershell
person-a update
```

也可使用模块入口，例如 `python -m knowledge_agent.cli ask "什么是 RAG？"`。运行知识问答需要可用的已发布索引；CLI 不会因提问自动完成首次建库。MBTI 多轮答题使用 `chat` 或网页会话；安装与配置见下文“MindCare MBTI 与 MCP”。

## 命令与会话

| 命令 | 作用 |
| --- | --- |
| `person-a build` | 完整构建并发布索引 |
| `person-a update` | 同步文档变化；已有入库逻辑判断必要时重建 |
| `person-a ask "问题"` | 执行一次问答 |
| `person-a chat` | 新建一个随机 ID 的本地持久化会话 |
| `person-a chat --session-id ID` | 按 ID 恢复已有会话 |
| `person-a sessions` | 列出可恢复的本地会话 |
| `person-a history ID` | 查看会话的完整持久化历史 |
| `person-a delete-session ID` | 永久删除会话及其历史 |

使用 `person-a --help` 或相应子命令的 `--help` 查看参数。`build`、`update`、`ask` 支持 `--json`：结构化业务结果（包括返回的失败状态）写入标准输出，进度和抛出的执行错误写入标准错误。脚本应同时检查结果状态和退出码。

在 `chat` 中可输入：

- `/help`：显示会话命令。
- `/clear`：永久清空当前会话的全部历史，但保留会话本身。
- `/refresh`：重新加载已经发布的索引，不执行建库或文档同步；文档变化后应先运行 `update`。
- `/exit`：保存已经完成的问答并退出；输入结束（EOF）也会保留已完成的问答。

会话记录保存在本地 SQLite 数据库中，默认位置为 `data/sessions/chat.db`。不带 `--session-id` 运行 `chat` 会新建会话；需要先用 `person-a sessions` 找到 ID，再用 `person-a chat --session-id ID` 恢复。`/exit`、EOF 或 `Ctrl+C` 只结束当前运行，不删除已提交的历史；`/clear` 会永久删除当前会话的全部历史；`delete-session` 则连同会话本身一起删除。按 `Ctrl+C` 中断时退出码为 `130`。

数据库保留完整的已提交历史，`history` 命令可分页查看；恢复会话时只有最近 `AgentConfig.max_history_turns` 轮会装入模型上下文，以控制请求大小，较早记录仍保留在数据库中。该存储仅面向当前设备上的本地使用，尚不提供多用户账户隔离或跨设备同步。运行时数据库及 SQLite 辅助文件已被 Git 忽略，请勿提交聊天记录。默认日志不输出消息正文，只应记录必要的会话操作和错误类别。

退出码约定：`0` 表示命令正常结束，包含资料不足、需要澄清或问题超出范围的正常业务结果；`1` 表示执行或服务错误，或达到调用上限；`2` 表示参数错误；`130` 表示用户中断。

交互式 `chat` 中的单轮失败会显示错误并允许继续提问，正常输入 `/exit` 或 EOF 时仍返回 `0`；会话初始化失败返回 `1`。自动化脚本若需按一次问答的状态判断成败，请使用 `ask`。

## 本地网页与 HTTP API

先按“快速开始”配置模型并执行 `person-a build`，再安装 API 额外依赖并从项目根目录启动本机单进程服务：

```powershell
python -m pip install -e ".[api,mcp]"
python -m uvicorn knowledge_agent.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

在浏览器打开 [本地问答页面](http://127.0.0.1:8000/)，新建会话后输入问题并发送。左侧可以选择已有会话、继续追问，或在确认后清空历史、删除会话。回答完成后可查看引用中的文档名称、小节和行号；历史记录保存在本地 SQLite 中，刷新页面或重启服务后可以重新读取。

如果只使用知识问答，可安装 `.[api]`，并将 `MCP_CONFIG_PATH` 显式设为空字符串关闭 MCP。

页面使用原生 HTML、CSS 和 JavaScript，无需安装 Node.js 或运行 npm。页面和 API 同源，由上面的单 worker 服务一起提供；不要直接双击 `web/index.html`，也无需另开前端开发服务器。服务器只公开首页和 `web/assets/` 中的页面资源，不公开项目配置、原始文档或数据库文件。

回答正文在引用校验后一次性返回，等待时页面实时显示意图识别、技能选择、检索、工具和模型等执行阶段。遇到请求中断或超时，应先重新加载会话历史确认结果是否已经保存，再决定是否重发，避免重复问答。

回答方式可选择“普通问答”“自动选择 Skill”或三个手动技能。自动模式只在知识路线启用，每个计划知识子任务独立选择。默认算法 `SKILL_SELECTION_MODE=rules` 不增加模型调用；设为 `hybrid` 时，仅规则不能确定且预算足够时尝试一次模型分类，并为回答保留一次调用。选择失败或无匹配时回退普通问答。选择模型只接收技能元数据，回答模型才加载本步选中技能的完整指令。

CLI 同样支持 `person-a ask "什么是 RAG？" --skill-mode auto`、`person-a chat --skill-mode auto` 和 `person-a ask "比较 Chain 和 Agent" --skill concept-comparison`。不传新参数保持普通模式；`manual` 必须配合 `--skill`，`auto/off` 不能配合技能 ID。

### 执行详情

回答下方的“执行详情”默认折叠，点击或通过键盘展开即可查看该轮的执行摘要。完整详情随回答返回，历史回答使用当时保存的记录；等待期间另显示最近八个实际阶段的实时状态，不展示模型内部推理。

摘要记录实际经过的意图识别、路由、问题分类、初始检索、Skill 准备、模型调用、工具调用和回答收尾等阶段，展示执行顺序、发起方、状态、可用的结果数量与耗时，以及工具名称、索引版本和错误类别。回答自带的调用次数和停止原因也会一并显示；提前结束的问答不会补造未执行的阶段，事件被截断或耗时缺失时会按实际情况提示。MBTI 分支的 MCP 工具由程序调用，执行记录不意味着一定调用了模型。

各项数据的含义如下：

| 项目 | 记录口径 |
| --- | --- |
| 模型调用次数 `model_calls` | Agent 发起的逻辑调用次数。一次逻辑调用可能包含适配层内部重试，重试不单独增加次数；该次模型事件的耗时包含内部重试等待，不代表实际 HTTP 请求次数。 |
| 工具调用次数 `tool_calls` | 包含程序主动执行的首次知识检索和后续实际执行的工具调用。被拒绝的工具请求也可能有事件记录，但没有执行就不增加次数，因此不能用事件条数代替工具调用次数。 |
| 总耗时 | 从进入本轮 Agent 问答到生成执行摘要，包含流程处理与调用等待；不含服务排队、会话提交和 HTTP 传输耗时。阶段耗时是其中各执行边界的计时，不要求相加后等于总耗时；缺失值不表示 `0 ms`。 |
| Skill 状态 | “手动指定”来自请求；`skill_select` 记录实际选中技能、模式、策略与原因；“准备”表示技能说明与工具配置已准备好；“进入模型调用”表示携带技能说明开始调用模型。多步骤任务按 `step_id` 展示各步结果，顶层最近一次准备/应用的值不代表全部步骤。 |

执行详情用于了解系统执行了哪些步骤，不是答案质量评分，也不能证明模型严格遵循了 Skill。评价回答是否正确、引用是否充分，仍需单独检查答案和资料。

新问答的摘要与该轮结果一起保存在 SQLite 中，无需另建数据库。旧历史没有 `trace` 时，会显示“历史记录未保存详细过程”，并保留已有的调用统计；不会用当前技能选择补写历史，也不会把缺少记录解释为没有执行。`service_error`、`limit_reached` 等失败轮仍沿用原有规则，不保存到会话历史；若错误响应带有结果，页面可暂时展示其详情并标明未保存，重新加载历史后无法恢复。会话写入失败的结果同样不能视为已保存。展开详情不会重新发起问答，断连后的恢复方式仍是先查询历史。

### HTTP API 使用

服务启动后也可访问 [交互式接口文档](http://127.0.0.1:8000/docs)。以下 PowerShell 示例依次创建会话、提问、读取完整问答历史，最后删除会话；服务生成的 ID 保存在变量中，示例问题不含真实聊天内容或密钥：

```powershell
$baseUrl = "http://127.0.0.1:8000"
$session = Invoke-RestMethod -Method Post -Uri "$baseUrl/sessions" `
  -ContentType "application/json" -Body '{"title":"演示知识库问答"}'
$sessionId = $session.session_id

$answer = Invoke-RestMethod -Method Post -Uri "$baseUrl/sessions/$sessionId/messages" `
  -ContentType "application/json" -Body '{"question":"RAG 的基本流程是什么？"}'

$history = Invoke-RestMethod -Method Get `
  -Uri "$baseUrl/sessions/$sessionId/messages?limit=50&offset=0"

Invoke-RestMethod -Method Delete -Uri "$baseUrl/sessions/$sessionId"
```

历史接口以“完整问答轮”为分页单位，按时间正序返回；`limit` 为 1—100，`offset` 是跳过的问答轮数。原 `POST /sessions/{session_id}/messages` 保持完整 JSON 返回。可传 `{"question":"什么是 RAG？","skill_mode":"auto"}` 开启自动技能；省略模式且提供 `skill_id` 仍按旧手动模式处理。

页面使用新增的 `POST /sessions/{session_id}/messages/stream`，请求体与原接口相同，通过 `fetch` 读取 SSE：`progress` 为公开阶段，`result` 为完整 AnswerResponse，`error` 为原错误结构加 `http_status`，可能含未保存的回答。连接开始后的业务错误通过事件传递，不能只检查 HTTP 200。普通 JSON 接口仍同时检查 HTTP 状态和业务 `status`。正文不逐字输出；最终事件在既有回答校验和持久化流程结束后发送。不新增请求 ID、重连或自动重试。

服务用单个工作线程串行处理 Agent 和 SQLite 操作。已有任务占用服务或同一会话正在问答时，其他修改请求可能收到 `409` 或 `503`；模型超时返回 `504`。客户端断开或停止等待并不保证后台任务被取消，稍后应查询历史确认该轮是否已经保存，避免自动重试造成重复问答。按 `Ctrl+C` 停止服务会等待在途任务清理并关闭连接，但不会删除已经提交的会话记录；重启后可以使用原会话 ID 继续访问。

若只想清空历史并保留会话，可调用 `DELETE /sessions/{session_id}/messages`；删除会话则调用 `DELETE /sessions/{session_id}`。两者成功时均返回 `204`，且没有响应正文。

## MindCare MBTI 与 MCP

MBTI 已接入 API 和 CLI 的 MCP 生命周期。宿主启动本地 stdio 子进程，子进程导入本项目 `src/mbti_mcp/core/` 内置的题库、确定性评分与类型描述。这四个 Python 文件源自 MindCare 的 `backend/mbti`，包内导入已适配为相对导入；运行 Person_A 不再需要原 MindCare 目录，也无需启动它的 Web 服务或数据库。

使用完整网页功能前，在 Person_A 使用的同一个 Python 环境中自行安装依赖：

```powershell
python -m pip install -e ".[api,mcp]"
```

项目默认配置 `config/mcp_servers.json` 启用 `mbti`。配置的 `command: "$PYTHON"` 由宿主替换为当前 Person_A 的 Python 解释器，参数只指定项目内的 `examples/mbti_mcp_server.py` 启动文件，不含外部项目路径或 `--mindcare-root` 参数。移动整个 Person_A 目录时，无需另行定位 MindCare，也不需要手动启动 MCP 服务。

未在环境变量或 `.env` 中设置 `MCP_CONFIG_PATH` 时，使用上述默认配置；如果旧 `.env` 已有空值 `MCP_CONFIG_PATH=`，MCP 会保持关闭，需要自行改为 `MCP_CONFIG_PATH=./config/mcp_servers.json`。开发过程没有覆盖实际 `.env`。`config/mcp_servers.mbti.example.json` 是默认禁用的独立模板，区别于已启用的运行配置。

在网页新建会话后输入“开始 MBTI 测试”，按题目点击 1—5 选项；CLI 的 `person-a chat` 可以直接回复数字 1—5。支持“暂停 MBTI 测试”“继续 MBTI 测试”“退出 MBTI 测试”“查看 MBTI 结果”“重新开始 MBTI 测试”。完成 48 题后展示四维得分与类型，也可提问“介绍 INTJ 的特点”。历史题目卡不可重复提交。

答题进度与问答轮一起保存在现有 SQLite 的 `turns.result_json.assessment_state`，没有新建数据库或答题表；恢复会话时单独读取最新状态，不受模型历史窗口长度限制。清空或删除会话也会删除其中的答题记录。普通 MBTI 答题、评分与类型描述由程序调用 MCP，不消耗 DeepSeek 模型次数；知识问答仍走原检索与 DeepSeek 流程。

这是源自 MindCare 项目、现内置于 Person_A 的人格倾向自评问卷。得分比例表示两极答题分值的占比，不能解释为置信度；平局保留原项目的默认规则并明确显示，不把默认字母当成明确倾向。完整配置、工具契约与待验收事项见 [MBTI_MCP.md](project_docs/MBTI_MCP.md)。

## B 站搜索与本机浏览器

在网页会话、`person-a ask` 或 `person-a chat` 中提出“我想在 bili 学习数据结构”或“帮我在 B 站搜索 Python 装饰器教程”，Agent 可路由到 `mcp__browser__search_bilibili`。意图规则只选择路线，模型根据本次问题生成 `{"query":"数据结构"}` 形式的参数，宿主和 MCP 服务分别校验后调用浏览器。

默认运行配置 `config/mcp_servers.json` 已在保留 MBTI 的基础上启用 `browser`，白名单只有 `search_bilibili`；独立模板 `config/mcp_servers.browser.example.json` 默认禁用。本功能复用现有 `.[api,mcp]` 依赖与 Python 标准库，不增加新依赖。若 `MCP_CONFIG_PATH` 显式为空，浏览器工具和其他 MCP 服务都会保持关闭；仅关闭浏览器工具可将 `browser.enabled` 设为 `false`，配置调整后重启原服务。

工具只能打开固定的 B 站搜索地址，参数不接受其他 URL、命令或额外字段。宿主和服务端保证仅合法参数可执行，不承诺模型首次生成的 JSON 一定正确。浏览器分支关闭模型传输重试，至多发送两次模型请求：首次生成参数，必要时再修正一次；实际调用发生后本轮不重复打开。**最终状态和链接由程序根据工具返回结果生成，不再交给模型改写。** 普通知识问答不打开浏览器；手动知识 Skill 与 MCP 操作冲突时沿用现有提示。

浏览器位于运行后端的电脑，远程网页不会直接控制访问者的浏览器。返回“已发出打开请求”不表示网页已加载或已找到视频；工具不读取搜索结果。本次没有启动浏览器或运行联调，完整边界及待验收项见 [BROWSER_MCP.md](project_docs/BROWSER_MCP.md)。

## 项目结构

```text
Person_A/
├── docs/                         # 本地 Markdown 知识文档
├── project_docs/                 # 项目设计、实现记录与待验收清单
│   ├── README.md                # 开发文档导航及本次迁移说明
│   ├── INTENT_PLANNING.md       # 意图边界与多步骤任务规划
│   ├── INTENT_MCP_TODO.md       # 意图与 MCP 初始开发计划存档
│   ├── AUTO_SKILL_TODO.md       # 自动 Skill 实现记录与待验收项
│   ├── MBTI_MCP.md              # 内置 MBTI 接入说明
│   └── BROWSER_MCP.md           # B 站搜索与浏览器 MCP 接入说明
├── src/knowledge_agent/
│   ├── __init__.py
│   ├── cli.py                    # 建库、更新、单次与多轮问答入口
│   ├── api.py                    # HTTP 兼容启动入口
│   ├── application.py            # 应用生命周期、持久化会话与资源管理
│   ├── config.py                 # 配置读取与校验
│   ├── models.py                 # 对话与 Embedding 模型适配
│   ├── schemas.py                # 片段、检索结果与回答的数据结构
│   ├── session_store.py          # 本地 SQLite 会话持久化
│   ├── skill_loader.py           # 项目 Skill 的发现、加载与工具约束
│   ├── evaluation.py             # 效果评估 TODO
│   ├── agent/                    # Agent 执行流程、提示词与执行记录
│   │   ├── __init__.py
│   │   ├── runner.py             # KnowledgeAgent 单轮与多步编排入口
│   │   ├── state.py              # 单轮状态与准备结果结构
│   │   ├── preparation.py        # 技能选择、检索与模型上下文准备
│   │   ├── model_loop.py         # 模型 / 工具循环及预算控制
│   │   ├── browser_actions.py    # B 站搜索意图与操作边界
│   │   ├── browser_execution.py  # 浏览器单动作执行、有限修正与结果生成
│   │   ├── answers.py            # 回答整理、引用校验与结果摘要
│   │   ├── intent.py             # 意图识别节点
│   │   ├── router.py             # 知识 / MCP / 直接回复路由
│   │   ├── skill_selector.py     # 自动技能选择及本步快照
│   │   ├── planner.py            # 有限步骤规划
│   │   ├── plan_executor.py      # 依赖调度和共享预算
│   │   ├── assessment.py         # 会话内 MBTI 答题状态与工具编排
│   │   ├── prompts.py            # 问答约束与来源标记规则
│   │   └── trace.py              # 单轮执行事件、耗时与公开摘要
│   ├── http/                     # HTTP 传输层，业务继续复用 Agent
│   │   ├── app.py                # FastAPI 创建、生命周期与路由装配
│   │   ├── schemas.py            # 请求与公开响应模型
│   │   ├── serialization.py      # 领域结果转公开响应，统一来源及摘要字段
│   │   ├── errors.py             # HTTP 错误映射与公开错误结构
│   │   ├── runtime.py            # 单工作线程与 SQLite 线程归属
│   │   ├── streaming.py          # 有界进度通道与 SSE 输出
│   │   └── routes/
│   │       ├── chat.py           # 提问、完整回答与实时阶段进度
│   │       ├── sessions.py       # 会话 CRUD 与历史
│   │       ├── knowledge.py      # 文档与索引管理
│   │       └── system.py         # 健康检查、技能列表与页面入口
│   ├── knowledge/                # 文档、索引、检索与知识库管理
│   │   ├── __init__.py
│   │   ├── ingest.py             # 文档读取、分块、索引构建与同步
│   │   ├── retrieval.py          # 向量与关键词检索、上下文补查
│   │   ├── skill_retrieval.py    # 技能检索策略和实际证据覆盖
│   │   ├── code_analysis.py      # 选择器与检索共用的可见代码识别
│   │   ├── keywords.py           # 共享关键词分词规则
│   │   └── manager.py            # 文档上传、预览与索引管理
│   ├── tools/                    # 本地工具与统一注册、执行校验
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   └── knowledge.py
│   ├── mcp/                      # 外部工具连接与注册器适配
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── bootstrap.py          # API / CLI 共享的 MCP 启动与关闭
│   │   └── adapter.py
│   └── images/                   # 图片提问扩展 TODO，暂缓开发
│       ├── __init__.py
│       ├── ocr.py
│       ├── attachments.py
│       └── image_question.py
├── src/mbti_mcp/                 # 内置 MBTI 核心与独立 MCP 服务
│   ├── __init__.py
│   ├── service.py                # 题库分页、严格校验、评分与描述
│   ├── server.py                 # stdio MCP 工具入口
│   └── core/                     # 从 MindCare 复制并适配的本地核心
│       ├── __init__.py
│       ├── questions.py          # 48 题题库与作答选项
│       ├── scoring.py            # 确定性计分与类型判定
│       └── descriptions.py       # 类型名称与说明
├── src/browser_mcp/              # 固定 B 站搜索链接与本机浏览器 MCP 工具
├── skills/                       # 概念讲解、概念比较、代码解读的 Skill
├── config/
│   ├── mcp_servers.json          # 运行配置，启用内置 MBTI 与 browser
│   ├── mcp_servers.mbti.example.json # MBTI 模板，默认禁用
│   ├── mcp_servers.browser.example.json # 浏览器工具模板，默认禁用
│   └── mcp_servers.example.json  # 演示服务模板，默认禁用
├── examples/
│   ├── mbti_mcp_server.py        # 确保源码包可导入的 MBTI 启动入口
│   ├── browser_mcp_server.py     # 本地浏览器 MCP 启动入口
│   └── mcp_demo_server.py        # 无副作用演示 MCP 服务 TODO
├── web/
│   ├── index.html                # 本地问答页面
│   └── assets/                   # 原生 JavaScript 与 CSS
├── data/index/                   # 建库后发布的索引与清单
├── data/sessions/                # 本地会话数据库（不提交）
├── evaluation/questions.jsonl    # 5 条人工验收问题草案
├── tests/                        # 按验证层次整理；本次未执行
│   ├── unit/agent/              # Agent 与执行记录验证
│   ├── unit/storage/            # 会话存储验证
│   └── integration/http/        # 静态资源及基础路由验证
├── .env                          # 本地配置，不提交真实凭据
├── .env.example                  # 配置模板
├── pyproject.toml                # 依赖声明与命令入口
└── README.md
```

Python 模块按功能分包：`agent/` 组织问答与答题执行，`http/` 处理 HTTP 请求与实时阶段传输，`knowledge/` 处理知识库，`tools/` 统一注册和校验工具，`mcp/` 管理外部工具接入，`images/` 预留图片提问。公共配置、数据结构、模型适配、会话存储和 Skill 加载仍放在包根目录；`application.py` 负责应用及会话生命周期。

本次目录整理把大文件按职责拆开，并将开发说明集中到 `project_docs/`。知识库教材仍在 `docs/`，不会把项目设计说明加入默认检索材料；索引、会话数据库、技能目录和运行配置路径保持不变。`cli.py`、`api.py` 保留原入口，`person-a`、`python -m knowledge_agent.cli` 和 `uvicorn knowledge_agent.api:create_app` 的命令不变。迁移对应关系见 [开发文档导航](project_docs/README.md)。**目录调整只完成开发，未运行测试、导入检查、编译或服务联调。**

意图识别、路由、MCP 和 MBTI 已有实现，运行结果仍待统一验证；图片提问与效果评估继续保留 TODO。

索引以本地文件保存，无需另行安装数据库服务。实际索引文件路径以 `manifest.json` 中的记录为准，不要手动修改片段、向量或清单来绕过建库流程。

## 数据与问答规则

建库流程：读取 `docs/` → 保留文档结构并分块 → 生成 Embedding → 发布片段、向量、关键词索引与文档清单。请求先结合会话状态识别意图并分流；知识问答执行混合检索 → 必要时补查章节 → 基于证据回答 → 校验并生成来源引用，MBTI 则进入独立答题与 MCP 工具流程。

- **对话与向量模型分开配置。** 入库和查询使用兼容的 Embedding 配置；更换向量模型或向量维度后需要重建索引。
- **保留 Markdown 结构。** 片段保存标题层级；代码、概念图和 Prompt 与对应说明保持关联。文档中的 Prompt 是知识内容，不是 Agent 的运行指令。
- **引用由程序生成。** 引用协议要求模型只输出本轮有效的来源标记，如 `[S1]`。程序校验标记，再使用检索记录生成完整文件名、章节和原文起止行号，模型不能自行编造来源元数据。同为第 26 章的两份文档通过完整文件名区分。
- **同步新增、修改和删除。** 文档清单记录内容与配置版本；更新时同步片段、向量和关键词记录，检索使用同一版索引，避免引用已删除或过期内容。
- **答案受证据范围约束。** 资料不足时说明缺失信息，只回答有依据的部分。教材中的业务示例不代表任何公司的正式政策；不同文档存在差异时展示出处和差异，不自行确定哪个说法权威。
- **MCP 结果有独立来源。** MBTI 问卷与评分来自本项目内置的 MCP 工具，核心代码源自 MindCare，不生成本地教材的 `[S1]` 引用；工具只接受实际用户作答，不用模型推测缺失答案。

## 待统一验收

自动 Skill 选择、技能检索策略边界修正、API/CLI/页面接入及 SSE 阶段进度已完成代码开发，尚未运行测试、语法检查、模型或服务联调。开发说明与未执行的验收清单见 [AUTO_SKILL_TODO.md](project_docs/AUTO_SKILL_TODO.md)。效果评估、请求标识与断线恢复继续暂缓。

`src/knowledge_agent/agent/prompts.py` 已与 Agent 统一为 `[S1]` 来源标记协议；最终文档名称、小节和行号由程序校验并生成，引用正确性仍需在真实问答中核对。

`evaluation/questions.jsonl` 保留 5 条人工验收草案，覆盖概念比较、检索与生成质量、资料不足、跨章节检索及无答案场景。每行记录问题、期望来源、回答要点和验收标准，目前没有执行结果或通过率。

开发完成后统一检查：首次建库与文档更新是否正确；检索是否命中对应章节；回答是否有原文支持；引用能否定位实际片段；多轮追问、清空会话和索引刷新是否符合预期；模型服务异常、参数错误与中断时是否正确退出。网页的会话切换、历史分页、清空与删除确认、请求异常提示也需纳入完整端到端联调。假运行时下的页面和路由检查不能代替真实模型与索引的验收。

新增 MBTI 接入还需统一验证：三工具发现与调用、完整 48 题评分、分页和版本不一致处理、暂停恢复与重启续答、会话隔离、旧题目卡重复点击、存储失败及 MCP 退出清理。具体清单见 [MBTI_MCP.md](project_docs/MBTI_MCP.md#待统一验收)。以上均为待执行项目，没有本次测试通过结论。

B 站搜索 MCP 的自然语言路由、参数双端校验、一次修正与防重复打开、真实浏览器唤起及旧流程兼容也尚未验证，清单见 [BROWSER_MCP.md](project_docs/BROWSER_MCP.md#待统一验收)。没有运行测试、导入检查、编译、安装或服务启动。
