# Person_A：文档知识库问答 Agent

本项目使用本地 `docs/` 中的 Markdown 教材，通过 Ollama 生成向量，结合向量与关键词检索，再由 DeepSeek 回答 Python / LangChain 学习问题并给出来源引用。另通过本地 MCP 服务提供 MBTI 问卷与评分，支持在同一会话中答题、暂停、恢复和查看结果。MBTI 核心代码源自 MindCare，已复制并适配到本项目，可独立运行。

浏览器 MCP 工具支持“我想在 bili 学习数据结构”这类请求：模型提取搜索主题，程序向后端电脑的默认浏览器发出 B 站搜索页打开请求，并返回请求状态与链接。工具不读取页面或视频，也不确认搜索结果。

文档入库、检索、模型适配、Agent、意图识别、MCP 接入、MBTI 答题、命令行、HTTP API 和本地网页已有实现。会话业务意图一期已有离线测试记录；2026-10-18 的意图合并及 B 站补参修复仅完成开发和文本审阅，未运行测试或联调。当前结构与待验收用例见 [会话意图说明](project_docs/architecture/DIALOGUE_INTENT_TODO.md)。

本次进一步细化咨询与操作的意图边界，并加入本轮最多 3 步、共享调用预算的任务规划。执行结果随回答保存，页面可展开“任务执行步骤”查看完成和跳过情况；简单问题继续使用原流程。实现边界、示例与待验收事项见 [意图边界与多步骤任务规划](project_docs/architecture/INTENT_PLANNING.md)。**这两项同样只完成开发，尚未测试。**

## 快速开始

新增 B 站未知表达的语义补充、自然语言多任务组合和长对话摘要。例如“帮我在B站找点数据结构的视频”“先解释Python装饰器，再在B站搜索Python装饰器”。较早知识对话可压缩为 SQLite 摘要，在页面“早期对话摘要”中查看；完整历史保留。默认启用，无新增依赖，配置与使用用例见 [三项功能说明](project_docs/features/SEMANTIC_PLANS_MEMORY.md)。**本次只完成开发与文本审阅，未运行测试、服务、数据库迁移或模型联调。**

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

数据库保留完整的已提交历史，`history` 命令可分页查看；恢复会话时加载最近 `AgentConfig.max_history_turns` 轮原始历史，以及已保存的早期知识对话摘要，较早完整记录仍保留在数据库中。该存储仅面向当前设备上的本地使用，尚不提供多用户账户隔离或跨设备同步。运行时数据库及 SQLite 辅助文件已被 Git 忽略，请勿提交聊天记录。默认日志不输出消息正文，只应记录必要的会话操作和错误类别。

会话同时保存一个当前业务焦点和至多一个待补参数任务。闲聊不会清除待补任务；明确的新任务会替代它；“继续、取消、下一题、5”等短输入只按当前业务焦点解释。数据库 schema 为 v3，启动时会从 v1/v2 原地迁移并保留已有会话与问答。普通回答与下一业务状态原子提交；B 站待补任务会在外部派发前先持久消费，结果未知时不会因“继续”自动重放。取消只停止后续业务交互，不撤销已完成的浏览器操作，也不具备中断正在执行请求的能力。

退出码约定：`0` 表示命令正常结束，包含资料不足、需要澄清或问题超出范围的正常业务结果；`1` 表示执行或服务错误，或达到调用上限；`2` 表示参数错误；`130` 表示用户中断。

交互式 `chat` 中的单轮失败会显示错误并允许继续提问，正常输入 `/exit` 或 EOF 时仍返回 `0`；会话初始化失败返回 `1`。自动化脚本若需按一次问答的状态判断成败，请使用 `ask`。

## 本地网页与 HTTP API

先按“快速开始”配置模型并执行 `person-a build`，再安装 API 额外依赖并从项目根目录启动本机单进程服务：

```powershell
python -m pip install -e ".[api,mcp]"
python -m uvicorn knowledge_agent.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

安装只需在准备运行项目的 Python 环境中执行一次；以后修改现有 Python 源码无需重新安装。增加依赖、修改入口或迁移项目位置后，需要重新执行安装。`-e` 使用可编辑安装，把当前 `src/` 下的包接入该环境的模块搜索路径，因此无需每次设置 `$env:PYTHONPATH="src"`。

如果 `uvicorn` 和安装时的 `python` 属于同一个环境，之后可直接运行：

```powershell
uvicorn knowledge_agent.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

使用虚拟环境时，每次打开新终端仍需先激活该环境，或让 IDE 自动激活。可通过 `Get-Command python, uvicorn | Select-Object Name, Source` 查看两个命令的位置；如果来自不同环境，先切换到日常运行项目的环境，再执行安装。上面的 `python -m uvicorn` 写法可明确使用当前 `python` 所属环境，避免误用 PATH 中其他环境或已失效的 `uvicorn.exe`。

`PYTHONUTF8=1` 是 Python 的 UTF-8 模式开关，解决默认文本编码兼容问题，不负责查找模块。本项目读取配置、文档和索引时已明确指定 UTF-8，目前没有发现必须设置该变量才能启动 API 的代码路径；若省略后出现编码异常，应根据具体异常定位。PowerShell 的 `$env:...` 赋值只作用于当前终端进程及之后启动的子进程，不会自动保存到新终端；把这两个变量写进项目 `.env` 也不能替代解释器启动前的环境设置。

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

新问答的摘要与该轮结果一起保存在 SQLite 中，无需另建数据库。旧历史没有 `trace` 时，会显示“历史记录未保存详细过程”，并保留已有的调用统计；不会用当前技能选择补写历史，也不会把缺少记录解释为没有执行。普通 `service_error`、`limit_reached` 失败轮沿用原有规则，不保存到会话历史；已经进入外部派发保护的 B 站任务例外，其已确认或结果未知状态会独立收尾并可随对应提示保存，防止恢复后自动重放。若错误响应带有未保存结果，页面只暂时展示并标明未保存。会话写入失败的结果不能视为已保存。展开详情不会重新发起问答，断连后的恢复方式仍是先查询历史。

### HTTP API 使用

服务启动后也可访问 [交互式接口文档](http://127.0.0.1:8000/docs)。以下 PowerShell 示例依次创建会话、提问、读取完整问答历史，最后删除会话；服务生成的 ID 保存在变量中，示例问题不含真实聊天内容或密钥：

回答新增可选的 `dialogue` 与 `business_state` 公开字段；旧记录缺少字段时仍可读取。`GET /sessions/{session_id}/business-state` 返回最新已提交的安全投影，不返回原始授权输入、槽值来源或数据库 JSON。

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

项目默认配置 `config/mcp_servers.json` 启用 `mbti`。配置的 `command: "$PYTHON"` 由宿主替换为当前 Person_A 的 Python 解释器，参数只指定项目内的 `scripts/mbti_mcp_server.py` 启动文件，不含外部项目路径或 `--mindcare-root` 参数。移动整个 Person_A 目录时，无需另行定位 MindCare，也不需要手动启动 MCP 服务。

未在环境变量或 `.env` 中设置 `MCP_CONFIG_PATH` 时，使用上述默认配置；如果旧 `.env` 已有空值 `MCP_CONFIG_PATH=`，MCP 会保持关闭，需要自行改为 `MCP_CONFIG_PATH=./config/mcp_servers.json`。开发过程没有覆盖实际 `.env`。`config/mcp_servers.mbti.example.json` 是默认禁用的独立模板，区别于已启用的运行配置。

在网页新建会话后输入“开始 MBTI 测试”，按题目点击 1—5 选项；CLI 的 `person-a chat` 可以直接回复数字 1—5。支持“暂停 MBTI 测试”“继续 MBTI 测试”“退出 MBTI 测试”“查看 MBTI 结果”“重新开始 MBTI 测试”。完成 48 题后展示四维得分与类型，也可提问“介绍 INTJ 的特点”。历史题目卡不可重复提交。

答题进度与问答轮一起保存在现有 SQLite 的 `turns.result_json.assessment_state`，没有新建数据库或答题表；恢复会话时单独读取最新状态，不受模型历史窗口长度限制。清空或删除会话也会删除其中的答题记录。普通 MBTI 答题、评分与类型描述由程序调用 MCP，不消耗 DeepSeek 模型次数；知识问答仍走原检索与 DeepSeek 流程。

这是源自 MindCare 项目、现内置于 Person_A 的人格倾向自评问卷。得分比例表示两极答题分值的占比，不能解释为置信度；平局保留原项目的默认规则并明确显示，不把默认字母当成明确倾向。完整配置、工具契约与待验收事项见 [MBTI_MCP.md](project_docs/features/MBTI_MCP.md)。

## B 站搜索与本机浏览器

在网页会话、`person-a ask` 或 `person-a chat` 中提出“我想在 bili 学习数据结构”“帮我在 B 站搜索 Python 装饰器教程”或“我想在 B站看火影忍者”，Agent 可路由到 `mcp__browser__search_bilibili`。有会话状态时，程序把已确认主题和来源绑定交给独立搜索执行器；无会话的单次请求和计划子步骤沿用模型原生工具调用，并校验主题一致。两种路径都由宿主和 MCP 服务校验参数。观看表达也只打开搜索结果页，不自动选集或播放。

也支持跨轮补全：先发送“帮我在B站找点视频”，系统保存缺主题待办并询问；同一会话再发“火影忍者”，先补参并进入搜索，不再把短回复送去普通意图模型或知识检索。期间可以闲聊。主题绑定保留原请求轮次和当前补充输入的来源；任务取消、被新任务替代、跨会话、已消费或修订过期时均拒绝派发。升级后需重启后端；旧错误轮次如果没有保存待办，需要重新发起第一句请求。

默认运行配置 `config/mcp_servers.json` 已在保留 MBTI 的基础上启用 `browser`，白名单只有 `search_bilibili`；独立模板 `config/mcp_servers.browser.example.json` 默认禁用。本功能复用现有 `.[api,mcp]` 依赖与 Python 标准库，不增加新依赖。若 `MCP_CONFIG_PATH` 显式为空，浏览器工具和其他 MCP 服务都会保持关闭；仅关闭浏览器工具可将 `browser.enabled` 设为 `false`，配置调整后重启原服务。

工具只能打开固定的 B 站搜索地址，参数不接受其他 URL、命令或额外字段。使用模型生成参数的分支关闭传输重试，至多首次生成和一次修正；会话执行器直接使用已经确认并绑定来源的参数。实际调用后本轮不重复打开。**最终状态和链接由程序核对工具返回的身份、关键词、URL 和动作后生成。** 普通知识问答不打开浏览器；手动知识 Skill 与 MCP 操作冲突时提示切换回答方式。

浏览器位于运行后端的电脑，远程网页不会直接控制访问者的浏览器。返回“已发出打开请求”不表示网页已加载或已找到视频；工具不读取搜索结果。本次没有启动浏览器或运行联调，完整边界及待验收项见 [BROWSER_MCP.md](project_docs/features/BROWSER_MCP.md)。

## 项目结构

```text
Person_A/
├─ src/
│  ├─ knowledge_agent/
│  │  ├─ application.py         # 应用和会话生命周期
│  │  ├─ cli.py                 # 命令行入口
│  │  ├─ api.py                 # 原 Uvicorn 启动入口
│  │  ├─ config.py              # 公共配置
│  │  ├─ schemas.py             # 共享数据结构
│  │  ├─ agent/
│  │  │  ├─ runner.py           # 总编排
│  │  │  ├─ router.py           # 执行分支选择
│  │  │  ├─ state.py            # 单轮状态
│  │  │  ├─ preparation.py      # 检索和上下文准备
│  │  │  ├─ model_loop.py       # 模型与工具循环
│  │  │  ├─ prompts.py          # 系统提示词
│  │  │  ├─ assessment.py       # MBTI会话流程
│  │  │  ├─ intent.py           # 原意图接口的兼容导出
│  │  │  ├─ dialogue/           # 意图、待补参数和业务状态
│  │  │  ├─ browser/            # actions、semantic、execution
│  │  │  ├─ answering/          # answers、contract、diagnostics、citations
│  │  │  └─ planning/           # planner、executor、dependencies、topic_references
│  │  ├─ models/               # provider.py、diagnostics.py
│  │  ├─ session/              # store.py、memory.py
│  │  ├─ skills/               # loader.py、selector.py
│  │  ├─ observability/        # trace.py
│  │  ├─ knowledge/            # 文档、索引、检索及知识库管理
│  │  ├─ tools/                # 工具注册与本地知识工具
│  │  ├─ mcp/                  # 外部MCP客户端及适配
│  │  ├─ http/                 # HTTP应用、序列化、SSE及routes/
│  │  ├─ images/               # 图片提问扩展预留
│  │  ├─ evaluation.py         # 现有评估实现
│  │  ├─ session_store.py      # 旧存储导入路径的兼容入口
│  │  └─ skill_loader.py       # 旧技能加载路径的兼容入口
│  ├─ browser_mcp/             # 独立浏览器MCP服务
│  └─ mbti_mcp/                # 独立MBTI服务、题库及评分
├─ web/                       # 前端页面、JS和CSS
├─ config/                    # MCP运行配置及模板
├─ scripts/                   # MBTI和浏览器MCP启动脚本
├─ examples/                  # 演示MCP服务
├─ skills/                    # 技能正文SKILL.md
├─ docs/                      # 用户知识库教材
├─ project_docs/
│  ├─ README.md               # 开发文档导航
│  ├─ architecture/           # 当前目录、意图与规划设计
│  ├─ features/               # 各功能说明与待验收项
│  └─ archive/                # 旧计划和阶段性修复记录
├─ tests/
│  ├─ unit/                   # agent、session、skills、evaluation
│  └─ integration/            # HTTP和MCP集成用例
├─ evaluation/                # 题集和评估报告
├─ data/                      # 索引与会话数据库
├─ .tmp/                      # 临时文件与pytest缓存
├─ .env                       # 本地配置，不提交
├─ .env.example               # 配置模板
├─ pyproject.toml             # 包声明、依赖和命令入口
└─ README.md
```

本轮按职责迁移现有文件，内部导入和默认 MCP 启动配置已同步。根目录 `skills/` 保存技能正文，`src/knowledge_agent/skills/` 保存加载与选择代码；`docs/` 仍是知识库材料，不与开发说明混用。索引、会话数据库、真实 `.env`、用户教材及评估报告未移动或修改。

`api.py`、`agent/intent.py` 保留原入口，模型公共 API、旧存储与技能加载导入路径保留兼容。`person-a`、`python -m knowledge_agent.cli`、`uvicorn knowledge_agent.api:create_app` 的启动方式不变；默认配置的 MCP 启动脚本已由 `examples/` 调整到 `scripts/`。完整映射见 [目录与迁移说明](project_docs/architecture/LAYOUT.md)。**本轮未运行测试、项目导入、编译或服务联调，重启后端后需按清单统一验证。**

2 个旧测试临时目录已完整归入 `.tmp/legacy-pytest/`；其他目录移动中因 Windows 拒绝访问而未完全整理，原路径和归档路径可能各有残余。后续 pytest 缓存写入 `.tmp/pytest-cache/`。

## 数据与问答规则

建库流程：读取 `docs/` → 保留文档结构并分块 → 生成 Embedding → 发布片段、向量、关键词索引与文档清单。请求先结合会话状态识别意图并分流；知识问答执行混合检索 → 必要时补查章节 → 基于证据回答 → 校验并生成来源引用，MBTI 则进入独立答题与 MCP 工具流程。

- **对话与向量模型分开配置。** 入库和查询使用兼容的 Embedding 配置；更换向量模型或向量维度后需要重建索引。
- **保留 Markdown 结构。** 片段保存标题层级；代码、概念图和 Prompt 与对应说明保持关联。文档中的 Prompt 是知识内容，不是 Agent 的运行指令。
- **引用由程序生成。** 引用协议要求模型只输出本轮有效的来源标记，如 `[S1]`。程序校验标记，再使用检索记录生成完整文件名、章节和原文起止行号，模型不能自行编造来源元数据。同为第 26 章的两份文档通过完整文件名区分。
- **同步新增、修改和删除。** 文档清单记录内容与配置版本；更新时同步片段、向量和关键词记录，检索使用同一版索引，避免引用已删除或过期内容。
- **答案受证据范围约束。** 资料不足时说明缺失信息，只回答有依据的部分。教材中的业务示例不代表任何公司的正式政策；不同文档存在差异时展示出处和差异，不自行确定哪个说法权威。
- **MCP 结果有独立来源。** MBTI 问卷与评分来自本项目内置的 MCP 工具，核心代码源自 MindCare，不生成本地教材的 `[S1]` 引用；工具只接受实际用户作答，不用模型推测缺失答案。

## 待统一验收

自动 Skill 选择、技能检索策略边界修正、API/CLI/页面接入及 SSE 阶段进度已完成代码开发，尚未运行测试、语法检查、模型或服务联调。开发说明与未执行的验收清单见 [AUTO_SKILL_TODO.md](project_docs/features/AUTO_SKILL_TODO.md)。效果评估、请求标识与断线恢复继续暂缓。

`src/knowledge_agent/agent/prompts.py` 已与 Agent 统一为 `[S1]` 来源标记协议；最终文档名称、小节和行号由程序校验并生成，引用正确性仍需在真实问答中核对。

`evaluation/questions.jsonl` 保留 5 条人工验收草案，覆盖概念比较、检索与生成质量、资料不足、跨章节检索及无答案场景。每行记录问题、期望来源、回答要点和验收标准，目前没有执行结果或通过率。

开发完成后统一检查：首次建库与文档更新是否正确；检索是否命中对应章节；回答是否有原文支持；引用能否定位实际片段；多轮追问、清空会话和索引刷新是否符合预期；模型服务异常、参数错误与中断时是否正确退出。网页的会话切换、历史分页、清空与删除确认、请求异常提示也需纳入完整端到端联调。假运行时下的页面和路由检查不能代替真实模型与索引的验收。

新增 MBTI 接入还需统一验证：三工具发现与调用、完整 48 题评分、分页和版本不一致处理、暂停恢复与重启续答、会话隔离、旧题目卡重复点击、存储失败及 MCP 退出清理。具体清单见 [MBTI_MCP.md](project_docs/features/MBTI_MCP.md#待统一验收)。以上均为待执行项目，没有本次测试通过结论。

B 站搜索 MCP 的自然语言路由、参数双端校验、一次修正与防重复打开、真实浏览器唤起及旧流程兼容也尚未验证，清单见 [BROWSER_MCP.md](project_docs/features/BROWSER_MCP.md#待统一验收)。没有运行测试、导入检查、编译、安装或服务启动。
