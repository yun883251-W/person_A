# Person_A · 可观测的本地知识库 Agent

Person_A 是一个面向 Python / LangChain 学习资料的完整 RAG Agent 应用。它将 Markdown 文档入库、混合检索、受控工具调用、多轮会话、任务规划和可观测性整合在同一套 CLI、HTTP API 与原生 Web UI 中。

项目重点不只是“接上大模型”，而是把一个 Agent 从实验脚本做成可运行、可恢复、可验证的应用：检索来源由程序校验，会话和业务状态持久化，工具权限按请求收窄，并对并发、超时、断连和错误返回建立清晰边界。

## 核心能力

- **混合 RAG**：结合向量检索、关键词检索与 RRF 融合；保留 Markdown 标题层级、代码块和原文行号。
- **可信引用**：模型只生成本轮来源标记，文件名、章节和行号由后端根据真实检索结果组装，阻止伪造引用。
- **Agent 工具循环**：支持知识库补查、MCP 工具发现、Schema 校验、调用预算、重复调用防护和结果回填。
- **多任务规划**：将“先解释概念，再搜索视频”等复合请求拆成最多 3 个有依赖关系的步骤，并保存执行结果。
- **业务对话状态**：区分新任务、继续、取消、闲聊与补充参数；待补信息和业务焦点随 SQLite 会话恢复。
- **自动 Skill**：按问题选择概念讲解、概念比较或代码解读策略，也支持手动指定。
- **会话与画像**：持久化完整历史、长对话摘要、可编辑记忆和本地学习画像。
- **MCP 扩展**：内置 MBTI 问卷服务与受限 B 站搜索服务；工具白名单和参数在宿主、服务两端校验。
- **可观测执行**：记录意图、检索、Skill、模型、工具、引用校验等阶段的状态、耗时和调用次数；Web 页面可展开查看。
- **有界并发**：不同会话并行、同一会话串行，管理写操作独占，并提供队列、超时与健康状态。

## 系统架构

```text
CLI / Web UI / REST + SSE
            │
            ▼
   FastAPI + ServiceRuntime
            │
            ▼
  Dialogue Resolver / Planner
       ┌────┼───────────┐
       ▼    ▼           ▼
     Chat   RAG       MCP Tools
            │       ┌────┴─────┐
            ▼       ▼          ▼
      Hybrid Search MBTI   Bilibili
            │
            ▼
      Markdown Index

SQLite：会话、业务状态、摘要、画像、执行记录
```

关键设计原则：

1. **数据与指令隔离**：检索到的文档和外部工具结果均视为不可信数据，不获得系统指令权限。
2. **权限来自当前输入**：模型不能自行扩大工具范围；浏览器操作必须能追溯到用户当前请求中的明确主题。
3. **状态原子提交**：回答、业务焦点和待补任务一起提交，失败不会留下半个会话轮次。
4. **结果可解释**：最终答案之外，系统提供公开的执行摘要，但不暴露模型内部推理或敏感配置。

## 技术栈

| 层次 | 技术 |
| --- | --- |
| Agent / RAG | Python 3.10+、LangChain、向量 + 关键词混合检索 |
| 模型 | DeepSeek Chat、Ollama Embedding（均可通过配置替换） |
| API | FastAPI、SSE、Uvicorn |
| 数据 | SQLite、JSON / JSONL 本地索引 |
| 工具协议 | MCP、JSON Schema、stdio 子进程 |
| 前端 | 原生 HTML、CSS、JavaScript |
| 测试 | pytest、FastAPI TestClient、可控假模型 / 假工具 |

## 快速开始

### 1. 安装

```powershell
git clone <your-repository-url>
cd Person_A
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[api,mcp]"
Copy-Item .env.example .env
```

编辑 `.env`，至少确认以下配置：

```dotenv
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_BASE_URL=http://localhost:11434

CHAT_PROVIDER=deepseek
CHAT_MODEL=deepseek-chat
CHAT_API_KEY=your_api_key
CHAT_BASE_URL=https://api.deepseek.com
```

`.env`、索引、SQLite 数据库和日志均已加入 `.gitignore`，不会进入版本库。

### 2. 构建知识库

确保 Ollama 已启动且向量模型可用：

```powershell
ollama pull qwen3-embedding:0.6b
person-a build
```

文档变化后使用增量同步入口：

```powershell
person-a update
```

### 3. 运行

命令行单次问答：

```powershell
person-a ask "RAG 的基本流程是什么？"
```

持久化多轮会话：

```powershell
person-a chat
person-a sessions
person-a chat --session-id <session-id>
```

启动 Web 与 API：

```powershell
python -m uvicorn knowledge_agent.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

打开：

- Web UI：<http://127.0.0.1:8000/>
- OpenAPI：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

页面与 API 同源，无需 Node.js 或单独启动前端服务。

## 使用示例

### 知识问答与引用

```text
用户：向量检索和关键词检索有什么区别？
Agent：根据本地资料回答，并返回可定位到文件、章节和行号的来源。
```

### 自动 Skill

```powershell
person-a ask "比较 Chain 和 Agent" --skill-mode auto
person-a ask "解释这段 Python 代码" --skill code-explanation
```

自动模式只向分类器提供 Skill 元数据；回答阶段只加载最终选中的 Skill 正文，避免无关指令污染上下文。

### 多步骤任务

```text
先解释 Python 装饰器，再在 B 站搜索 Python 装饰器教程。
```

系统会构建受限计划、分配共享调用预算，并在页面“执行详情”中展示每一步的完成状态。

### MBTI MCP

在 Web 会话或 `person-a chat` 中输入：

```text
开始 MBTI 测试
```

问卷支持暂停、恢复、重新开始和结果查询。评分由确定性 MCP 工具完成，不使用模型猜测缺失答案。

### B 站搜索 MCP

只打开搜索页：

```text
帮我在 B 站搜索 Python 装饰器教程
```

读取公开标题与链接需要浏览器扩展依赖：

```powershell
python -m pip install -e ".[api,browser]"
python -m playwright install chromium
```

随后可请求：

```text
帮我在 B 站找 5 个 Python 装饰器教程，把标题和链接整理出来
```

浏览器能力仅允许固定 B 站搜索地址，不接受任意 URL、脚本或命令；不登录、不绕过验证码、不播放或下载视频。

## CLI 命令

| 命令 | 作用 |
| --- | --- |
| `person-a build` | 完整构建并原子发布索引 |
| `person-a update` | 同步 Markdown 文档变化 |
| `person-a ask "问题"` | 单次问答，支持 `--json` |
| `person-a chat` | 新建或恢复持久化会话 |
| `person-a sessions` | 列出本地会话 |
| `person-a history ID` | 分页查看完整历史 |
| `person-a delete-session ID` | 删除指定会话 |
| `person-a evaluate` | 运行评估题集并生成报告 |

`chat` 内支持 `/help`、`/clear`、`/refresh` 和 `/exit`。使用 `person-a --help` 查看完整参数。

## HTTP API

主要资源包括：

- `/sessions`：会话 CRUD
- `/sessions/{id}/messages`：问答与历史
- `/sessions/{id}/messages/stream`：SSE 阶段进度与最终结果
- `/sessions/{id}/business-state`：安全的业务状态投影
- `/sessions/{id}/memory`：会话记忆管理
- `/profile`：本地学习画像
- `/knowledge/*`：文档、预览、上传与索引管理
- `/skills`：可用 Skill 元数据
- `/concurrency`：并发配置与在途状态

示例：

```powershell
$baseUrl = "http://127.0.0.1:8000"
$session = Invoke-RestMethod -Method Post -Uri "$baseUrl/sessions" `
  -ContentType "application/json" -Body '{"title":"RAG 学习"}'

Invoke-RestMethod -Method Post `
  -Uri "$baseUrl/sessions/$($session.session_id)/messages" `
  -ContentType "application/json" `
  -Body '{"question":"RAG 的基本流程是什么？","skill_mode":"auto"}'
```

## 测试与质量

完整离线测试不依赖真实 DeepSeek、Ollama 或浏览器：

```powershell
python -m pytest -q
```

当前基线：**113 passed，4 skipped**。默认跳过项是需要外部提供方凭据或服务的在线探测，可由对应环境变量显式开启。

测试覆盖：

- 会话存储、事务回滚、幂等写入与 schema 迁移
- 意图边界、补参、取消、恢复和多轮业务状态
- Agent 调用预算、Skill 路由隔离和多步骤计划
- MCP 工具发现、Schema 校验、stdio 生命周期
- HTTP 静态资源边界、错误结构和 API 兼容性
- 评估题集解析、批处理容错与报告输出

## 运行边界

- 服务定位为本机单用户应用，画像使用固定 `local` 用户，不用于无身份隔离的公网多租户部署。
- Uvicorn 使用 `--workers 1`；进程内可有界并发，多进程协调不在本项目运行模型内。
- SSE 传输公开阶段和完整最终结果，不逐 Token 输出正文。
- 外部模型、Ollama、B 站页面和浏览器可用性由运行环境决定，失败会转换为结构化业务状态，不伪装成成功结果。
- 默认配置只允许受控、只读或显式授权的工具行为。

## 项目结构

```text
Person_A/
├─ src/knowledge_agent/     # Agent、RAG、会话、API、评估与可观测性
├─ src/mbti_mcp/            # MBTI 题库、评分与 MCP 服务
├─ src/browser_mcp/         # 受限 B 站搜索 MCP 服务
├─ web/                     # 原生 Web UI
├─ skills/                  # 可加载 Skill 定义
├─ docs/                    # 示例 Markdown 知识库
├─ tests/                   # 单元与集成测试
├─ evaluation/              # 评估题集与报告
├─ config/                  # MCP 配置与模板
├─ scripts/                 # MCP 服务入口和压测脚本
└─ project_docs/            # 架构与功能设计文档
```

更深入的设计说明见 [项目文档导航](project_docs/README.md)。

## 项目亮点总结

这个项目展示了从 RAG 原型走向工程化 Agent 的关键能力：检索与引用可信、状态可恢复、工具调用可约束、复杂任务可规划、运行过程可观测、异常与并发有边界，并且提供从 CLI 到 Web/API 的完整交付形态。
