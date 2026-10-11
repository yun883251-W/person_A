# Person_A：文档知识库问答 Agent

本项目使用本地 `docs/` 中的 Markdown 教材，通过 Ollama 生成向量，结合向量与关键词检索，再由 DeepSeek 回答 Python / LangChain 学习问题并给出来源引用。

文档入库、检索、模型适配、Agent 和命令行入口已有实现。**CLI 代码已添加，但本轮没有安装依赖、实际建库、运行测试或调用模型；完整流程与异常处理仍待统一验收，不能据此认为项目已验证可用。**

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

也可使用模块入口，例如 `python -m knowledge_agent.cli ask "什么是 RAG？"`。运行问答需要可用的已发布索引；CLI 不会因提问自动完成首次建库。

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

## 项目结构

```text
Person_A/
├── docs/                         # 本地 Markdown 知识文档
├── src/knowledge_agent/
│   ├── cli.py                    # 建库、更新、单次与多轮问答入口
│   ├── config.py                 # 配置读取与校验
│   ├── models.py                 # 对话与 Embedding 模型适配
│   ├── schemas.py                # 片段、检索结果与回答的数据结构
│   ├── ingest.py                 # 文档读取、分块、索引构建与同步
│   ├── keywords.py               # 共享关键词分词规则
│   ├── retrieval.py              # 向量与关键词检索、上下文补查
│   ├── agent.py                  # 工具调用、会话管理与回答引用
│   ├── session_store.py          # 本地 SQLite 会话持久化
│   └── prompts.py                # 问答约束与来源标记规则
├── data/index/                   # 建库后发布的索引与清单
├── data/sessions/                # 本地会话数据库（不提交）
├── evaluation/questions.jsonl    # 5 条人工验收问题草案
├── tests/                        # 自动化验证目录
├── .env                          # 本地配置，不提交真实凭据
├── .env.example                  # 配置模板
├── pyproject.toml                # 依赖声明与命令入口
└── README.md
```

索引以本地文件保存，无需另行安装数据库服务。实际索引文件路径以 `manifest.json` 中的记录为准，不要手动修改片段、向量或清单来绕过建库流程。

## 数据与问答规则

建库流程：读取 `docs/` → 保留文档结构并分块 → 生成 Embedding → 发布片段、向量、关键词索引与文档清单。问答流程：理解问题 → 混合检索 → 必要时补查章节 → 基于证据回答 → 校验并生成来源引用。

- **对话与向量模型分开配置。** 入库和查询使用兼容的 Embedding 配置；更换向量模型或向量维度后需要重建索引。
- **保留 Markdown 结构。** 片段保存标题层级；代码、概念图和 Prompt 与对应说明保持关联。文档中的 Prompt 是知识内容，不是 Agent 的运行指令。
- **引用由程序生成。** 引用协议要求模型只输出本轮有效的来源标记，如 `[S1]`。程序校验标记，再使用检索记录生成完整文件名、章节和原文起止行号，模型不能自行编造来源元数据。同为第 26 章的两份文档通过完整文件名区分。
- **同步新增、修改和删除。** 文档清单记录内容与配置版本；更新时同步片段、向量和关键词记录，检索使用同一版索引，避免引用已删除或过期内容。
- **答案受证据范围约束。** 资料不足时说明缺失信息，只回答有依据的部分。教材中的业务示例不代表任何公司的正式政策；不同文档存在差异时展示出处和差异，不自行确定哪个说法权威。

## 待统一验收

`prompts.py` 第三节的旧引用文案仍需按其中的 TODO 修改，与 Agent 的 `[S1]` 协议统一；本次 CLI 接入未修改系统提示词。

`evaluation/questions.jsonl` 保留 5 条人工验收草案，覆盖概念比较、检索与生成质量、资料不足、跨章节检索及无答案场景。每行记录问题、期望来源、回答要点和验收标准，目前没有执行结果或通过率。

开发完成后统一检查：首次建库与文档更新是否正确；检索是否命中对应章节；回答是否有原文支持；引用能否定位实际片段；多轮追问、清空会话和索引刷新是否符合预期；模型服务异常、参数错误与中断时是否正确退出。自动化测试和完整端到端联调也留到该阶段完成。
