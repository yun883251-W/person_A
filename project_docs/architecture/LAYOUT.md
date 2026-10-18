# 项目目录与迁移说明

本轮按职责迁移现有实现，并同步源码引用、测试引用、默认 MCP 配置和文档链接。没有修改业务算法、数据库表结构或索引格式；没有运行测试、项目导入、编译、模型或服务联调。

## 代码放在哪里

路径相对 `src/knowledge_agent/`。

| 位置 | 职责 |
| --- | --- |
| `application.py`、`cli.py`、`api.py` | 生命周期、命令行、兼容 HTTP 启动入口 |
| `config.py`、`schemas.py` | 公共配置和跨模块数据结构 |
| `agent/runner.py`、`preparation.py`、`model_loop.py` | 总编排、上下文准备、模型与工具循环 |
| `agent/dialogue/` | 唯一会话意图入口、待补参数、业务状态及受保护派发 |
| `agent/browser/` | B 站请求边界、语义识别与单动作执行 |
| `agent/planning/` | 多步骤计划、依赖关系和同轮主题引用 |
| `agent/planning/steps.py` | 单个计划步骤的上下文准备、模型/工具执行和答案校验；依赖显式传入，不提交会话 |
| `agent/answering/` | 回答状态协议、引用校验和诊断 |
| `agent/assessment.py` | MBTI 会话问答和工具编排 |
| `models/` | 对话与 Embedding 模型适配、模型诊断 |
| `session/` | SQLite 会话存储、知识对话摘要 |
| `skills/` | 技能加载、元数据与自动选择 |
| `observability/` | 执行事件与公开追踪记录 |
| `knowledge/` | 文档入库、索引、检索和相关性判断 |
| `knowledge/markdown.py` | Markdown 读取、结构解析、保留行号的分块、来源元数据和稳定片段 ID |
| `knowledge/ingest.py` | 入库配置与流程编排、片段导出、向量化、完整索引发布和增量更新 |
| `knowledge/index_reader.py` | 已发布索引的 manifest、片段、向量、关键词与元数据读取校验；构建内存快照 |
| `tools/` | 宿主工具注册与本地知识工具 |
| `mcp/` | 外部 MCP 连接、发现、适配和生命周期 |
| `http/` | HTTP 路由、序列化、错误和 SSE |
| `evaluation.py` | 现有离线评估实现 |

包初始化保持轻量。内部调用从具体实现模块导入，不通过兼容入口绕行。`agent/dialogue/browser_dispatch.py` 继续负责跨轮待办的状态和来源，`agent/browser/execution.py` 负责模型工具调用；两者没有被合并成重复的会话入口。

## 旧文件到新文件

| 旧位置 | 新位置 |
| --- | --- |
| `agent/browser_actions.py` | `agent/browser/actions.py` |
| `agent/browser_execution.py` | `agent/browser/execution.py` |
| `agent/semantic_intent.py` | `agent/browser/semantic.py` |
| `agent/answers.py` | `agent/answering/answers.py` |
| `agent/answer_contract.py` | `agent/answering/contract.py` |
| `agent/answer_diagnostics.py` | `agent/answering/diagnostics.py` |
| `agent/citations.py` | `agent/answering/citations.py` |
| `agent/planner.py` | `agent/planning/planner.py` |
| `agent/plan_executor.py` | `agent/planning/executor.py` |
| `agent/dependencies.py` | `agent/planning/dependencies.py` |
| `agent/topic_references.py` | `agent/planning/topic_references.py` |
| `skill_loader.py` | `skills/loader.py` |
| `agent/skill_selector.py` | `skills/selector.py` |
| `session_store.py` | `session/store.py` |
| `agent/memory.py` | `session/memory.py` |
| `models.py` | `models/provider.py` |
| `model_diagnostics.py` | `models/diagnostics.py` |
| `agent/trace.py` | `observability/trace.py` |

`session_store.py`、`skill_loader.py`、`agent/intent.py` 三个兼容文件已删除，项目内引用及现有测试导入已迁移。存储使用 `knowledge_agent.session.store`，技能加载使用 `knowledge_agent.skills.loader`；意图类型使用 `agent.dialogue.intent_types`，分类使用 `agent.dialogue.classifier`，规则使用 `agent.dialogue.rules`，会话入口使用 `agent.dialogue.resolver`。项目外脚本若使用这三个旧路径，也需同步修改。

`api.py` 保留原 HTTP 启动入口。`knowledge_agent.models` 改为包，其公共模型 API 通过 `models/__init__.py` 延迟导出。本次移除的是导入转发层，实际业务实现保持在各自模块中；未运行测试或服务。

## 项目根目录

- `docs/`：用户教材；默认知识库来源不变。
- `project_docs/architecture/`：当前结构、意图和规划设计。
- `project_docs/features/`：技能、MCP 等功能说明及待验收项。
- `project_docs/archive/`：阶段性修复记录和旧开发计划，保留历史验证范围。
- `skills/`：技能正文，与 Python 包 `knowledge_agent.skills` 分工不同。
- `scripts/`：正式使用的 MBTI、浏览器 MCP 启动脚本；`examples/` 保留演示服务。
- `tests/unit/`：按 `agent/`、`session/`、`skills/`、`evaluation/` 归类；`tests/integration/` 保留 HTTP/MCP 集成用例。
- `evaluation/`：评估题集与报告；已有报告未移动或删除。
- `data/`：向量索引和会话数据库；位置及内容未变。
- `.tmp/`：临时文件；后续 pytest 缓存由配置写到 `.tmp/pytest-cache/`。

旧测试临时目录中有 2 个完整移入 `.tmp/legacy-pytest/`。其他目录移动时遇到 Windows 拒绝访问，原位置及归档位置可能各保留一部分内容，尚未完全整理；本轮没有删除缓存或修改文件访问权限。

默认配置与模板已采用 `scripts/` 启动路径。自定义的仓库外 MCP 配置若仍引用旧 `examples/` 启动脚本，应同步修改。后端正在运行时，需重启后加载新模块位置。原 `person-a`、CLI 和 `uvicorn knowledge_agent.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1` 命令保持不变。

## 后续统一验证

`retrieval.py` 的索引文件读取已迁入 `knowledge/index_reader.py`。读取层只接收目录，不依赖检索配置或模型；检索模块保留接受 `RetrievalConfig` 的加载入口，负责默认配置解析，再调用读取层。`IndexSnapshot`、`PublishedIndexArtifacts`、`IndexNotBuiltError` 和片段加载函数仍可从原检索模块导入，内部知识工具直接引用读取层的异常类型。文件格式、路径边界和快照一致性校验原样保留；建库及管理模块各自的读取逻辑尚未合并。本轮未运行测试，后续需覆盖缺失/损坏索引、快照加载及热更新。

`runner.py` 的单步执行已迁入 `planning/steps.py`，直接传入模型、检索器、工具注册器、配置和提示词；计划调度与最终会话提交仍留在 runner，预算分配和依赖调度继续由 `planning/executor.py` 负责。规划错误码的状态及提示映射集中在 `planning/planner.py`。原私有 `_run_planned_step` 方法已移除，没有增加转发层；如有外部替身或补丁，需调整到新函数所在模块。本轮仅静态审阅，知识步骤、依赖证据和 MCP 步骤的回归尚未运行。

文档读取与分块已从 `knowledge/ingest.py` 提取到 `knowledge/markdown.py`，原函数实现及片段算法原样迁移。`manager.py` 直接引用新模块；`ingest.py` 通过显式导入保留原有公开类型和函数的导入路径，未增加独立兼容文件。解析用私有辅助函数随实现迁移，不作为兼容 API。新模块只依赖标准库与 `schemas.DocumentChunk`，不反向依赖入库编排。

此次拆分尚未运行测试、项目导入或模型联调。统一验证时需覆盖分块行号、代码围栏、重叠边界、片段 ID，以及完整建库和增量更新结果的一致性。

浏览器执行的公共逻辑已集中：`browser_mcp/contracts.py` 构造成功结果契约，供 MCP 服务返回、宿主预算估算和结果校验使用；`agent/browser/execution.py` 提供结果校验、公开错误码提取及回复文案，`agent/dialogue/browser_dispatch.py` 复用这些函数。后者仍负责槽位授权、派发前持久化及最终业务状态转换。超时和连接断开不能证明未执行，仍按未确认结果处理，不自动重放。

`runner.py` 已移除只有单一调用点的 `_dispatch_bilibili` 私有转发方法，直接调用会话派发函数。公共模型方法、诊断模块和 MCP 生命周期入口保留原有职责。本轮未新增 Python 文件，未运行测试或服务；浏览器两条执行路径的回归验证仍待进行。

本轮只进行文件迁移和文本审阅。后续需验证新旧公共导入入口、CLI/API 启动、默认 MCP 配置发现、会话恢复、待补搜索、知识问答、规划及技能选择；现有测试的通过记录不能替代此次迁移的验证。
