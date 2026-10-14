# 测试目录

按被验证的职责组织现有用例。本轮仅移动文件并调整导入和替身位置，**没有运行测试、编译检查、真实模型调用或服务联调**。

```text
tests/
├── README.md
├── conftest.py          # 共享装置：真实 MCP 发布面、在线探测开关
├── unit/
│   ├── agent/
│   │   └── test_intent_budget.py
│   └── storage/
│       └── test_session_store.py
└── integration/
    ├── http/
    │   └── test_static_routes.py
    └── mcp/
        ├── test_published_tool_schema.py   # 离线：发布面保守子集
        └── test_live_provider_schema.py    # 在线：真实提供方是否接受
```

## 保留的覆盖

- `unit/agent/test_intent_budget.py`：规则问候不调用模型，意图识别和回答共用预算，零预算拒绝执行，失败调用仍计费，以及异常信息不进入公开诊断。分类调用必须关闭传输重试，旧的“两次传输尝试算一次逻辑调用”断言已改为只尝试一次并回退。
- `unit/storage/test_session_store.py`：SQLite 创建和读写、既有 `turn_id` 幂等与冲突、事务回滚、锁与路径错误，以及应用重启后的会话恢复、隔离、清空和删除。数据库只创建于 `tmp_path`，模型与检索器使用替身。这里验证已有会话存储行为，不代表已开发请求标识或断线恢复功能。
- `integration/http/test_static_routes.py`：页面和资源路由、目录访问边界、健康检查、接口文档及会话不存在时的响应。通过 FastAPI `TestClient` 使用进程内请求和运行时替身，不启动外部服务器，不访问真实索引或会话数据库。

拆分之后，替身应设置在函数实际使用的模块：意图入口在 `agent.runner`，问题与模型准备在 `agent.preparation`，回答调用在 `agent.model_loop`；静态资源根路径位于 `http.app`。`knowledge_agent.api.create_app` 仍作为兼容入口受到静态路由用例覆盖。

## MCP 工具 Schema 契约（新增）

`mcp__browser__search_bilibili` 曾把 `[\ud800-\udfff]`（UTF-16 代理区范围）写进 `pattern`。本地全部通过 —— `jsonschema` 接受、运行时发现工具、当时所有用例为绿 —— 但 DeepSeek 的请求校验拒绝整个请求，该功能在完整的开发与提交流程之后 **100% 不可用**，而离线用例一个都没发现。

这一层以前完全没有覆盖，现在由两个文件分工：

- `integration/mcp/test_published_tool_schema.py`（离线，CI 无条件运行）：检查**真正发给提供方的载荷**（`convert_to_openai_tool` 的输出），而不是项目内部的 Schema 副本。约束包括：不含孤立代理字符、不含代理区转义、正则可编译、关键字都在 `_KNOWN_GOOD_KEYWORDS` 登记、配置里 `enabled` 的服务器必须真的发布出工具。最后一条覆盖另一种静默失败：服务器启动超时，配置看起来启用但模型一个工具都看不到。
- `integration/mcp/test_live_provider_schema.py`（在线，默认跳过）：把每个工具的真实 Schema 发给真实提供方，先跑一个不带工具的对照请求确认配置可用，再逐个工具、逐个服务器分组探测。分组探测复现生产绑定方式（MBTI 三个工具同轮绑定，浏览器工具受单工具轮约束自成一组）。

`_KNOWN_GOOD_KEYWORDS` 是**经验清单，不是规范子集**：新增关键字前，必须先用在线探测确认真实提供方接受，再登记。示例：`allOf`、`not`、`patternProperties` 都已对照真实提供方确认可用。

两个文件都只**绑定**工具、从不执行，因此浏览器工具不会被触发，不会打开页面。

在线探测需要真实凭据，本地运行：

```bash
PERSON_A_LIVE_SCHEMA_PROBE=1 pytest tests/integration/mcp/test_live_provider_schema.py -v
```

PowerShell：`$env:PERSON_A_LIVE_SCHEMA_PROBE="1"; ...`。已开启探测却读不到对话模型配置时报错而非跳过 —— 静默通过会把“没跑”伪装成“通过”。

## 持续集成

- `.github/workflows/ci.yml`：push 与 PR 上运行 `pytest -q`，只跑离线用例，不需要凭据，不消耗额度。当前只验证 Python 3.12；`pyproject` 声明 `requires-python >=3.10`，若要同时验证 3.10，把 workflow 里的 `python-version` 换成 matrix。
- `.github/workflows/mcp-live-schema.yml`：仅手动触发（`workflow_dispatch`）的在线探测，需要仓库 secret `CHAT_API_KEY`、`CHAT_MODEL`。不挂在 push 上：它会真实调用提供方，且失败原因可能是网络抖动而非 Schema 问题，不适合当作提交通道。

## 依赖与统一验证

现有 `dev` 可选依赖提供 `pytest`、`httpx`；HTTP 用例另需 `api` 可选依赖的 FastAPI。缺少 FastAPI 或 httpx 时，该文件会跳过，跳过不代表验证通过。依赖未在本轮安装。

开发完成并安排统一验证时，再按目录执行已有用例。首先确认拆分后的导入路径和运行时替身有效，再运行全部保留用例。本轮不增加评估模块、请求标识或断线恢复的开发与测试。

## 从空占位文件迁移的待验证项

原 `test_ingest.py` 和 `test_retrieval.py` 只有模块说明，没有可运行的用例，已删除。以下事项集中保留，后续需要开发相应用例后再验证：

- 文档入库：标题层级、代码与解释的关联、原文行号、文件名唯一性、文件增删改后的片段与索引一致性、Embedding 配置变更后的重建要求。
- 检索：中文语义查询、技术标识符查询、结果去重与引用保留、未建立索引时的提示，以及删除文档后旧内容不再被召回。

## 尚未清理的运行缓存

本次清理范围还发现旧 `tests/__pycache__/`、`.pytest_cache/`、`.pytest-tmp-session-store/` 及 `.tmp/` 内的 pytest 临时目录和 MCP smoke 文件。部分目录读取被操作系统拒绝；批量删除操作随后被自动审批策略拦截，未执行删除。以上本地缓存仍保留，已由 `.gitignore` 忽略；它们不属于迁移后保留的测试源码。
