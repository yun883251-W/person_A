# 测试目录

按被验证的职责组织现有用例。本轮仅移动文件并调整导入和替身位置，**没有运行测试、编译检查、真实模型调用或服务联调**。

```text
tests/
├── README.md
├── unit/
│   ├── agent/
│   │   └── test_intent_budget.py
│   └── storage/
│       └── test_session_store.py
└── integration/
    └── http/
        └── test_static_routes.py
```

## 保留的覆盖

- `unit/agent/test_intent_budget.py`：规则问候不调用模型，意图识别和回答共用预算，零预算拒绝执行，失败调用仍计费，以及异常信息不进入公开诊断。分类调用必须关闭传输重试，旧的“两次传输尝试算一次逻辑调用”断言已改为只尝试一次并回退。
- `unit/storage/test_session_store.py`：SQLite 创建和读写、既有 `turn_id` 幂等与冲突、事务回滚、锁与路径错误，以及应用重启后的会话恢复、隔离、清空和删除。数据库只创建于 `tmp_path`，模型与检索器使用替身。这里验证已有会话存储行为，不代表已开发请求标识或断线恢复功能。
- `integration/http/test_static_routes.py`：页面和资源路由、目录访问边界、健康检查、接口文档及会话不存在时的响应。通过 FastAPI `TestClient` 使用进程内请求和运行时替身，不启动外部服务器，不访问真实索引或会话数据库。

拆分之后，替身应设置在函数实际使用的模块：意图入口在 `agent.runner`，问题与模型准备在 `agent.preparation`，回答调用在 `agent.model_loop`；静态资源根路径位于 `http.app`。`knowledge_agent.api.create_app` 仍作为兼容入口受到静态路由用例覆盖。

## 依赖与统一验证

现有 `dev` 可选依赖提供 `pytest`、`httpx`；HTTP 用例另需 `api` 可选依赖的 FastAPI。缺少 FastAPI 或 httpx 时，该文件会跳过，跳过不代表验证通过。依赖未在本轮安装。

开发完成并安排统一验证时，再按目录执行已有用例。首先确认拆分后的导入路径和运行时替身有效，再运行全部保留用例。本轮不增加评估模块、请求标识或断线恢复的开发与测试。

## 从空占位文件迁移的待验证项

原 `test_ingest.py` 和 `test_retrieval.py` 只有模块说明，没有可运行的用例，已删除。以下事项集中保留，后续需要开发相应用例后再验证：

- 文档入库：标题层级、代码与解释的关联、原文行号、文件名唯一性、文件增删改后的片段与索引一致性、Embedding 配置变更后的重建要求。
- 检索：中文语义查询、技术标识符查询、结果去重与引用保留、未建立索引时的提示，以及删除文档后旧内容不再被召回。

## 尚未清理的运行缓存

本次清理范围还发现旧 `tests/__pycache__/`、`.pytest_cache/`、`.pytest-tmp-session-store/` 及 `.tmp/` 内的 pytest 临时目录和 MCP smoke 文件。部分目录读取被操作系统拒绝；批量删除操作随后被自动审批策略拦截，未执行删除。以上本地缓存仍保留，已由 `.gitignore` 忽略；它们不属于迁移后保留的测试源码。
