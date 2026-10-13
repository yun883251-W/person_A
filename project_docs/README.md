# 项目开发文档

这里集中保存 Person_A 的设计说明、实现记录和待验收清单。知识库教材仍在项目根目录的 `docs/`；开发文档的归档不改变 `KNOWLEDGE_DOCS_DIR`，也不会重建索引或修改已有会话数据。

| 文档 | 用途 |
| --- | --- |
| [意图边界与任务规划](INTENT_PLANNING.md) | 知识咨询、工具指令边界，以及有限步骤规划与共享预算 |
| [自动 Skill 与检索策略](AUTO_SKILL_TODO.md) | 自动选择、技能专用检索、前后端接入和未执行的验收清单 |
| [MBTI MCP 接入](MBTI_MCP.md) | 内置问卷、评分工具、会话答题及配置方式 |
| [意图与 MCP 初始计划](INTENT_MCP_TODO.md) | 原始开发任务存档；不作为当前实现完成度的判断依据 |

## 本次目录迁移

Python 路径以下均相对 `src/knowledge_agent/`。调整按职责拆分现有代码，不新增评估模块或请求级断线恢复。

| 原位置或职责 | 当前位置 |
| --- | --- |
| 根目录四份开发 Markdown 文档 | `project_docs/`，保留原文件名 |
| `agent/runner.py` 的应用及持久化会话生命周期 | `application.py` |
| `agent/runner.py` 的单轮状态和准备结果 | `agent/state.py` |
| `agent/runner.py` 的技能、检索与上下文准备 | `agent/preparation.py` |
| `agent/runner.py` 的模型和工具调用循环 | `agent/model_loop.py` |
| `agent/runner.py` 的回答、引用校验与结果摘要 | `agent/answers.py` |
| `agent/runner.py` 的 KnowledgeAgent 编排 | 继续位于 `agent/runner.py` |
| `api.py` 的应用创建和生命周期 | `http/app.py` |
| `api.py` 的请求及公开响应模型 | `http/schemas.py` |
| `api.py` 的领域结果到公开响应转换 | `http/serialization.py` |
| `api.py` 的错误响应和异常处理 | `http/errors.py` |
| `api.py` 的单工作线程运行时 | `http/runtime.py` |
| `api.py` 的实时阶段通道和 SSE 输出 | `http/streaming.py` |
| `api.py` 的提问、会话、知识库及系统路由 | `http/routes/chat.py`、`sessions.py`、`knowledge.py`、`system.py` |
| 根 `tests/` 中的实际验证代码 | `tests/unit/agent/`、`tests/unit/storage/`、`tests/integration/http/` |

原有 `api.py` 继续提供兼容启动入口；应用生命周期类型从 `application.py` 导入，原 `agent.runner` 相关导出保留兼容。纯 TODO 测试占位不再与实际验证代码混放；待开发功能仍由其源码说明和开发清单记录。

启动命令保持原样，从项目根目录运行：

```powershell
person-a build
person-a ask "什么是 RAG？"
person-a chat
python -m knowledge_agent.cli ask "什么是 RAG？"
python -m uvicorn knowledge_agent.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

本次只完成目录整理和代码拆分，没有执行上述命令，也没有运行测试、导入检查、编译、服务或模型联调。原有验证项目需要在用户安排统一验收时再运行，不能把文件已迁移视为测试通过。

返回 [项目 README](../README.md)。
