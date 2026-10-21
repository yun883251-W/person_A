# 项目开发文档

这里集中保存 Person_A 的设计说明、实现记录和待验收清单。知识库教材仍在项目根目录的 `docs/`；开发文档的归档不改变 `KNOWLEDGE_DOCS_DIR`，也不会重建索引或修改已有会话数据。

| 文档 | 用途 |
| --- | --- |
| [单进程有界并发](features/CONCURRENCY.md) | 协程调度、独立通道、共享 MCP、限流、管理屏障及压测边界 |
| [聊天与会话记忆](features/CHAT_MEMORY.md) | 回答模式、上下文预算、摘要、纠错与遗忘边界、待验收项 |
| [用户画像与候选提取](features/USER_PROFILE.md) | 字段来源时间、人工确认、模型候选、接口与边界 |
| [规划、检索与回答可靠性修复](archive/RELIABILITY_FIXES.md) | 本轮十项问题的处理、相关性门槛、回答状态协议、日志与未验证范围 |
| [语义意图、多任务组合与长对话摘要](features/SEMANTIC_PLANS_MEMORY.md) | 新增能力、配置、调用预算、摘要存储和未执行的使用用例 |
| [意图边界与任务规划](architecture/INTENT_PLANNING.md) | 知识咨询、工具指令边界，以及有限步骤规划与共享预算 |
| [会话意图与任务承接开发清单](architecture/DIALOGUE_INTENT_TODO.md) | 一期开发记录；记录单焦点/单 pending、当时的 SQLite v3（当前为 v4）、公开投影及剩余真实联调项 |
| [自动 Skill 与检索策略](features/AUTO_SKILL_TODO.md) | 自动选择、技能专用检索、前后端接入和未执行的验收清单 |
| [MBTI MCP 接入](features/MBTI_MCP.md) | 内置问卷、评分工具、会话答题及配置方式 |
| [B 站搜索 MCP](features/BROWSER_MCP.md) | 自然语言提取搜索主题、固定 B 站链接、本机浏览器唤起及待验收项 |
| [意图与 MCP 初始计划](archive/INTENT_MCP_TODO.md) | 原始开发任务存档；不作为当前实现完成度的判断依据 |

## 本次目录迁移

当前目录与全部迁移映射见 [项目目录与迁移说明](architecture/LAYOUT.md)。

- `architecture/`：当前项目结构、会话意图和任务规划。
- `features/`：技能、MBTI、浏览器及摘要等功能说明。
- `archive/`：旧开发计划和阶段性修复记录；其中的测试记录仅适用于对应历史版本。

代码按 `agent/dialogue`、`browser`、`answering`、`planning` 归组，公共模型、会话、技能和执行记录分别位于 `models/`、`session/`、`skills/`、`observability/`。这些目录均相对 `src/knowledge_agent/`；独立 MCP 服务仍在 `src/browser_mcp/` 和 `src/mbti_mcp/`。

正式 MCP 启动脚本在 `scripts/`，演示服务在 `examples/`。已有启动命令保持不变：

```powershell
person-a build
person-a ask "什么是 RAG？"
person-a chat
python -m uvicorn knowledge_agent.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

最近的聊天、记忆和画像功能仅完成开发与文档同步，未执行上述命令、测试、服务或模型联调。历史迁移与验收记录不代表当前版本已通过验证。

返回 [项目 README](../README.md)。
