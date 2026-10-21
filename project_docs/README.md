# Person_A 项目文档

这里记录系统的架构决策、功能契约与运行边界。项目使用方式、能力概览和验证基线请先阅读根目录 [README](../README.md)。

## 架构

| 文档 | 内容 |
| --- | --- |
| [目录与模块职责](architecture/LAYOUT.md) | 源码分层、入口与模块依赖 |
| [对话意图与业务状态](architecture/DIALOGUE_INTENT.md) | 意图解析、补参、状态转换和派发保护 |
| [意图边界与任务规划](architecture/INTENT_PLANNING.md) | 单步/多步路由、依赖与共享预算 |

## 功能

| 文档 | 内容 |
| --- | --- |
| [自动 Skill](features/AUTO_SKILL.md) | 选择策略、加载边界和检索策略 |
| [聊天与记忆](features/CHAT_MEMORY.md) | 回答模式、摘要和可编辑记忆 |
| [学习画像](features/USER_PROFILE.md) | 本地画像字段、候选提取和注入边界 |
| [MBTI MCP](features/MBTI_MCP.md) | 工具契约、确定性评分和会话恢复 |
| [浏览器 MCP](features/BROWSER_MCP.md) | B 站搜索授权边界与结果模式 |
| [并发模型](features/CONCURRENCY.md) | 调度、会话串行化、超时和关闭行为 |
| [语义补充、规划与摘要](features/SEMANTIC_PLANS_MEMORY.md) | 语义分类、自然语言任务拆分和长对话摘要 |

## 验证

离线自动化测试统一位于 `tests/`，执行：

```powershell
python -m pytest -q
```

当前基线为 **113 passed，4 skipped**。在线提供方探测默认关闭，以保证常规测试可重复且不会产生外部调用。
