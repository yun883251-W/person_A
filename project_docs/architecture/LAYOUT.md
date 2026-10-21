# 目录与模块职责

## 顶层目录

| 目录 | 职责 |
| --- | --- |
| `src/knowledge_agent/` | 主应用：Agent、知识库、会话、HTTP、评估与配置 |
| `src/mbti_mcp/` | MBTI 题库、评分和独立 MCP 服务 |
| `src/browser_mcp/` | 受限 B 站搜索 MCP 服务 |
| `web/` | 无构建步骤的原生 Web UI |
| `skills/` | Skill 元数据与指令正文 |
| `docs/` | 示例 Markdown 知识库 |
| `tests/` | 单元测试与集成测试 |
| `evaluation/` | 评估题集与结构化报告 |
| `config/` | MCP 运行配置和模板 |

## 主应用分层

```text
knowledge_agent/
├─ api.py / cli.py              # 对外入口
├─ application.py               # 应用与会话生命周期
├─ http/                        # FastAPI、SSE、序列化、路由、调度
├─ agent/
│  ├─ runner.py                 # 单轮总编排
│  ├─ dialogue/                 # 意图与业务状态
│  ├─ planning/                 # 多步骤计划与执行
│  ├─ answering/                # 回答校验与引用
│  └─ browser/                  # 浏览器操作语义与执行
├─ knowledge/                   # 解析、入库、索引、检索与管理
├─ tools/                       # 本地工具注册、校验和执行
├─ mcp/                         # MCP 生命周期与工具适配
├─ session/                     # SQLite 会话、记忆和上下文
├─ skills/                      # Skill 发现、加载和选择
├─ models/                      # 模型工厂、调用与错误诊断
└─ observability/               # 执行 trace
```

`api.py`、`knowledge_agent.models` 和 CLI 命令是稳定公共入口；内部模块按职责直接导入，避免仅为转发而存在的兼容层。

运行时数据位于 `data/`，本地凭据位于 `.env`。索引、SQLite、日志、缓存和真实凭据均由 `.gitignore` 排除。
