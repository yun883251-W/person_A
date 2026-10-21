# 测试说明

运行完整离线测试：

```powershell
python -m pytest -q
```

当前基线为 **113 passed，4 skipped**。默认跳过项带有 `live` 标记，需要外部提供方凭据或本地服务；常规测试不会访问 DeepSeek、Ollama 或真实浏览器。

## 分层

- `tests/unit/agent/`：意图、业务状态、预算、Skill 隔离和任务规划。
- `tests/unit/session/`：SQLite 初始化、迁移、事务、幂等和锁冲突。
- `tests/unit/evaluation/`：题集校验、批处理与报告输出。
- `tests/integration/http/`：静态资源边界、OpenAPI 和错误响应。
- `tests/integration/mcp/`：工具 schema、stdio 服务生命周期与在线探测。

测试使用 `tmp_path`、假模型、假检索器和可控工具隔离运行数据，不修改项目的真实知识索引、会话数据库或 `.env`。
