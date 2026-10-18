# 运行启动脚本

这里保存实际运行的本地 MCP 服务启动入口：

- `browser_mcp_server.py`：启动 `src/browser_mcp/`。
- `mbti_mcp_server.py`：启动 `src/mbti_mcp/`。

默认配置和独立配置模板已指向本目录。宿主使用自身 Python 解释器启动子进程；脚本补充项目 `src/` 路径，无需手动设置 `PYTHONPATH`。

演示服务仍位于 `examples/mcp_demo_server.py`。如果自行维护了额外 MCP 配置，需要把其中旧的 `examples/browser_mcp_server.py`、`examples/mbti_mcp_server.py` 改为对应 `scripts/` 路径。

本轮仅迁移文件和修改配置，没有启动服务。
