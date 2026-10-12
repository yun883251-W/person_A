"""本地 MCP 示例 Server：待开发，仅用于后续 stdio 联调。

拟提供 add_numbers 工具，输入两个数值，返回确定性的加法结果。
本文件尚未实现服务，config/mcp_servers.example.json 默认禁用它。
这里不是 MBTI 工具，也不保存用户数据或会话状态。
"""

# TODO 01：选择与客户端兼容的官方 MCP Python SDK 服务端入口。
# 使用 stdio 传输；后续实现时采用显式 main 入口，import 模块不能启动服务。

# TODO 02：注册 add_numbers(a, b)。
# 明确参数为有限数值，拒绝 bool 和非有限值；输出结构化 sum 和简短文本。
# 加法不访问文件、网络、环境变量或数据库，便于检查端到端工具调用是否连通。

# TODO 03：规范协议输出与运行说明。
# stdout 仅用于 MCP 协议；需要的诊断写 stderr，并不输出参数或环境中的秘密。
# 在 README 的后续接入说明中写明启动命令、依赖与示例配置启用方法。
# 保持配置中的 remote tool 名称 add_numbers 与实际注册名一致。

# TODO 04：开发完成后统一验证。
# 先验证 SDK 客户端能 initialize/list_tools/call_tool，再验证宿主的白名单、
# namespaced 工具名 mcp__demo__add_numbers、参数校验、调用计数、超时与关闭清理。
# 本次仅创建 TODO，不运行示例、不安装依赖、不启动进程。
