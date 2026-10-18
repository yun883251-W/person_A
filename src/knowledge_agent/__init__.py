"""Person_A 文档知识库 Agent。

agent/ 组织问答、会话意图、浏览器和规划流程，knowledge/ 管理知识库。
models/ 适配模型，session/ 保存会话与摘要，skills/ 加载和选择技能。
observability/ 记录执行事件，tools/ 注册工具，mcp/ 接入外部工具。
http/ 提供 HTTP 服务，application.py 管理应用与会话生命周期。
API 兼容入口与 CLI 入口仍在包根目录；图片提问骨架归档在 archive/image-qa 分支。
"""
