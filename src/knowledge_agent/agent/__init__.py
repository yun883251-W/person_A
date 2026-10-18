"""Agent 执行流程、意图识别、路由、提示词和执行记录。

运行入口位于 runner.py；dialogue/ 识别会话意图，router.py 选择执行分支。
planning/ 处理多步骤任务，browser/ 处理浏览器请求，answering/ 校验回答和引用。
state 定义阶段数据，preparation 准备检索与提示词，model_loop 执行模型与工具。
技能选择、摘要和执行事件分别由 knowledge_agent.skills、session、observability 维护。
本包不在导入时创建模型、检索器或会话服务。
"""
