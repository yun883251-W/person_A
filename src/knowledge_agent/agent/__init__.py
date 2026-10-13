"""Agent 执行流程、意图识别、路由、提示词和执行记录。

运行入口位于 runner.py；intent/router 选择单步分支，planner/plan_executor 处理多步骤任务。
state 定义阶段数据，preparation 准备检索与提示词，model_loop 执行模型与工具，answers 校验引用。
本包不在导入时创建模型、检索器或会话服务。
"""
