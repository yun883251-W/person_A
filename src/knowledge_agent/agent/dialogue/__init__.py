"""会话意图与业务状态的契约、解析和纯状态转换。

功能目标：承接待补充任务，识别当前任务的新建、继续、取消、暂停和恢复，
再把经过校验的决策交给已有知识问答、B 站搜索、MBTI、MCP 和规划流程。

文件职责：
    contracts.py   数据结构、字段约束、来源和公开投影的 TODO C01—C12。
    transitions.py 纯状态转换和执行结果归并的 TODO T01—T12。
    resolver.py    规则、可选模型分类及原有意图兼容的 TODO R01—R12。

开发顺序：先 contracts，再 transitions，再 resolver，最后完成现有模块接入。
接入项 I01—I12、集中验收项 V01—V10 见 project_docs/DIALOGUE_INTENT_TODO.md。
公开调用入口由 ``contracts``、``resolver`` 与 ``transitions`` 各模块显式提供；
runner 只消费校验后的决定与状态提案，不把本包注册成工具或 Skill。

边界：不注册成工具或 Skill，不在导入时创建模型、读数据库或启动 MCP。
一期每会话只有一个前台业务焦点和一个待补充任务；已有 MBTI 快照继续独立保存。
现已由 runner、SQLite 会话存储、应用生命周期和公开接口接入。
一期不实现运行中调用中断、断线恢复、请求标识系统或部分计划的自动恢复执行。
"""
