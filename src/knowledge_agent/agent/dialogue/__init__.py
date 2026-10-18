"""会话意图与业务状态的契约、解析和纯状态转换。

功能目标：承接待补充任务，识别当前任务的新建、继续、取消、暂停和恢复，
再把经过校验的决策交给已有知识问答、B 站搜索、MBTI、MCP 和规划流程。

文件职责：
    intent_types.py    原分类与执行路由共用的轻量类型。
    rules.py           确定性规则、历史追问和能力匹配，不调用模型。
    classifier.py      统一 rules/hybrid 策略，至多选择一种语义补充。
    contracts.py       会话决策、字段约束、参数来源和公开投影。
    resolver.py        唯一会话决策入口，先处理生命周期和待补参数。
    transitions.py     纯状态转换和执行结果归并。
    browser_dispatch.py 校验待执行搜索、提交派发保护、调用工具并核对结果。

agent/intent.py 仅保留兼容导出，不再维护独立分类逻辑。
接入记录和未执行验收项见 project_docs/DIALOGUE_INTENT_TODO.md。
公开调用入口由 ``contracts``、``resolver`` 与 ``transitions`` 各模块显式提供；
runner 只消费校验后的决定与状态提案，不把本包注册成工具或 Skill。

边界：不注册成工具或 Skill，不在导入时创建模型、读数据库或启动 MCP。
一期每会话只有一个前台业务焦点和一个待补充任务；已有 MBTI 快照继续独立保存。
现已由 runner、SQLite 会话存储、应用生命周期和公开接口接入。
一期不实现运行中调用中断、断线恢复、请求标识系统或部分计划的自动恢复执行。
"""
