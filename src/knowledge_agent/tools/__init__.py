"""Agent 工具模块（待开发）。

模块职责：提供工具注册、统一执行及知识库工具的公开入口。
registry.py 管理工具定义与调用；knowledge.py 封装知识库搜索和小节读取。

当前只有文件骨架与 TODO，agent.py 中的工具逻辑仍然是实际运行入口。
开发顺序：registry.py → knowledge.py → skill_loader.py → Agent/API/页面接入。
本阶段不迁移旧代码、不初始化模型或数据库，验证留到开发完成后统一进行。
"""


# TODO 01：在 registry.py 和 knowledge.py 实现后，确定工具包的公开接口。
# 导出注册器、工具定义结构和知识库工具工厂，避免对外暴露内部辅助函数。

# TODO 02：整理依赖方向，避免 tools 与 agent 相互导入。
# 工具通过参数接收检索器、每轮上下文或回调，不从 agent.py 导入运行实例。
# 导入工具包时不读取索引、不加载技能、不创建模型或会话连接。
