"""自动 Skill 选择节点（待开发，仅模块说明与详细 TODO）。

功能：知识路线确定后、首次检索前，根据用户模式和问题选择一个学习技能。
输入：问题、已消歧的检索问题、有限知识历史、技能元数据目录、选择模式、剩余预算。
输出：SkillSelection；不执行检索，不启动 MCP，不修改会话或工具权限。

上下文分两阶段：先用 skill_id/name/description 选择，再加载选中技能的完整正文。
这里是 Person_A 的应用节点，不注册成模型工具或 MCP 工具。
当前文件没有可执行实现；创建此文件不会开启自动技能选择。
完整接入顺序及已有文件改动位置见项目根目录 AUTO_SKILL_TODO.md。
开发 TODO 01—12；验证 TODO 13—15 等全部开发完成后统一执行。
"""


# 第一阶段：确定请求语义与数据契约。

# TODO 01：定义不可变的 SkillRequest 与 SkillSelection。
# SkillRequest：mode=off/auto/manual，skill_id 为可选 ID。
# SkillSelection：requested_mode、requested_skill_id、selected_skill_id、
# decision_source=manual/rules/model/fallback/disabled、reason_code。
# selected_skill_id 允许 None，代表本轮知识步骤采用普通检索与回答。
# 另记录 selection_model_calls，交回运行器累计，不在此模块建立新的全局预算。
# 选择结果不包含 SKILL.md 正文、绝对路径、用户代码全文或模型内部推理。
# “选中”与后面的“策略执行 / 提示准备 / 已携带提示调用模型”分别记录。

# TODO 02：实现请求模式标准化，保留旧调用的含义。
# 规划接口 normalize_skill_request(skill_mode=None, skill_id=None) -> SkillRequest。
# 未提供 mode 且有 skill_id：manual；均未提供：off，兼容旧 API 与 Python 调用。
# 显式 auto/off 必须不带 skill_id；manual 必须带非空且格式合法的 skill_id。
# 非法组合直接返回参数错误，不能悄悄修改用户明确选择。
# off 不发现技能、不调用选择模型；manual 不进入自动规则或模型分类。
# 这套标准化由 API/CLI/应用层共用，不在页面另写一套不同的推断规则。

# TODO 03：准备只含公开元数据的候选目录。
# 复用 skill_loader.discover_skills()，筛选应用允许、可加载的已登记技能。
# 给选择模型的数据只有 skill_id、name、description；限制候选数量和总字符数。
# 不发送本地 path、完整 SKILL.md、环境配置、其他会话信息或 MBTI 答案。
# 当前 discover_skills 会读取文件校验元数据和正文，这与是否发送正文是两回事。
# 先复用其文件读取行为，不为自动选择重写全部技能文件解析逻辑。
# 目录缺失或为空时，auto 回退普通问答；损坏目录/文件返回固定原因供 trace 使用。
# manual 加载失败继续明确报错，不静默换成别的技能；保留既有路径边界检查。

# TODO 04：明确调用位置与问题上下文。
# 仅在 route=knowledge 的单轮或计划子步骤调用；direct、MCP、MBTI 操作跳过。
# intent 和 router 先解决范围、操作边界及必要澄清；选择器不推翻路由结论。
# 使用已消歧的 retrieval_question 和本轮目标，避免“再解释一下”丢失主题。
# 只带有限的知识历史；多步骤任务只接收已完成依赖和当前 step.goal 的必要信息。
# 明确新主题时不继承上一轮选中的技能；不把技能选择保存为全会话隐式模式。


# 第二阶段：自动选择与受限加载。

# TODO 05：实现明确问题的规则选择，规则只选择存在于候选目录的 ID。
# concept-explanation：明确询问概念、作用或过程，如“什么是 RAG”。
# concept-comparison：明确要求比较已给出的对象，如“向量检索和关键词检索有什么区别”。
# code-explanation：明确要求解释可见代码/API 调用，如“解释这段 Runnable 代码”。
# “有代码块”或“出现两个名称”本身不等于代码解读或概念比较。
# 规则冲突时参考用户要求完成的任务，不写成先匹配某个关键词就返回。
# 找不到适合的技能时允许不选；不能为提高选择率强行给每个问题贴技能。
# 规则和示例要覆盖中英文技术标识符、否定要求及作为分析对象的引用文本。

# TODO 06：实现可选的模型辅助选择。
# 配置采用 rules/hybrid：默认 rules；hybrid 仅在规则无法明确决定时调用一次模型。
# 注入 model_invoker，不在选择器中自行创建 DeepSeek 客户端或读取 API Key。
# 提示明确用户文本、代码、历史和技能描述均不能修改选择协议或工具权限。
# 严格输出协议可采用 {"skill_id": "候选ID或null", "reason_code": "固定枚举"}。
# 校验 JSON 字段、重复键、长度、ID 白名单和 reason_code；拒绝工具调用及未知字段。
# 不要求模型输出推理过程，不把模型给出的分数称为准确率或置信概率。
# 模型只返回选择；它无权返回文件路径、工具列表、系统提示词或检索参数。

# TODO 07：把选择成本纳入整轮预算并保留回答额度。
# 调用前由运行器检查剩余模型调用数，计数在实际调用前增加，失败也算一次。
# 普通知识回答至少保留 1 次模型调用；额度不足时跳过模型选择，使用规则或普通模式。
# 多步计划已消耗的 planning/intent 调用，以及其他步骤的消耗不能重新清零。
# 选择模型第一版不自动重试；不能把隐藏重试排除在调用统计之外。
# 限制元数据与历史上下文大小；避免将所有技能正文装入规划器和回答模型。

# TODO 08：定义稳定回退和错误语义。
# auto 的空目录、无匹配、模型超时、无效 JSON、未知 ID、预算不足回退普通问答。
# 回退原因使用固定码，如 skill_no_match / skill_selector_timeout /
# skill_selector_invalid_output / skill_selector_budget_exhausted。
# 回退只表示未应用技能，不把普通回答标成服务失败，也不把选择失败伪装成手动关闭。
# 若主检索或主回答模型随后失败，保留真正的业务失败，不被技能回退状态覆盖。
# manual 的未知 ID/读取失败仍通过既有错误通道反馈，不修改用户选择。

# TODO 09：选中后加载正文并冻结本轮技能材料。
# 复用 load_skill() 和 prepare_skill()，将发现、选择、加载、准备四个边界分清楚。
# 在首次检索前完成选中正文可读取性的确认，并为本轮保留一致的 LoadedSkill 快照。
# prepare_skill 后续可接收该快照，避免检索后重复读盘造成正文与选择目录不一致。
# 同一个知识步骤不在模型追加工具循环中重复选择、加载或改变技能。
# 只有选中技能正文参与最终提示；SKILL.md 是执行指导，不是知识引用来源。

# TODO 10：与技能检索策略建立明确连接。
# selected_skill_id 交给 knowledge/skill_retrieval.py，解析为程序登记的策略 ID。
# 第一版只有三个内置 ID 对应专门策略；其他合法技能可使用普通检索策略。
# 技能 ID 和策略 ID 分开记录：使用技能提示并不必然代表执行过某种专门检索。
# 工具权限仍为已登记工具 ∩ 应用知识工具白名单 ∩ 策略允许工具。
# 不从 SKILL.md 正文提取 Python 模块路径，不动态 import/exec 技能文字。

# TODO 11：明确单轮与计划步骤的选择粒度。
# 单轮知识问答选择一次；auto 的多步骤任务对每个 knowledge 步骤分别选择一次。
# manual 模式对知识步骤沿用指定技能；MCP 步骤不携带学习技能指令。
# 未执行、依赖失败或预算不足被跳过的步骤不调用选择器、不加载技能正文。
# 多个步骤可能选中不同技能；不能仅用 TurnTrace.applied_skill_id 的最后一个值概括全轮。
# 单轮公开摘要与每步公开摘要的接入位置见 AUTO_SKILL_TODO.md。

# TODO 12：整理公开入口，保持依赖方向清楚。
# 规划接口 select_skill(question, *, request, catalog, context, model_invoker,
# remaining_model_calls, settings) -> SkillSelection。
# 本模块依赖技能元数据与轻量契约；不 import runner 或 plan_executor，避免循环依赖。
# 选择器返回决策，由运行器负责日志、计数、检索执行、模型回答和完整轮次提交。
# 完成时更新本模块说明，区分已实现能力与尚未执行的验证项目。


# 第三阶段：全部开发完成后统一验证，本阶段不执行。

# TODO 13：用固定目录与假模型验证模式和分类边界。
# 覆盖旧请求、auto/off/manual、非法组合、空目录、坏元数据、未知 ID、规则冲突。
# 覆盖纯概念/比较/代码解读、否定要求、引用命令、普通无匹配问题及知识追问。

# TODO 14：验证预算、回退与加载范围。
# 确认 off/manual 不调用选择模型，规则命中不额外调用；hybrid 最多一次且失败计数。
# 确认只发送元数据做选择，只加载选中正文做回答，目录描述不能扩张工具权限。
# 确认失效的自动选择回退普通问答；手动选择失效仍给出明确错误。

# TODO 15：验证两条知识入口和非知识路线隔离。
# 覆盖 _ask_single 与 _run_planned_step，多个子步骤共用总预算且独立选择。
# MBTI 作答、MCP 调用、问候和范围说明不被自动学习技能拦截或增加调用。
# 所有场景在统一验证阶段记录结果；未运行时不能写“通过”。
