"""意图识别节点（待开发，仅模块说明与 TODO）。

模块功能：在知识检索和工具调用之前，判断本轮请求应进入哪类处理流程。
输入：用户问题、同一会话的有限历史、用户指定的 Skill、当前可用工具的公开描述。
输出：经过校验的 IntentDecision；不执行检索、MCP 工具或会话写入。

本模块由程序主动调用，不注册到 tools/registry.py，也不作为 MCP 工具暴露。
agent/router.py 根据识别结果决定分支；agent/runner.py 负责实际执行与统一的单轮预算。
第一版保留现有知识问答和手动 Skill 选择，增加通用 MCP 工具请求识别。
MBTI 答题流程和图片问答不在本次开发范围内。

当前没有可执行实现，现有 Agent 仍使用 agent/runner.py 中的 classify_question()。
开发 TODO 01—09；两个节点和 MCP 接入完成后统一验证 TODO 10—12。
完整接入顺序和现有文件改动位置见项目根目录 INTENT_MCP_TODO.md。
"""


# 第一阶段：数据契约与规则识别。

# TODO 01：定义独立的 IntentKind 和不可变 IntentDecision。
# kind 枚举：knowledge、greeting、scope、needs_clarification、out_of_scope、mcp_tool。
# retrieval_query：知识分支使用的检索问题，保留原始用户问题，不直接覆盖它。
# allowed_tool_names：本轮候选 MCP 工具公开名称的 tuple，非 MCP 分支必须为空。
# decision_source：rules、model、fallback，只记录实际决策来源，不编造置信度。
# clarification_question：仅需要澄清时携带的一条明确问题，其余情况为空。
# reason_code：可公开的固定类别，用于说明规则命中、能力不可用或回退原因。
# 不在此对象保存完整历史、密钥、任意命令、工具参数、模型思维过程或可执行函数。
# 本模块不导入 agent/runner.py；用独立的输入结构/Protocol 避免循环依赖。

# TODO 02：迁移 agent/runner.py 现有 classify_question() 的基础规则。
# 迁移问候、功能范围、信息不全、明显越界等规则，保留未知普通问题默认 knowledge。
# 校验 question 是非空字符串；沿用已有空白、末尾标点与大小写处理。
# 现有 classify_question() 保留兼容包装入口供旧调用者使用，不复制两份规则。
# 旧 QuestionKind 不支持 mcp_tool；兼容入口只运行原有分类，不承接新 MCP 路由。
# 不因为问题出现 MCP、工具、Agent 等教学术语就判定用户要求执行工具。
# 示例：“MCP 是什么？”属于知识问题；“调用 add_numbers 计算 2 加 3”是工具请求候选。

# TODO 03：处理有限会话上下文与追问。
# 仅接收当前 session 已提交的完整历史，沿用 max_history_turns 并限制输入字符数。
# 将 agent/runner.py 的 resolve_followup_question() 中检索问题补全与意图输入衔接起来。
# 优先识别当前问题的明确转题，再判断“它”“继续”等指代，避免旧主题覆盖新请求。
# 可补全知识追问的 retrieval_query；缺少可确定的指代对象时返回 needs_clarification。
# 工具追问只能参考同一会话、同一路由的有效上下文，不自动重复上一轮工具动作。
# “选 A”“下一题”目前没有答题状态可依赖，不实现或假装已有 MBTI 业务状态。
# 历史 ToolMessage 只作为历史数据，不作为本轮已执行工具或可引用的知识证据。

# TODO 04：引入当前可用的 MCP 能力快照。
# 接收 mcp.client.McpToolSpec 的公开子集：公开名称、描述及必要的输入字段说明。
# 候选工具必须已由本地 allowed_tools 授权，且所在服务当前可用。
# 工具标识统一使用 mcp__{server}__{tool}；不接受用户或模型提供启动命令/服务路径。
# MCP 未启用、工具不可用或名称无法唯一匹配时，不产生可执行的 mcp_tool 决策。
# 显式要求不可用工具时给出明确能力说明或澄清，不伪造结果、不暗中换工具。
# 普通知识问题继续走 knowledge，不能因为某个 MCP 服务启动失败而整体阻断问答。
# 有 MCP 工具时，scope 的能力说明可以列出实际可用能力；不宣称已支持 MBTI 测试。


# 第二阶段：可选模型识别与入口衔接。

# TODO 05：提供可选的 LLM 辅助识别，保持规则模式可独立工作。
# 在 config.py 设计 IntentSettings / get_intent_settings()，INTENT_MODE=rules|hybrid。
# 默认 rules；hybrid 先处理确定规则，再对无法直接判定的知识/工具请求调用模型。
# 为 hybrid 使用独立分类提示和严格结构化输出；复用现有 DeepSeek 模型适配。
# 识别模型只看到用户输入、有限上下文、可用能力描述，不能绑定或直接执行工具。
# 明确区分“询问某工具如何使用”和“要求实际调用”；混合多个任务时优先澄清边界。
# 不把工具描述或历史中的角色指令当作系统提示；不根据模型自由文本执行跳转。
# 规则模式不增加模型调用。配置缺失时使用明确默认值，不读取或改写实际 .env 密钥。

# TODO 06：校验模型决策，处理失败和歧义。
# 检查输出结构、枚举、字段类型/长度、候选工具名与分支字段之间的约束。
# allowed_tool_names 必须是本次快照的子集，拒绝模型捏造的工具/路径/代码。
# 对 tool intent 需要明确请求依据；不以低可信关键词猜测代替用户行动意图。
# JSON 解析失败、无内容、超时、能力匹配失败时使用固定 reason_code。
# 普通知识请求可回退规则识别；明确工具请求但目标不明时应澄清，不能冒充已执行。
# 不无限重试识别；错误不带完整提示、响应、凭据或内部异常文本。

# TODO 07：把意图调用纳入同一轮模型预算和执行记录。
# 在 agent/runner.py 创建单轮状态/TraceRecorder 之后调用识别，再决定是否首次检索。
# 识别入口建议 identify_intent(question, *, context, available_tools, settings, ...)。
# 模型调用通过由 Agent 提供的计数回调/调用入口执行，调用前检查剩余预算。
# 每次逻辑识别调用计入 state.model_calls；请求层重试按现有模型统计口径记录。
# 不让识别节点和回答循环各自使用一份完整 max_model_calls 配额。
# agent/trace.py 新增 intent 阶段，记录耗时、规则/模型/回退来源、结果类别和固定错误码。
# 只输出可公开的决策摘要，不记录模型隐藏推理；事件字段变化需同步序列化/API/UI。

# TODO 08：保留现有手动 Skill 的语义。
# skill_id=None 仍表示普通问答，不在本次接入中自动挑选 Skill。
# 显式 Skill 仍先验证 ID，现有三个知识 Skill 仅在知识分支使用。
# 用户明确请求 MCP 工具却同时选了知识 Skill 时，由 router 返回需要澄清的结果。
# 不静默丢弃用户选择，不把知识 Skill 提示或知识工具权限扩展到任意 MCP 工具。
# 后续若增加自动 Skill 模式，需另有明确配置/API 字段和实际应用记录，本期不实现。

# TODO 09：通过 agent/router.py 接到当前 ask() 入口，避免重复分类。
# KnowledgeAgent.ask() 调用本模块取得一次 IntentDecision，再交给 router 决定分支。
# prepare_question() 接收已校验的知识决策/检索问题，不重复调用识别或额外消耗模型。
# 保留 prepare_question() 被单独调用时的旧行为，以便逐步迁移现有调用者。
# classify 旧 trace 阶段仍可读；新流程不要将同一次意图识别重复记成两次执行。
# 未接入前，本文件不得仅因被导入就创建模型、加载 MCP 服务或修改全局注册器。


# 第三阶段：开发完成后统一验证，本阶段不执行。

# TODO 10：用假模型和固定能力快照验证识别边界。
# 覆盖问候、范围、普通知识、MCP 教学、明确调用、同名工具歧义、不可用能力和混合任务。
# 覆盖错误 JSON、额外字段、伪造工具名、提示注入、超长输入与模型超时。
# 确认识别过程不会执行工具，不会把暂未实现的 MBTI/图片能力识别成可执行路由。

# TODO 11：验证会话与计数兼容性。
# 覆盖知识追问、工具追问、主动转题、跨会话隔离、历史裁剪、无历史的“继续”。
# 检查 rules 模式新增模型调用数为 0；hybrid 调用计入同轮预算，耗尽后不进入回答模型。
# 检查手动 Skill/普通问答语义与旧历史读取保持兼容，旧入口不会重复运行意图模型。

# TODO 12：最后进行真实模型与 MCP 联调。
# 在用户安排统一联调后，用同一批知识问题和演示工具请求核对实际路由、耗时与调用数。
# 记录误判样例并调整规则/提示，不能仅凭返回合法枚举或一次调用成功宣称识别准确。
