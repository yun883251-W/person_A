"""会话任务数据契约：仅说明和详细 TODO，尚未定义运行时接口。

功能：分离模型提供的意图候选、程序校验后的决策、持久化业务状态和公开摘要。
输入：当前用户消息、程序读取的会话状态、实际可用能力及现有 MBTI 状态。
输出目标：IntentProposal、DialogueDecision、SessionBusinessState 和状态更新提案。
约束：不执行识别、检索、工具、状态写入；模型输出不得自行取得执行权限。

建议以不可变数据结构和显式校验函数实现。以下类型、函数名是开发目标，尚不可调用。
现有 IntentDecision、RouteSummary、AnswerResult 的兼容改动在接入文档中实施。
"""

# TODO C01：定义相互独立的意图、业务类型和执行模式。
# DialogueIntent：new_business、continue_business、cancel_business、pause_business、
# resume_business、chitchat、unclear、unsupported。
# BusinessType：knowledge、bilibili_search、mbti、mcp_tool；未知业务使用空值。
# ExecutionMode：single、plan；混合计划的顶层 business_type 可为空，各步骤另行分类。
# 明确字段组合约束：plan 不靠 business_type=complex 表达，unsupported 不能进入执行。
# 这些是新契约，不直接替换旧 IntentKind 或修改数据库已有路由枚举。

# TODO C02：定义仅允许模型提供的 IntentProposal。
# 固定字段为 intent、business_type、execution_mode、slots、missing_slots。
# 明确字符串长度、对象深度、字段数量和各枚举范围；拒绝未知字段、重复键和错误类型。
# 不接收 session_id、revision、authorized、工具白名单或可信参数来源等程序字段。
# missing_slots 只作为模型建议，程序必须依照当前业务所需字段重新计算。
# 模型格式错误应返回固定校验码，不得自动当作 knowledge 或可执行工具请求。

# TODO C03：为每种业务定义参数白名单和类型校验。
# B 站搜索仅接受 query，复用 browser_mcp.contracts.validate_search_arguments。
# MBTI 动作和答案复用 assessment 的合法动作、题号、选项约束，不复制题库到 slots。
# knowledge 的追问背景不能作为工具参数；通用 MCP 只使用实际已注册工具的 schema。
# 一期只对已明确实现参数承接规则的业务开放续办；未知工具的任意缺参不自动补齐执行。
# 检查 missing_slots 与已确认参数不矛盾，全部必填参数通过校验才算 ready。

# TODO C04：定义程序维护的参数来源 SlotBinding。
# 至少能表达参数名、确认值、原始输入、已有来源轮次以及来源种类。
# 区分 current_user_input、pending_user_request 和 same_request_clause。
# 来源字段由宿主基于真实输入建立，不采用模型自报来源、聊天摘要或工具回答。
# B 站补主题需要同时保留先前搜索请求与本轮主题；本轮用户原话必须原样保留。
# 复用 topic_references 的同请求绑定时保留来源子句含义，不冒充跨轮授权。

# TODO C05：定义通过校验的 DialogueDecision。
# 保存合法意图、业务、执行模式、确认参数、澄清问题、decision_source 和 reason_code。
# 可携带兼容现有分支的 IntentDecision、原始计划请求及程序构造的参数来源绑定。
# 明确区分“继续知识讨论”和“继续外部操作”；前者不得授予任何工具权限。
# 工具允许集合取自现有配置、能力快照和原始请求，不能直接使用模型输出的名字。
# 决策引用预期的状态 revision，执行前仍复核该状态没有被替换或消费。

# TODO C06：定义 PendingBusiness（等待用户补充的任务）。
# 保存原始请求、业务类型、确认参数、参数来源、缺失参数、追问和 clarification_count。
# 绑定所属会话与实际来源轮次；不把恢复时最近一次模型回答当成原始请求。
# 同一会话最多一个 pending；只有 awaiting_input 状态能接收补充。
# 明确新任务替代旧 pending 后，旧待办不得被裸“继续”重新激活。
# 已执行计划摘要不是 pending，不保存可自动恢复执行的 plan 或原生 tool_call。

# TODO C07：定义 SessionBusinessState 与状态枚举。
# 建议字段：schema_version、revision、focus、pending、updated_at。
# focus 描述 business_type、phase 和业务关联；MBTI 仅关联已有 test_id。
# 阶段至少区分 idle、awaiting_input、ready、active、paused、dispatching、
# completed、cancelled、execution_unknown；只允许符合业务的阶段组合。
# knowledge 可保留讨论焦点，completed 搜索不可因为有焦点而自动再执行。
# 题库、测试答案、模型内部消息和检索来源仍归原有模块，不复制到本状态。

# TODO C08：定义纯状态更新提案 StateTransition。
# 包含 expected_revision、候选 next_state、固定原因码和后续动作描述。
# 明确普通“随问答提交”与“工具派发前单独消费 pending”的提交时机。
# 提案不写数据库、不递增全局对象、不调用工具；存储成功后再发布新内存状态。
# 对外操作结果是否确定由执行器提供，不能让分类模型把状态直接改成 completed。
# 时间等不确定输入由调用方传入，便于后续统一验证纯转换逻辑。

# TODO C09：实现契约间的不变量校验。
# focus、pending、business_type、MBTI test_id 和 revision 必须相互一致。
# 拒绝跨会话来源、已消费 pending、未来状态版本、非法阶段跳转和未校验参数。
# ready 必须无缺参；dispatching/terminal 阶段不得残留可自动消费的 pending。
# 程序计算不出唯一取消/恢复目标时返回需明确目标，不能任选一个历史任务。
# 为旧会话定义显式空状态；损坏或未知版本状态不能静默重置为可执行状态。

# TODO C10：定义持久化与公开投影两个独立接口。
# validate_business_state / state_to_payload / state_from_payload 严格限制版本和字段。
# public_business_state 只公开业务、阶段、缺失字段、追问、允许展示的参数和动作。
# 原始授权文本、内部消息、问卷答案、运行时对象和模型原始 JSON 不发到页面。
# 历史轮次摘要与会话当前状态分开：翻历史不改变焦点，也不重新执行工具。
# 新字段对旧 AnswerResult/API 消费方可选；不要扩大旧 result_json 的接受范围而免校验。

# TODO C11：列清固定原因码及其结果分类。
# 覆盖 pending_created、slot_filled、pending_cancelled、pending_replaced、
# no_active_business、ambiguous_business_target、state_conflict、execution_unknown。
# 单列模型超时、无预算、非法 JSON、参数校验失败和存储失败；不计为用户澄清失败。
# 延续 AnswerStatus 的现有枚举；普通“已取消待办”是成功的状态操作而不是模型错误。
# 未理解、能力不支持、工具不可用分别表达；不要合并成“我没听懂”。

# TODO C12：整理接口依赖与交付边界，供其他 TODO 引用。
# contracts 不依赖 runner、application 或 HTTP，避免契约反向加载运行时。
# 建议对外名称清单先在本文件实现，resolver/transitions 仅在实现后导入。
# 文档明确字段是否可为空、默认值与不可变复制规则，所有调用方使用同一份契约。
# 开发完成后静态核对 C01—C11 与接入清单；验收场景见文档 V01—V10。
# 本阶段不要创建测试文件或执行导入、编译、测试、服务和真实模型验证。
