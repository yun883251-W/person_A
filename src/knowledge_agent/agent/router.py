"""意图路由与 Agent 分支衔接（待开发，仅模块说明与 TODO）。

模块功能：根据已经校验的 IntentDecision 和当前能力，决定本轮执行分支。
输入：意图、显式 Skill 选择、可用工具快照及单轮上下文。
输出：RouteDecision，供 KnowledgeAgent.ask() 执行知识问答、MCP 工具或直接回复。

路由节点不注册为工具，不自行调用识别模型，也不负责启动 MCP 子进程。
第一版只增加 knowledge、mcp、direct 三种分支，不创建 workflows 或 MBTI 答题流程。
知识分支保留检索与 [S1] 引用协议；MCP 分支使用独立提示、工具权限和结果校验。

当前只有说明和 TODO，尚未接入 agent/runner.py。
开发 TODO 01—09；完成全部接入后统一验证 TODO 10—12。
"""


# 第一阶段：路由数据与分支选择。

# TODO 01：定义 RouteKind 和不可变 RouteDecision。
# RouteKind 使用 knowledge、mcp、direct。
# RouteDecision 至少包含 route、intent、allowed_tool_names 和固定 reason_code。
# direct 分支附带规范的 AnswerStatus、回复正文及 stop_reason，不调用检索/工具模型。
# 接口建议 resolve_route(intent, *, skill_id, available_tools) -> RouteDecision。
# 使用独立数据结构，不导入整个 agent/runner.py，避免 intent/router/agent 相互循环依赖。
# 不从用户文本动态导入模块，不把模型返回的字符串当函数名执行。

# TODO 02：实现明确的路由表和能力复核。
# knowledge -> knowledge；mcp_tool -> mcp；问候/范围/澄清/越界 -> direct。
# mcp 路由再次确认工具属于本地白名单并仍然可用，不只信任 IntentDecision。
# 本轮能力不可用时终止当前工具请求并给出说明；不偷偷切换另一个服务完成动作。
# 没有 MCP 配置时，知识路由仍能独立工作；scope 内容只描述实际启用的能力。
# 本期没有 MBTI route；以后加入测试工具和业务状态时再扩展，不预先实现问卷流程。

# TODO 03：处理手动 Skill 与跨能力请求的冲突。
# 现有三个 Skill 继续属于知识路由；知识路由沿用 prepare_skill() 的权限交集。
# 明确 MCP 请求同时指定知识 Skill 时返回 needs_clarification，说明需取消选择或改问法。
# 普通问答不自动加载 Skill，MCP 分支不继承知识 Skill 的提示词和默认工具集合。
# 同一句话同时要求知识解释和独立工具动作、又无法确定处理顺序时先澄清。
# 本期不做自动多任务规划或任意分支嵌套；路由不能在 ask() 内递归调用 Application.ask()。


# 第二阶段：在 agent/runner.py 衔接分支，保持现有生命周期。

# TODO 04：调整 KnowledgeAgent.ask() 的执行顺序。
# 顺序：校验输入/Skill -> 创建单轮状态与 trace -> 读取有限会话上下文 -> identify_intent。
# 再 resolve_route -> 对应分支执行 -> 整理 AnswerResult -> 统一完成 trace 与 _commit_turn。
# direct 分支用现有状态类别返回；knowledge 分支进入原 prepare_question()/模型循环。
# 将路由放在初始 retrieve() 之前，MCP 请求不依赖已有向量索引、不消耗首次检索预算。
# 如果重构 prepare_question()，保留独立调用的兼容入口，不重置已累计的模型/工具计数。
# 沿用 KnowledgeAgentApplication 的操作锁，不新增重入调用或另一套并发会话入口。

# TODO 05：在 agent/runner.py 增加 MCP 分支的模型回合构造。
# 根据 RouteDecision.allowed_tool_names 从 MCP registry 创建本轮工具集合。
# 用独立的 MCP_TOOL_PROMPT（后续放 agent/prompts.py），说明先调用工具，再依据结果回答。
# 只绑定本轮授权的 MCP 工具；不能把 search_knowledge、章节读取和所有 MCP 工具混装。
# 复用现有模型调用适配、AIMessage/ToolMessage 配对和总调用预算。
# 先阅读 continue_model_loop() 的知识证据处理，抽出通用消息循环后再复用。
# MCP 结果不进入 evidence/source_map，不套用知识资料拼装或未检索禁止回答的分支。
# 工具参数不全时允许请求澄清；不能为了凑参数擅自生成用户身份、路径或选择答案。

# TODO 06：为工具分支增加独立的结果收尾规则。
# 知识分支继续使用 finalize_answer() 的 [S1]、文件名和行号校验。
# MCP 分支返回 AnswerResult，sources=()、index_version=None，并标明本轮 route/intent。
# 统计本轮实际调用及成功/失败结果；未调用成功工具不能把动作描述为已经完成。
# 参数不全可返回 needs_clarification；工具失败/超时按固定 stop_reason 返回明确说明。
# 超时可能已经产生外部效果，不声明“肯定没有执行”，不自动重试整个问答。
# 工具响应及历史内容是数据，不能升级为系统指令；结果中出现 [S1] 不生成知识引用。
# 截断/不支持内容类型/部分失败要明确保留状态，不能把缺失数据包装为完整结果。
# 总预算耗尽仍使用 limit_reached；服务失败仍使用 service_error，沿用现有持久化约定。

# TODO 07：保留跨轮路由来源并隔离模型上下文。
# schemas.py 的 AnswerResult 增加可选 intent/route 元数据，或一个可选的路由摘要结构。
# 在 session_store.py 的 result JSON 显式允许这些可选字段；旧记录缺失时按 legacy knowledge 读取。
# 同步 API 回答/历史序列化；新增字段有明确默认值，不改变现有必填字段或 status 枚举。
# 仅增加 result JSON 可选字段时不应盲目修改数据库版本/消息版本，避免旧库无法启动。
# 当前问答的模型历史只包含适合当前路由的完整轮次；切回知识时不得注入外部工具结果当证据。
# 意图识别可读取有限的跨路由问答摘要来理解转题，不能串用不同 session 的内容。
# 会话恢复时只还原已存状态，不重新执行 MCP 工具；刷新知识索引不重启 MCP 服务。

# TODO 08：接入统一执行记录与前端展示。
# agent/trace.py 增加 route 阶段及可公开路由摘要；与 intent 阶段各记一次，耗时用单调时钟。
# MCP 工具执行仍使用 registry 的 tool 事件，记录公开命名空间名称、状态和稳定错误码。
# 服务启动/发现发生在应用启动期间，不伪造成每轮问答中的工具调用或模型调用。
# 同步 trace_to_dict()/trace_from_dict() 的字段白名单和旧 trace 兼容逻辑。
# api.py 的公开模型、web/assets/app.js 的执行详情增加意图/路由说明与新停止原因标签。
# 不把 MCP 参数、环境变量、原始异常或完整工具输出放入默认公开执行详情。
# 本次只增加已有执行详情的必要字段，不实现 MBTI 页面或图片上传界面。

# TODO 09：明确模块调用方向与完成条件。
# agent/runner.py -> agent/intent.py -> agent/router.py；agent/runner.py -> mcp/adapter.py -> mcp/client.py。
# 上述箭头表示组织关系；intent/router 只读取工具元信息，不能自行执行 MCP 工具。
# 运行时由 Application/API/CLI 持有并关闭，单轮 ask() 只借用已就绪的 registry。
# 完成标志是 CLI 和 API 入口都经过同一意图/路由入口，不能只在演示脚本中调用新节点。
# 实现细节集中在对应模块；别复制一份知识 Agent、会话管理器或注册器到 agent/router.py。


# 第三阶段：开发完成后统一验证，本阶段不执行。

# TODO 10：验证知识分支兼容性及分支隔离。
# 覆盖原有建库后问答、手动 Skill、多轮知识追问、引用、资料不足和旧会话恢复。
# 覆盖 MCP 路由在索引缺失时仍可调用工具；知识路线不因 MCP 未安装或未启用而失败。
# 确认普通知识问题不会携带外部工具定义，MCP 结果不会污染本轮知识来源。

# TODO 11：验证工具循环、预算与异常的结果约束。
# 覆盖未调用工具就宣称完成、未知工具、参数缺失、工具错误、重复 ID、超时和超长结果。
# 覆盖意图模型已消耗预算后进入工具循环、同轮多个工具调用和成功失败混合返回。
# 检查 AI/Tool 消息完整配对、计数准确、失败不误存成成功、trace 与实际路径一致。

# TODO 12：最后集中验证 CLI/API/页面全链路。
# 用演示 MCP 工具走完提问、识别、路由、调用、回答、保存、恢复历史、关闭服务。
# 覆盖断连后读取历史、关闭时在途请求、工具分支切回知识分支和多会话隔离。
# 所有验证完成后再标记已接入/可用，不把 TODO 文件存在等同于功能已经实现。
