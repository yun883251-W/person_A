"""按 Skill 生成并执行知识检索策略（待开发，仅说明与详细 TODO）。

功能：把已选中的学习技能转为有界查询步骤，使技能影响首次检索和材料组织。
输入：问题、明确的分析对象、selected_skill_id、剩余工具/追加检索/上下文预算。
输出：SkillRetrievalPlan、执行结果与覆盖情况；不生成最终知识答案。

本模块复用 retrieval.py 的混合检索，以及 tools/knowledge.py 的只读执行器。
不重新向量化、不创建新数据库、不修改全局 retriever 配置、不运行用户代码。
当前没有可执行实现；所有编号都是待开发项，验证留到最后统一执行。
接入映射见项目根目录 AUTO_SKILL_TODO.md；先完成 agent/skill_selector.py 的契约。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RetrievalOperation = Literal["search", "read_section"]
RetrievalStepStatus = Literal["executed", "skipped", "error"]


@dataclass(frozen=True, slots=True)
class RetrievalStep:
    """One planned knowledge-tool operation."""

    step_id: str
    operation: RetrievalOperation
    target: str
    required: bool
    query: str | None = None
    anchor_id: str | None = None


@dataclass(frozen=True, slots=True)
class SkillRetrievalPlan:
    """Immutable retrieval strategy selected for one knowledge step."""

    strategy_id: str
    selected_skill_id: str | None
    steps: tuple[RetrievalStep, ...]
    coverage_targets: tuple[str, ...]
    reason_code: str


@dataclass(frozen=True, slots=True)
class RetrievalStepResult:
    """Execution result for a single planned operation."""

    step_id: str
    status: RetrievalStepStatus
    returned_chunk_ids: tuple[str, ...] = ()
    retained_chunk_ids: tuple[str, ...] = ()
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SkillRetrievalOutcome:
    """Aggregate execution and coverage result for a retrieval plan."""

    step_results: tuple[RetrievalStepResult, ...]
    executed: int
    skipped: int
    errors: int
    missing_targets: tuple[str, ...] = ()
    budget_limited: bool = False


# 第一阶段：契约、策略注册与分析对象。

# TODO 02：建立固定策略映射和普通模式。
# None -> default；concept-explanation -> concept；concept-comparison -> comparison；
# code-explanation -> code；其他合法技能 -> default，并保留其技能提示。
# default 复用原来的单次混合检索；off 和自动回退均走该策略。
# 映射是程序维护的固定表，不从用户输入、SKILL.md 或模型输出动态加载执行代码。
# 第一版不增加 reranker、向量库服务或全局 top_k 调整，保持已有索引兼容性。

# TODO 03：从问题和可见上下文提取检索对象。
# 使用消歧后的 retrieval_question，记录对象原文及来自哪一段用户输入。
# 概念比较提取两项明确对象；代码解读提取可见 import/类名/函数名/调用链。
# 仅做有界文本分析，必要时可用 ast.parse；不 eval/exec、不导入用户代码中的模块。
# 不根据本地文件路径推测代码内容，不把代码字符串内的指令当作检索任务。
# 对象不明确时返回固定澄清或普通查询策略，不自行补造比较对象或 API 名称。
# 第一版不额外调用查询生成模型；若今后添加，必须经统一模型预算入口。


# 第二阶段：三种技能产生不同的查询步骤。

# TODO 04：实现概念讲解策略。
# 先构造“明确概念 + 用户关注点”的查询，关注点可为定义、用途或执行过程。
# 第二条用途/示例查询仅作为可选补充，有明确缺口和剩余额度时才执行。
# 不机械追加无关例子；用户只问一个细节时围绕该细节检索。
# 覆盖标签用于组织材料，不把“有命中”直接当作资料充分或概念已被解释正确。

# TODO 05：实现概念比较策略。
# 提取 A、B 后，首选分别查询 A 与 B 的相同关注维度，而不是只检索整个长句。
# 保留目标归属：A 命中了哪些片段、B 命中了哪些片段、共享片段支持哪个目标。
# 第一版最多两个对象；三个以上对象应明确缩小范围或回退普通有界查询。
# 仅剩一次搜索额度时用包含 A、B 与比较关注点的联合查询，标记 budget_limited。
# 不能把一次联合查询描述为已分别检索两侧，也不能把缺失的一侧推断为“不支持”。
# 无法识别对象时先澄清；确定了对象但证据只支持一侧时可返回有依据的部分。

# TODO 06：实现代码解读策略。
# 从可见代码提取少量关键 API/类/函数，结合用户疑问生成精确技术标识符查询。
# 原问题保留为任务输入，但避免把大段完整代码直接拼进每条向量查询。
# 至多生成两条有区别的查询；内部自定义变量名通常不应当作框架 API 检索。
# 只有路径而无代码时，沿用现有“请提供代码片段”的澄清，不新增任意文件读取。
# 代码来自文档片段时保留该片段出处；用户粘贴代码本身不是知识库引用来源。

# TODO 07：定义有界章节补读条件。
# 只有已返回片段明显缺少相邻说明/代码闭合部分，且预算允许时才考虑一次补读。
# 复用 read_knowledge_section，只接受本次知识步骤登记过的 chunk_id。
# 不从历史计划步骤复制 anchor ID，不猜路径、章节号或任意片段 ID。
# 没有可靠的缺口判据时交给已有模型工具循环，不能给每个命中片段都自动补读。
# 章节补读和追加查询分别计数；补读不消耗追加搜索次数，但消耗总工具和上下文额度。


# 第三阶段：真实执行、聚合与共享预算。

# TODO 08：通过已有工具执行器顺序运行策略步骤。
# 构造或注入 tools/knowledge.create_search_executor / create_section_executor。
# 新模块定义轻量 Protocol 或接收回调，不 import agent.runner.TurnState。
# 运行器持有真实 TurnState，把必要的证据表、状态和执行回调传入。
# 多查询顺序执行以确定预算消费与索引版本，不额外开线程同时修改同一 evidence 表。
# 不为了策略执行伪造 AIMessage.tool_calls；程序检索继续记录 initiator=program。

# TODO 09：统一计数，防止多查询绕过限制或重复扣费。
# 每次实际 search/read_section 在执行前检查并消耗一次 tool_calls；失败也计数。
# 每个知识步骤的首个 search 为首次检索；后续所有 search 均消耗 followup_searches。
# 后续 search 包括策略生成的第二条查询和回答模型追加查询，共用剩余次数。
# 比较策略两次搜索不是“一次批处理工具”；不能额外新增未计数的底层 retrieve 调用。
# 确定唯一计数责任方：若用 registry 已计数就不在策略重复计数，反之由统一回调负责。
# 计划里各步骤接收 plan_executor.StepBudget 的剩余额度，不能给每步恢复默认 6/2。
# 预算耗尽时标记未执行步骤；没有任何可用证据时不能宣称检索成功。

# TODO 10：聚合证据与限制上下文。
# 每次结果保留原始 RetrievalResult，按 (index_version, chunk_id) 去重并保留命中目标。
# 分别检索的 fused_score 只在各自查询内有意义，不直接跨查询比较成统一概率。
# 可采用按目标轮流取片段的确定性顺序，避免 A 的长片段挤掉 B 的所有材料。
# 控制片段数和实际序列化后的上下文字符数；依赖步骤上下文已占用的额度不重新使用。
# 重复片段不重复占用最终保留材料；实际送入模型的工具文本仍按现有协议统计。
# source_map 只登记实际提供给模型的证据；被裁掉的片段不能留下可引用的来源标记。
# 不切断内容后仍冒用原来的完整行号；若裁剪片段必须同步维护准确元数据。

# TODO 11：检查索引一致性与多查询错误聚合。
# 同一知识步骤所有查询和补读使用同一索引版本；版本变化不能拼成一份来源表。
# 当前 create_search_executor 会逐次覆盖 retrieval_status；不能用最后一次状态代表全部。
# 任一次故障、空结果、成功有结果和预算跳过分别保留，最后计算整体 outcome。
# 部分查询失败但有有效证据时保留可用部分并说明缺口；全部服务失败应为 service_error。
# 成功查询均为空才属于资料不足；超时/模型失败/索引缺失不能被描述成“没有资料”。

# TODO 12：接回现有证据构造与回答流程。
# 替换 prepare_question 中固定的一次 search，而不是在原始搜索前后再重复搜索。
# 聚合后继续调用现有 prepare_evidence_context/source_map 构造和 finalize_answer。
# 运行器把覆盖缺口和裁剪情况作为数据交给回答模型，保持 [S1] 来源协议。
# 检索策略不能自己生成最终事实答案、改系统角色或绕过资料不足处理。
# 选中的技能提示在材料准备后供模型使用；已完成的策略步骤需告知模型以免重复检索。

# TODO 13：保持后续工具循环与实际覆盖结果一致。
# 技能策略结束后，模型仍可在剩余预算内 search/read_section，不另设无限补查循环。
# 追加结果继续更新当前步骤的证据和目标覆盖摘要，但不改变已选技能。
# 记录证据缺口，不以关键词命中或片段数量保证语义充分；效果仍需之后的评估确认。
# 策略只提供查询差异与材料组织，不把三个技能的输出质量预先宣称为提升。

# TODO 14：公开小型摘要，保持查询材料与工具权限边界。
# 摘要包含 strategy_id、目标标签、计划/已执行搜索数、补读数、保留片段数、
# skipped_reason/coverage_missing；不返回完整检索文本、用户代码全文或模型提示。
# 不在持久化摘要中复制 evidence/source_map；来源仍用已有 AnswerResult.sources。
# 策略参数只限制已有知识工具，不能因选择 Skill 自动开启网络、MCP 或代码运行。
# 完成时更新三个 skills/*/SKILL.md：先检查已提供的策略结果，再针对缺口补查。


# 第四阶段：全部开发与接入完成后统一验证，本阶段不执行。

# TODO 15：用假检索器验证实际查询序列与计数。
# 分别覆盖 default、concept、comparison、code，记录真实查询/章节读取的参数和次数。
# 检查单查询预算、比较两侧、代码 API 标识符、重复查询、空查询及代码缺失的处理。
# 不把“生成了检索计划”当作“已执行检索”；失败调用与跳过调用的统计要不同。

# TODO 16：验证溯源、上下文和错误聚合。
# 覆盖重复片段、多查询分数不可比、版本变化、A 成功 B 失败、全部为空与全部故障。
# 检查被裁掉内容不会留有效 [S1]，原文路径与行号不变；章节锚点必须来自当前步骤。
# 三步计划共享工具、追加搜索及上下文预算，不影响其他会话或 retriever 的全局设置。

# TODO 17：最后执行统一的实际效果对照。
# 对同一组问题分别用 off、manual、auto，比对选择、查询、证据、引用、调用数及耗时。
# 使用 concept-explanation / concept-comparison / code-explanation 的代表性问题。
# 通过既有 evaluation 模块的后续接口或人工记录汇总，不在本次 TODO 阶段执行。
