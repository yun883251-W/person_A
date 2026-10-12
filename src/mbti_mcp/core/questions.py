"""
MBTI 人格测试 — 问卷数据模块

本模块提供标准化的 MBTI 测试题本, 包含:
  - 48 道题目, 每维度 12 道, 覆盖 E/I、S/N、T/F、J/P 四个维度
  - 5 点李克特量表选项: 1=完全不符合 ~ 5=完全符合
  - 每题标注所属维度(dimension)和方向(direction), 用于后续计分

题目设计说明:
  - 每个维度分为"正向"(指向外向/实感/思考/判断)和"反向"(指向内向/直觉/情感/感知)两类题目
  - positive_direction 表示该题目得分高指向哪个维度极(如 'E' 表示得高分者为外向)
  - EI 维度: E(外向) vs I(内向)
  - SN 维度: S(实感) vs N(直觉)
  - TF 维度: T(思考) vs F(情感)
  - JP 维度: J(判断) vs P(感知)

文件结构:
  1. MBTI_QUESTIONS — 主题目列表(48题)
  2. OPTIONS — 5点李克特量表选项文本
  3. get_questions() — 对外查询接口, 支持按维度分组
"""

# 5 点李克特量表选项
OPTIONS = [
    {'value': 1, 'label': '完全不符合'},
    {'value': 2, 'label': '不太符合'},
    {'value': 3, 'label': '中立/不确定'},
    {'value': 4, 'label': '比较符合'},
    {'value': 5, 'label': '完全符合'},
]

# ═══════════════ MBTI 题目库(48题) ═══════════════
# 题目结构:
#   id        : 题目编号(0-47)
#   text      : 题目文本
#   dimension : 所属维度(EI / SN / TF / JP)
#   direction : 高分指向的维度极(E/I/S/N/T/F/J/P)

MBTI_QUESTIONS = [
    # ──────────── EI 维度(12题) ────────────
    # E(外向): 从外部世界和人际互动中获取能量
    # I(内向): 从独处和内省中获取能量
    {'id': 0, 'dimension': 'EI', 'direction': 'E',
     'text': '在社交聚会中, 我通常感到精力充沛, 而不是疲惫'},
    {'id': 1, 'dimension': 'EI', 'direction': 'I',
     'text': '我需要很多独处时间来思考和恢复精力'},
    {'id': 2, 'dimension': 'EI', 'direction': 'E',
     'text': '我喜欢结识新朋友, 并享受与陌生人交谈的过程'},
    {'id': 3, 'dimension': 'EI', 'direction': 'I',
     'text': '相对于一群人, 我更倾向于与一两个亲密朋友深度交流'},
    {'id': 4, 'dimension': 'EI', 'direction': 'E',
     'text': '我通常先行动后思考, 在行动中逐渐理清思路'},
    {'id': 5, 'dimension': 'EI', 'direction': 'I',
     'text': '在做决定之前, 我需要先独自思考和消化信息'},
    {'id': 6, 'dimension': 'EI', 'direction': 'E',
     'text': '我可以轻松地与不同类型的人打成一片'},
    {'id': 7, 'dimension': 'EI', 'direction': 'I',
     'text': '在群体讨论中, 我倾向于先倾听再发言'},
    {'id': 8, 'dimension': 'EI', 'direction': 'E',
     'text': '在工作或学习中, 与他人合作比独自工作更让我有动力'},
    {'id': 9, 'dimension': 'EI', 'direction': 'I',
     'text': '长时间社交后我需要独处来"充电"'},
    {'id': 10, 'dimension': 'EI', 'direction': 'E',
     'text': '我倾向于通过讨论来理清自己的想法'},
    {'id': 11, 'dimension': 'EI', 'direction': 'I',
     'text': '比起接电话或当面说话, 我更喜欢用文字消息沟通'},

    # ──────────── SN 维度(12题) ────────────
    # S(实感): 关注具体事实、细节和现实经验
    # N(直觉): 关注抽象概念、可能性和未来愿景
    {'id': 12, 'dimension': 'SN', 'direction': 'S',
     'text': '我更相信亲身经历和具体事实, 而不是抽象理论'},
    {'id': 13, 'dimension': 'SN', 'direction': 'N',
     'text': '我经常沉浸在对未来的想象和各种可能性的思考中'},
    {'id': 14, 'dimension': 'SN', 'direction': 'S',
     'text': '我注重细节和具体步骤, 喜欢按既定的方法做事'},
    {'id': 15, 'dimension': 'SN', 'direction': 'N',
     'text': '我喜欢探索新想法和概念, 即使它们不一定有实际用途'},
    {'id': 16, 'dimension': 'SN', 'direction': 'S',
     'text': '学习新知识时, 我更喜欢从具体例子入手, 而不是先看理论框架'},
    {'id': 17, 'dimension': 'SN', 'direction': 'N',
     'text': '我常常发现事物之间的隐藏联系和深层模式'},
    {'id': 18, 'dimension': 'SN', 'direction': 'S',
     'text': '相对于创新, 我更倾向于信赖经过验证的方法和经验'},
    {'id': 19, 'dimension': 'SN', 'direction': 'N',
     'text': '我更喜欢关注整体图景和宏观方向, 而不是具体细节'},
    {'id': 20, 'dimension': 'SN', 'direction': 'S',
     'text': '我认为脚踏实地、关注当下比空想未来更重要'},
    {'id': 21, 'dimension': 'SN', 'direction': 'N',
     'text': '我常常产生一些别人觉得"脑洞大开"的想法和灵感'},
    {'id': 22, 'dimension': 'SN', 'direction': 'S',
     'text': '我习惯按部就班地完成任务, 不喜欢频繁变动'},
    {'id': 23, 'dimension': 'SN', 'direction': 'N',
     'text': '做重复性工作让我感到无聊, 我渴望不断尝试新方法'},

    # ──────────── TF 维度(12题) ────────────
    # T(思考): 以逻辑和客观标准做决策
    # F(情感): 以价值观和对人的影响做决策
    {'id': 24, 'dimension': 'TF', 'direction': 'T',
     'text': '做决策时, 我主要依靠逻辑和客观分析, 而非个人感受'},
    {'id': 25, 'dimension': 'TF', 'direction': 'F',
     'text': '我很容易感受到他人的情绪, 并会考虑决定对他人的影响'},
    {'id': 26, 'dimension': 'TF', 'direction': 'T',
     'text': '我认为公平和一致性比照顾个别人的感受更重要'},
    {'id': 27, 'dimension': 'TF', 'direction': 'F',
     'text': '当他人遇到困难时, 我的第一反应是给予情感支持而非解决问题'},
    {'id': 28, 'dimension': 'TF', 'direction': 'T',
     'text': '在争论中, 我倾向于基于事实和逻辑来说服对方'},
    {'id': 29, 'dimension': 'TF', 'direction': 'F',
     'text': '即便对方的观点在逻辑上站得住脚, 如果违背了我的价值观我也难以接受'},
    {'id': 30, 'dimension': 'TF', 'direction': 'T',
     'text': '我喜欢直截了当的沟通方式, 不太在意措辞是否"好听"'},
    {'id': 31, 'dimension': 'TF', 'direction': 'F',
     'text': '和谐的人际关系对我来说比完成某个具体任务更重要'},
    {'id': 32, 'dimension': 'TF', 'direction': 'T',
     'text': '面对批评时, 我更关注批评内容是否正确, 而非对方的态度'},
    {'id': 33, 'dimension': 'TF', 'direction': 'F',
     'text': '我经常设身处地为他人着想, 理解他们行为背后的原因'},
    {'id': 34, 'dimension': 'TF', 'direction': 'T',
     'text': '我认为在大多数情况下, 规则和原则应该一视同仁地适用'},
    {'id': 35, 'dimension': 'TF', 'direction': 'F',
     'text': '即使一个决定在逻辑上是合理的, 如果它会伤害到别人我也会犹豫'},

    # ──────────── JP 维度(12题) ────────────
    # J(判断): 喜欢计划、组织和确定性
    # P(感知): 喜欢灵活、开放和自发性
    {'id': 36, 'dimension': 'JP', 'direction': 'J',
     'text': '我喜欢提前制定清晰的计划, 并严格按照计划执行'},
    {'id': 37, 'dimension': 'JP', 'direction': 'P',
     'text': '我喜欢保持选择的开放性, 不喜欢过早做决定'},
    {'id': 38, 'dimension': 'JP', 'direction': 'J',
     'text': '工作和生活环境的整洁有序让我感到安心'},
    {'id': 39, 'dimension': 'JP', 'direction': 'P',
     'text': '我对临时产生的计划变更感到兴奋, 并能快速适应'},
    {'id': 40, 'dimension': 'JP', 'direction': 'J',
     'text': '我会为任务设置明确的截止时间, 并努力遵守'},
    {'id': 41, 'dimension': 'JP', 'direction': 'P',
     'text': '我倾向于在截止日期临近时才集中精力完成任务'},
    {'id': 42, 'dimension': 'JP', 'direction': 'J',
     'text': '我希望事情能够按预期的节奏推进, 不喜欢突发状况'},
    {'id': 43, 'dimension': 'JP', 'direction': 'P',
     'text': '我享受随性和即兴的生活方式, 不喜欢被计划束缚'},
    {'id': 44, 'dimension': 'JP', 'direction': 'J',
     'text': '做出决定后我感到轻松, 反复纠结让我很有压力'},
    {'id': 45, 'dimension': 'JP', 'direction': 'P',
     'text': '我喜欢同时处理多个项目, 在不同的任务间自由切换'},
    {'id': 46, 'dimension': 'JP', 'direction': 'J',
     'text': '我习惯列清单和管理日程, 那是我的生活掌控感来源'},
    {'id': 47, 'dimension': 'JP', 'direction': 'P',
     'text': '我认为生活中最美好的时刻往往来自于计划之外的惊喜'},
]


def get_questions(group_by_dimension=False):
    """获取 MBTI 题目列表

    参数:
        group_by_dimension: 是否按维度分组返回。默认 False, 返回扁平的题目列表;
                           True 时返回 {维度: [题目列表]} 的字典结构。

    返回:
        list 或 dict: 题目数据, 每道题包含 id、text、dimension、direction 字段

    用法:
        questions = get_questions()              # 扁平列表, 48 道题
        grouped = get_questions(group_by_dimension=True)  # 按 EI/SN/TF/JP 分组
    """
    if group_by_dimension:
        grouped = {'EI': [], 'SN': [], 'TF': [], 'JP': []}
        for q in MBTI_QUESTIONS:
            grouped[q['dimension']].append(q)
        return grouped
    return MBTI_QUESTIONS


def get_dimension_info():
    """获取维度说明信息

    返回:
        dict: 各维度的名称、两个极的描述, 以及题目数量
    """
    return {
        'EI': {
            'name': '外向(E) vs 内向(I)',
            'description': '你从哪里获得能量?',
            'pole_a': {'code': 'E', 'name': '外向', 'desc': '从外部世界和人际互动中获取能量'},
            'pole_b': {'code': 'I', 'name': '内向', 'desc': '从独处和内省中获取能量'},
            'question_count': 12,
        },
        'SN': {
            'name': '实感(S) vs 直觉(N)',
            'description': '你如何获取信息?',
            'pole_a': {'code': 'S', 'name': '实感', 'desc': '关注具体事实、细节和现实经验'},
            'pole_b': {'code': 'N', 'name': '直觉', 'desc': '关注抽象概念、可能性和未来愿景'},
            'question_count': 12,
        },
        'TF': {
            'name': '思考(T) vs 情感(F)',
            'description': '你如何做决策?',
            'pole_a': {'code': 'T', 'name': '思考', 'desc': '以逻辑和客观标准做决策'},
            'pole_b': {'code': 'F', 'name': '情感', 'desc': '以价值观和对人的影响做决策'},
            'question_count': 12,
        },
        'JP': {
            'name': '判断(J) vs 感知(P)',
            'description': '你如何应对外部世界?',
            'pole_a': {'code': 'J', 'name': '判断', 'desc': '喜欢计划、组织和确定性'},
            'pole_b': {'code': 'P', 'name': '感知', 'desc': '喜欢灵活、开放和自发性'},
            'question_count': 12,
        },
    }


if __name__ == '__main__':
    # 模块自检: 验证题目数量和维度分布
    print(f"共加载 {len(MBTI_QUESTIONS)} 道 MBTI 测试题")
    from collections import Counter
    dims = Counter(q['dimension'] for q in MBTI_QUESTIONS)
    for dim, count in sorted(dims.items()):
        directions = Counter(q['direction'] for q in MBTI_QUESTIONS if q['dimension'] == dim)
        print(f"  {dim}: {count}题 — 方向分布: {dict(directions)}")
