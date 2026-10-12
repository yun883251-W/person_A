"""
MBTI 人格测试 — 计分与类型判定模块

本模块提供 MBTI 测试的核心计分逻辑, 包括:
  1. calculate_mbti()         — 根据答题结果计算 4 个维度的类型
  2. get_dimension_breakdown() — 获取各维度的百分比得分明细
  3. validate_answers()       — 验证答题数据的完整性

计分算法说明:
  - 每个维度有 12 道题, 其中 6 道指向该维度的第一个极(如 E), 6 道指向第二个极(如 I)
  - 每道题得分 1-5(Likert 5点量表)
  - 维度各极的总分 = 该方向上所有题目的得分之和
  - 百分比 = 该极得分 / 该维度所有题目总分 × 100%
  - 类型判定: 百分比更高的一极胜出; 平局时默认取该维度更常见的极(E/S/T/J)

文件结构:
  1. 常量定义 — 维度方向映射、平局默认值
  2. validate_answers()       — 答题验证
  3. calculate_mbti()         — 主计算函数
  4. get_dimension_breakdown() — 维度细分明细
"""

from .questions import MBTI_QUESTIONS

# ═══════════════ 常量定义 ═══════════════

# 四个维度的两极字母
DIMENSION_POLES = {
    'EI': ('E', 'I'),   # 外向 vs 内向
    'SN': ('S', 'N'),   # 实感 vs 直觉
    'TF': ('T', 'F'),   # 思考 vs 情感
    'JP': ('J', 'P'),   # 判断 vs 感知
}

# 平局时默认选择(取该维度在一般人群中更常见的一极)
# E 比 I 常见, S 比 N 常见, F 比 T 常见, J 比 P 常见
TIE_DEFAULTS = {
    'EI': 'I',   # 内向者在人群中略多（内向倾向更常见于反思性测试）
    'SN': 'S',   # 实感型约占 70% 人口
    'TF': 'F',   # 情感型在一般人群中约占 60%
    'JP': 'J',   # 判断型在一般人群中约占 54%
}


def validate_answers(answers):
    """验证答题数据的完整性和有效性

    参数:
        answers: dict, 格式为 {题目id(int或str): 得分(int 1-5), ...}

    返回:
        tuple: (is_valid: bool, error_message: str or None)
            is_valid=True  时 error_message 为 None
            is_valid=False 时 error_message 为错误描述

    校验规则:
        1. answers 不能为空
        2. 必须涵盖所有 48 道题
        3. 每道题的得分必须在 1-5 之间
        4. 题目 ID 必须在有效范围内(0-47)
    """
    if not answers or not isinstance(answers, dict):
        return False, '答题数据为空或格式不正确'

    # 检查是否答完所有题目
    required_ids = set(range(len(MBTI_QUESTIONS)))
    answered_ids = set()
    for key in answers:
        try:
            qid = int(key)
            answered_ids.add(qid)
        except (ValueError, TypeError):
            return False, f'无效的题目ID: {key}'

    missing = required_ids - answered_ids
    if missing:
        return False, f'还有 {len(missing)} 道题未作答(题目ID: {sorted(missing)[:5]}...)'

    extra = answered_ids - required_ids
    if extra:
        return False, f'包含无效的题目ID: {sorted(extra)}'

    # 检查每题得分范围
    for qid, score in answers.items():
        try:
            score_val = int(score)
            if score_val < 1 or score_val > 5:
                return False, f'题目 {qid} 的得分({score_val})不在有效范围(1-5)内'
        except (ValueError, TypeError):
            return False, f'题目 {qid} 的得分({score})不是有效数字'

    return True, None


def calculate_mbti(answers):
    """根据答题结果计算 MBTI 类型

    参数:
        answers: dict, 格式为 {题目id: 得分}, 如 {'0': 4, '1': 2, ..., '47': 5}
                 题目id 可以是 int 或 str 类型, 得分范围为 1-5

    返回:
        dict: {
            'type': str,           # 最终类型, 如 'INTJ'
            'type_name': str,      # 中文名称, 如 'INTJ — 建筑师'
            'type_nickname': str,  # 英文昵称, 如 'Architect'
            'dimension_scores': {  # 每个维度的详细得分
                'EI': {
                    'E': int, 'I': int,                       # 各极总分
                    'E_pct': float, 'I_pct': float,           # 各极百分比
                    'result': 'E',                             # 胜出极
                    'label': '外向(E) — 从人际互动中获取能量',  # 结果描述
                },
                'SN': {...}, 'TF': {...}, 'JP': {...}
            },
            'is_valid': bool,      # 数据是否有效
            'error': str or None,  # 错误信息(有效时为None)
        }

    算法步骤:
        1. 验证答题数据完整性
        2. 按维度和方向分组, 计算各极总分
        3. 计算百分比, 确定胜出极
        4. 拼接 4 字母类型代码
        5. 查询类型描述并返回

    用法:
        answers = {'0': 3, '1': 5, ...}  # 48道题的作答
        result = calculate_mbti(answers)
        print(result['type'])  # 'INTJ'
    """
    # 步骤1: 验证
    is_valid, error = validate_answers(answers)
    if not is_valid:
        return {
            'type': None,
            'type_name': None,
            'type_nickname': None,
            'dimension_scores': {},
            'is_valid': False,
            'error': error,
        }

    # 步骤2: 按维度累计各极得分
    # 初始化得分累计器: {维度: {极: 总分, ...}}
    raw_scores = {}
    for dim, (pole_a, pole_b) in DIMENSION_POLES.items():
        raw_scores[dim] = {pole_a: 0, pole_b: 0}

    for question in MBTI_QUESTIONS:
        qid = question['id']
        dim = question['dimension']
        direction = question['direction']
        # 兼容 int 和 str 两种 key 格式
        score = int(answers.get(str(qid), answers.get(qid, 3)))
        raw_scores[dim][direction] += score

    # 步骤3: 计算百分比, 确定各维度结果
    dimension_scores = {}
    type_letters = []

    for dim, (pole_a, pole_b) in DIMENSION_POLES.items():
        score_a = raw_scores[dim][pole_a]
        score_b = raw_scores[dim][pole_b]
        total = score_a + score_b

        # 计算百分比(保留1位小数)
        pct_a = round(score_a / total * 100, 1) if total > 0 else 50.0
        pct_b = round(score_b / total * 100, 1) if total > 0 else 50.0

        # 确定胜出极
        if pct_a > pct_b:
            result_pole = pole_a
        elif pct_b > pct_a:
            result_pole = pole_b
        else:
            # 平局: 使用预设默认值
            result_pole = TIE_DEFAULTS[dim]

        type_letters.append(result_pole)

        # 构建维度标签描述
        pole_names = {
            'E': '外向(E) — 从人际互动中获取能量',
            'I': '内向(I) — 从独处和内省中获取能量',
            'S': '实感(S) — 关注具体事实和现实经验',
            'N': '直觉(N) — 关注抽象概念和未来可能',
            'T': '思考(T) — 以逻辑和客观标准做决策',
            'F': '情感(F) — 以价值观和对人的影响做决策',
            'J': '判断(J) — 喜欢计划、组织和确定性',
            'P': '感知(P) — 喜欢灵活、开放和自发性',
        }

        dimension_scores[dim] = {
            pole_a: score_a,
            pole_b: score_b,
            f'{pole_a}_pct': pct_a,
            f'{pole_b}_pct': pct_b,
            'total': total,
            'result': result_pole,
            'label': pole_names[result_pole],
        }

    # 步骤4: 拼接类型代码
    mbti_type = ''.join(type_letters)  # 如 'INTJ'

    # 步骤5: 获取类型描述
    from .descriptions import get_type_description, get_type_name

    desc = get_type_description(mbti_type)
    type_name = desc['name'] if desc else f'{mbti_type}'
    type_nickname = desc['nickname'] if desc else ''

    return {
        'type': mbti_type,
        'type_name': type_name,
        'type_nickname': type_nickname,
        'dimension_scores': dimension_scores,
        'is_valid': True,
        'error': None,
    }


def get_dimension_breakdown(answers):
    """获取各维度的详细百分比分解(用于前端图表展示)

    参数:
        answers: dict, 同 calculate_mbti() 的 answers 参数

    返回:
        dict: {
            'EI': {'E': 28, 'I': 22, 'E_pct': 56.0, 'I_pct': 44.0, 'result': 'E'},
            'SN': {...}, 'TF': {...}, 'JP': {...},
            'type': 'ESTJ',
            'scores_raw': {...},  # 每题原始得分
        }
    """
    result = calculate_mbti(answers)
    if not result['is_valid']:
        return {'error': result['error']}

    # 附加每题原始得分(按维度分组)
    answers_by_dim = {'EI': {}, 'SN': {}, 'TF': {}, 'JP': {}}
    for q in MBTI_QUESTIONS:
        qid = str(q['id'])
        score = int(answers.get(qid, answers.get(int(qid), 3)))
        answers_by_dim[q['dimension']][qid] = {
            'text': q['text'],
            'direction': q['direction'],
            'score': score,
        }

    return {
        'type': result['type'],
        'dimension_scores': result['dimension_scores'],
        'answers_detail': answers_by_dim,
    }


if __name__ == '__main__':
    # 模块自检: 用全3分(中立)的答案测试计算逻辑
    print("=== MBTI 计分模块自检 ===")

    # 测试1: 全中立答案
    neutral = {str(i): 3 for i in range(48)}
    result = calculate_mbti(neutral)
    print(f"全中立答案 → 类型: {result['type']} (预期: 全取默认值 I/S/F/J → ISFJ)")
    print(f"  维度得分: EI={result['dimension_scores']['EI']}")

    # 测试2: 极端外向/实感/思考/判断
    extreme_estj = {}
    for q in MBTI_QUESTIONS:
        if q['direction'] in ('E', 'S', 'T', 'J'):
            extreme_estj[str(q['id'])] = 5
        else:
            extreme_estj[str(q['id'])] = 1

    result2 = calculate_mbti(extreme_estj)
    print(f"极端ESTJ答案 → 类型: {result2['type']} (预期: ESTJ)")

    # 测试3: 答案不完整
    partial = {'0': 4, '1': 2, '2': 5}  # 仅3道题
    result3 = calculate_mbti(partial)
    print(f"不完整答案 → 有效: {result3['is_valid']}, 错误: {result3['error']}")
