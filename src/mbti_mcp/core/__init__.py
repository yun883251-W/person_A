"""
MBTI 人格测试 — Person_A 内置核心模块

源代码复制自 MindCare/backend/mbti，保留原题库与计分逻辑，仅适配本地包导入。
该问卷用于人格倾向自评，并非官方 MBTI 测评。

子模块说明:
  1. questions.py     — 48 道自评题目 + 5 点李克特量表
  2. descriptions.py  — 16 种人格类型的详细中文描述
  3. scoring.py       — 计分算法: 维度得分计算 + 类型判定 + 平局处理

对外接口:
  - get_questions()            — 获取题目列表(支持按维度分组)
  - get_dimension_info()       — 获取维度说明
  - calculate_mbti(answers)    — 计算 MBTI 类型和维度得分
  - get_dimension_breakdown(answers) — 获取维度详细分解
  - get_type_description(type) — 获取类型详细描述
  - get_type_name(type)        — 获取类型中文名
  - validate_answers(answers)  — 验证答题数据

典型调用流程:
  from mbti_mcp.core import get_questions, calculate_mbti, get_type_description

  # 1. 前端获取题目
  questions = get_questions()

  # 2. 用户提交答案后计算类型
  answers = {'0': 4, '1': 2, '2': 5, ...}  # 48 道题的作答
  result = calculate_mbti(answers)
  print(result['type'])  # 如 'INTJ'

  # 3. 获取该类型的详细描述
  desc = get_type_description(result['type'])
  print(desc['name'], desc['traits'])
"""

# 题目相关
from .questions import (
    MBTI_QUESTIONS,
    OPTIONS,
    get_questions,
    get_dimension_info,
)

# 计分相关
from .scoring import (
    calculate_mbti,
    get_dimension_breakdown,
    validate_answers,
    DIMENSION_POLES,
    TIE_DEFAULTS,
)

# 类型描述相关
from .descriptions import (
    MBTI_DESCRIPTIONS,
    MBTI_TYPE_NAMES,
    get_type_description,
    get_type_name,
)

# 统一对外暴露的接口列表
__all__ = [
    # 题目
    'MBTI_QUESTIONS',
    'OPTIONS',
    'get_questions',
    'get_dimension_info',
    # 计分
    'calculate_mbti',
    'get_dimension_breakdown',
    'validate_answers',
    'DIMENSION_POLES',
    'TIE_DEFAULTS',
    # 描述
    'MBTI_DESCRIPTIONS',
    'MBTI_TYPE_NAMES',
    'get_type_description',
    'get_type_name',
]
