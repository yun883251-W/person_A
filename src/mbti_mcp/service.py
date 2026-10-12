"""调用本项目内置的问卷与评分核心，源代码来自 MindCare。

本服务没有答题会话、数据库或模型。宿主负责保存用户明确提交的答案；
这里在收到完整且版本一致的答案后调用 core 中的确定性评分函数。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from . import core

QUESTION_COUNT = 48
MAX_PAGE_SIZE = 12
ASSESSMENT_NOTICE = (
    "这是 MindCare 项目的 48 题人格倾向自评问卷，用于自我了解；"
    "不是官方 MBTI 测评或心理诊断。维度百分比是答题得分占比，不是置信度。"
)
_TYPE_CODE = re.compile(r"[EI][SN][TF][JP]")
_DESCRIPTION_FIELDS = (
    "name", "nickname", "group", "traits", "strengths", "growth_areas",
    "careers", "relationships",
)


class MbtiSourceError(RuntimeError):
    """内置题库或数据契约不兼容；消息不包含用户答案。"""


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MbtiService:
    """一个进程持有一个题库快照，工具调用不修改源数据或保存用户输入。"""

    def __init__(self) -> None:
        self._source = core
        try:
            self._questions = copy.deepcopy(self._source.get_questions())
            self._options = copy.deepcopy(self._source.OPTIONS)
            self._poles = copy.deepcopy(self._source.DIMENSION_POLES)
            self._tie_defaults = copy.deepcopy(self._source.TIE_DEFAULTS)
            self._validate_source_data()
            self.questionnaire_version = "mindcare-questionnaire-v1:" + _json_digest({
                "questions": self._questions,
                "options": self._options,
            })
            # 内容版本保留计分文件哈希，修改源计分代码后旧结果仍可辨认所用规则。
            scoring_bytes = Path(__file__).with_name("core").joinpath("scoring.py").read_bytes()
            self.scoring_version = "mindcare-scoring-v1:" + hashlib.sha256(scoring_bytes).hexdigest()
        except MbtiSourceError:
            raise
        except Exception as exc:
            raise MbtiSourceError("mindcare_data_invalid: 题库、选项或评分规则不兼容。") from exc

    def _validate_source_data(self) -> None:
        if not isinstance(self._questions, list) or len(self._questions) != QUESTION_COUNT:
            raise MbtiSourceError("mindcare_data_invalid: 当前接入要求完整的 48 题问卷。")
        expected_poles = {"EI": ("E", "I"), "SN": ("S", "N"), "TF": ("T", "F"), "JP": ("J", "P")}
        if self._poles != expected_poles or set(self._tie_defaults) != set(expected_poles):
            raise MbtiSourceError("mindcare_data_invalid: 四个维度的配置不兼容。")
        for dimension, poles in expected_poles.items():
            if self._tie_defaults[dimension] not in poles:
                raise MbtiSourceError("mindcare_data_invalid: 平局规则不兼容。")
        counts = {pole: 0 for poles in expected_poles.values() for pole in poles}
        for position, question in enumerate(self._questions):
            if not isinstance(question, dict) or type(question.get("id")) is not int or question["id"] != position:
                raise MbtiSourceError("mindcare_data_invalid: 题目 ID 必须按 0 至 47 连续排列。")
            if not isinstance(question.get("text"), str) or not question["text"].strip():
                raise MbtiSourceError("mindcare_data_invalid: 题目文本缺失。")
            dimension, direction = question.get("dimension"), question.get("direction")
            if dimension not in expected_poles or direction not in expected_poles[dimension]:
                raise MbtiSourceError("mindcare_data_invalid: 题目维度或计分方向无效。")
            counts[direction] += 1
        if set(counts.values()) != {6}:
            raise MbtiSourceError("mindcare_data_invalid: 每个计分方向应包含 6 道题。")
        if not isinstance(self._options, list) or len(self._options) != 5:
            raise MbtiSourceError("mindcare_data_invalid: 问卷须提供五个量表选项。")
        for value, option in enumerate(self._options, start=1):
            if (not isinstance(option, dict) or type(option.get("value")) is not int
                    or option["value"] != value or not isinstance(option.get("label"), str)
                    or not option["label"].strip()):
                raise MbtiSourceError("mindcare_data_invalid: 量表选项应为整数 1 至 5。")

    def get_mbti_questionnaire(self, offset: int = 0, limit: int = MAX_PAGE_SIZE) -> dict[str, Any]:
        """返回一页题目与统一选项，隐藏用于计分的 direction 和 dimension。"""
        if type(offset) is not int or not 0 <= offset < QUESTION_COUNT:
            raise ValueError("invalid_offset: offset 必须是 0 至 47 的整数。")
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError("invalid_limit: limit 必须是 1 至 12 的整数。")
        end = min(offset + limit, QUESTION_COUNT)
        return {
            "questions": [{"id": q["id"], "text": q["text"]} for q in self._questions[offset:end]],
            "options": [{"value": item["value"], "label": item["label"]} for item in self._options],
            "total": QUESTION_COUNT,
            "offset": offset,
            "limit": limit,
            "next_offset": end if end < QUESTION_COUNT else None,
            "questionnaire_version": self.questionnaire_version,
            "scoring_version": self.scoring_version,
            "notice": ASSESSMENT_NOTICE,
        }

    def score_mbti_answers(self, answers: dict[str, int], questionnaire_version: str) -> dict[str, Any]:
        """完整、严格校验后调用原评分函数；绝不补答、强制转换或由模型评分。"""
        if not isinstance(questionnaire_version, str) or questionnaire_version != self.questionnaire_version:
            raise ValueError("questionnaire_version_mismatch: 题库已变化，请重新获取问卷后答题。")
        if type(answers) is not dict:
            raise ValueError("invalid_answers: answers 必须是题目 ID 到整数分值的 JSON 对象。")
        required_ids = {str(index) for index in range(QUESTION_COUNT)}
        if any(type(key) is not str or key not in required_ids for key in answers):
            raise ValueError("invalid_question_id: 题目 ID 必须严格使用字符串 0 至 47，不能包含前导零。")
        if set(answers) != required_ids:
            raise ValueError("incomplete_answers: 必须提交全部 48 道题的真实答案，不会自动补答。")
        if any(type(value) is not int or not 1 <= value <= 5 for value in answers.values()):
            raise ValueError("invalid_answer_value: 每题必须是 1 至 5 的整数，不接受字符串、布尔值或小数。")
        result = self._source.calculate_mbti(dict(answers))
        if not isinstance(result, dict) or result.get("is_valid") is not True:
            raise MbtiSourceError("mindcare_scoring_failed: 源评分函数未返回有效结果。")
        scores = result.get("dimension_scores")
        type_code = result.get("type")
        if not isinstance(scores, dict) or set(scores) != set(self._poles) or not isinstance(type_code, str) or _TYPE_CODE.fullmatch(type_code) is None:
            raise MbtiSourceError("mindcare_scoring_failed: 源评分结果结构不兼容。")
        ties = {}
        for dimension, (pole_a, pole_b) in self._poles.items():
            # 采用源算法实际比较的百分比，忠实保留其平局行为。
            ties[dimension] = scores[dimension][f"{pole_a}_pct"] == scores[dimension][f"{pole_b}_pct"]
        return {
            "type": type_code,
            "type_name": result["type_name"],
            "type_nickname": result["type_nickname"],
            "dimension_scores": copy.deepcopy(scores),
            "ties": ties,
            "tie_defaults": copy.deepcopy(self._tie_defaults),
            "questionnaire_version": self.questionnaire_version,
            "scoring_version": self.scoring_version,
            "notice": ASSESSMENT_NOTICE + " 平局类型字母按原项目预设规则产生，不表示该维度存在明确倾向。",
        }

    def get_mbti_description(self, type_code: str) -> dict[str, Any]:
        """返回源项目类型描述，省略含无出处人口比例的 overview 字段。"""
        if not isinstance(type_code, str) or len(type_code) > 16:
            raise ValueError("invalid_type_code: 请提供 INTJ 等四字母人格类型。")
        normalized = type_code.strip().upper()
        if _TYPE_CODE.fullmatch(normalized) is None:
            raise ValueError("invalid_type_code: 类型应依次由 E/I、S/N、T/F、J/P 组成。")
        description = self._source.get_type_description(normalized)
        if not isinstance(description, dict):
            raise MbtiSourceError("mindcare_description_missing: 源项目没有该类型的描述。")
        return {
            "type": normalized,
            "description": {key: copy.deepcopy(description[key]) for key in _DESCRIPTION_FIELDS if key in description},
            "notice": ASSESSMENT_NOTICE + " 以下为类型的一般性描述，不据此推断用户尚未表达的特征。",
        }
