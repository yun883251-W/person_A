"""Session-owned MBTI interaction; questionnaire and scoring run through MCP.

Each handled turn produces a new snapshot. The Agent commits that snapshot with
the answer, so a failed database write never advances the in-memory questionnaire.
This module neither loads MindCare nor owns a database or MCP connection.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

QUESTION_TOOL = "mcp__mbti__get_mbti_questionnaire"
SCORE_TOOL = "mcp__mbti__score_mbti_answers"
DESCRIPTION_TOOL = "mcp__mbti__get_mbti_description"
TOOLS = (QUESTION_TOOL, SCORE_TOOL, DESCRIPTION_TOOL)
ACTIONS = frozenset({"start", "restart", "answer", "resume", "pause", "cancel", "result", "describe"})
STATUSES = frozenset({"active", "paused", "ready", "completed", "cancelled"})
_TYPE_RE = re.compile(r"(?<![A-Za-z])([EI][SN][TF][JP])(?![A-Za-z])", re.I)
_ANSWER_RE = re.compile(r"^MBTI答题 ([a-f0-9]{32}) ([0-9]{1,2}) ([1-5])$", re.I)
_SCORE_RE = re.compile(r"^(?:我选|选择|选)?\s*([1-5])(?:分)?[。.!！]?$", re.I)
_LABELS = {"完全不符合": 1, "不太符合": 2, "中立": 3, "中立/不确定": 3,
           "不确定": 3, "比较符合": 4, "完全符合": 5}
NOTICE = "这是 MindCare 的人格偏好自评问卷；结果用于自我了解，维度得分比例不是置信度。"


class AssessmentToolError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class AssessmentReply:
    answer: str
    state: dict[str, Any] | None = None
    status: str = "success"
    stop_reason: str = "mbti_progress"


def request_action(question: str, state: Mapping[str, Any] | None) -> str | None:
    """Recognize explicit test controls and answers without guessing personality."""
    text = question.strip()
    folded = text.casefold()
    has_mbti = "mbti" in folded or "人格测试" in text or "性格测试" in text
    ongoing = state is not None and state.get("status") in {"active", "paused", "ready"}
    if folded.startswith("mbti答题 "):
        return "answer"
    if has_mbti and any(word in text for word in ("重新", "重做", "重测")):
        return "restart"
    if has_mbti or ongoing:
        if any(word in folded for word in ("退出测试", "退出 mbti", "退出mbti", "结束测试", "取消测试")):
            return "cancel"
        if "暂停" in text and (has_mbti or text in {"暂停", "暂停测试"}):
            return "pause"
        if any(word in text for word in ("继续", "恢复")) and (
            has_mbti or text in {"继续", "继续测试", "继续答题", "恢复测试", "下一题"}
        ):
            return "resume"
        if has_mbti and any(word in text for word in ("结果", "得分", "评分")):
            return "result"
    if has_mbti and not any(word in text for word in ("是什么", "怎么实现", "如何实现", "原理", "代码")):
        if any(word in text for word in ("开始", "想做", "想测", "做个", "做一下", "测一下", "帮我测", "我要测", "我要做")):
            return "start"
        if folded in {"mbti", "mbti测试", "mbti 测试", "人格测试", "性格测试"}:
            return "start"
    if _TYPE_RE.search(text) and any(word in text for word in ("介绍", "解释", "了解", "特点", "是什么", "描述")):
        return "describe"
    if ongoing and (_SCORE_RE.fullmatch(text) or text in _LABELS):
        return "answer"
    if ongoing and text in {"下一题", "继续答题"}:
        return "resume"
    return None


def validate_snapshot(value: object) -> dict[str, Any]:
    """Bound and validate persisted questionnaire state, including old snapshots."""
    if not isinstance(value, dict):
        raise ValueError("Invalid MBTI snapshot")
    required = {"schema_version", "test_id", "status", "total", "questionnaire_version",
                "scoring_version", "answers", "questions", "options", "result"}
    if set(value) != required or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported MBTI snapshot")
    if not isinstance(value["test_id"], str) or not re.fullmatch(r"[a-f0-9]{32}", value["test_id"]):
        raise ValueError("Invalid MBTI test ID")
    if not isinstance(value["status"], str) or value["status"] not in STATUSES or type(value["total"]) is not int or value["total"] != 48:
        raise ValueError("Invalid MBTI progress")
    for key in ("questionnaire_version", "scoring_version"):
        if not isinstance(value[key], str) or not 1 <= len(value[key]) <= 160:
            raise ValueError("Invalid MBTI version")
    answers = value["answers"]
    if not isinstance(answers, dict) or len(answers) > 48 or set(answers) != {str(i) for i in range(len(answers))}:
        raise ValueError("Invalid MBTI answer sequence")
    if any(type(score) is not int or not 1 <= score <= 5 for score in answers.values()):
        raise ValueError("Invalid MBTI answer")
    questions = value["questions"]
    if not isinstance(questions, list) or not 1 <= len(questions) <= 48 or len(answers) > len(questions):
        raise ValueError("Invalid MBTI questions")
    for index, item in enumerate(questions):
        if (not isinstance(item, dict) or set(item) != {"id", "text"}
                or type(item["id"]) is not int or item["id"] != index
                or not isinstance(item["text"], str) or not 1 <= len(item["text"]) <= 2000):
            raise ValueError("Invalid MBTI question")
    options = value["options"]
    if not isinstance(options, list) or len(options) != 5:
        raise ValueError("Invalid MBTI options")
    for index, item in enumerate(options, 1):
        if (not isinstance(item, dict) or set(item) != {"value", "label"}
                or type(item["value"]) is not int or item["value"] != index
                or not isinstance(item["label"], str) or not 1 <= len(item["label"]) <= 80):
            raise ValueError("Invalid MBTI option")
    if value["result"] is not None:
        _validate_result(value["result"], value)
    if value["status"] == "completed" and (len(answers) != 48 or value["result"] is None):
        raise ValueError("Incomplete MBTI result")
    if value["status"] == "ready" and len(answers) != 48:
        raise ValueError("Incomplete MBTI answers")
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(serialized) > 100_000:
        raise ValueError("MBTI snapshot too large")
    return copy.deepcopy(value)


def _validate_result(value: object, state: Mapping[str, Any]) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("type"), str) or not _TYPE_RE.fullmatch(value["type"]):
        raise ValueError("Invalid MBTI result")
    if value.get("questionnaire_version") != state["questionnaire_version"] or value.get("scoring_version") != state["scoring_version"]:
        raise ValueError("MBTI version changed")
    scores = value.get("dimension_scores")
    ties = value.get("ties")
    if not isinstance(scores, dict) or not isinstance(ties, dict):
        raise ValueError("Invalid MBTI dimensions")
    for dimension, poles in {"EI": "EI", "SN": "SN", "TF": "TF", "JP": "JP"}.items():
        item = scores.get(dimension)
        if not isinstance(item, dict) or item.get("result") not in tuple(poles) or type(ties.get(dimension)) is not bool:
            raise ValueError("Invalid MBTI dimension")
        for pole in poles:
            score = item.get(pole + "_pct")
            if type(score) not in {int, float} or not 0 <= score <= 100:
                raise ValueError("Invalid MBTI percentage")


def public_assessment(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    state = validate_snapshot(value)
    answered = len(state["answers"])
    question = None
    if state["status"] in {"active", "paused"} and answered < len(state["questions"]):
        item = state["questions"][answered]
        question = {"id": item["id"], "number": answered + 1, "text": item["text"]}
    return {
        "test_id": state["test_id"], "status": state["status"],
        "answered_count": answered, "total": state["total"],
        "current_question": question, "options": state["options"], "result": state["result"],
    }


def _question_text(state: dict[str, Any], prefix: str = "") -> str:
    public = public_assessment(state)
    question = public["current_question"]
    if question is None:
        return prefix + "当前题目暂未加载，请发送“继续 MBTI 测试”重试。"
    options = "；".join(f"{item['value']}={item['label']}" for item in state["options"])
    return (f"{prefix}MBTI 自评：第 {question['number']}/{state['total']} 题\n\n"
            f"{question['text']}\n\n{options}\n\n回复 1—5，或点击选项。可发送“暂停 MBTI 测试”或“退出 MBTI 测试”。")


def _load_page(state: dict[str, Any], call: Callable[..., dict[str, Any]]) -> None:
    offset = len(state["questions"])
    page = call(QUESTION_TOOL, {"offset": offset, "limit": 12})
    if (page.get("questionnaire_version") != state["questionnaire_version"]
            or page.get("scoring_version") != state["scoring_version"]
            or page.get("total") != state["total"]):
        raise AssessmentToolError("mbti_version_changed")
    questions = page.get("questions")
    if not isinstance(questions, list) or not questions:
        raise AssessmentToolError("mbti_invalid_questionnaire")
    state["questions"].extend(questions)
    validate_snapshot(state)


def _result_text(result: dict[str, Any]) -> str:
    lines = [f"本次自评结果：{result.get('type_name') or result['type']}", "", "四个维度的得分比例："]
    for dimension, poles in {"EI": "EI", "SN": "SN", "TF": "TF", "JP": "JP"}.items():
        item = result["dimension_scores"][dimension]
        label = " / ".join(f"{pole} {item[pole + '_pct']}%" for pole in poles)
        tie = "（平局，按原问卷默认规则取 " + item["result"] + "）" if result["ties"][dimension] else ""
        lines.append(f"- {label}{tie}")
    lines.extend(["", NOTICE, f"可以继续提问“介绍 {result['type']} 的特点”，或发送“重新开始 MBTI 测试”。"])
    return "\n".join(lines)


def handle_request(
    action: str, question: str, previous: dict[str, Any] | None,
    call: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> AssessmentReply:
    """Advance one host-managed step; call only the three fixed MBTI tools."""
    state = validate_snapshot(previous) if previous is not None else None
    if action == "describe":
        match = _TYPE_RE.search(question)
        if match is None:
            return AssessmentReply("请提供要了解的四字母类型，例如 INTJ。", state, "needs_clarification")
        response = call(DESCRIPTION_TOOL, {"type_code": match.group(1).upper()})
        description = response.get("description")
        if not isinstance(description, dict):
            raise AssessmentToolError("mbti_invalid_description")
        lines = [str(description.get("name") or match.group(1).upper())]
        for key, title in (("traits", "特点"), ("strengths", "优势"), ("growth_areas", "可探索的成长方向")):
            values = description.get(key)
            if isinstance(values, list):
                lines.extend(["", title, *(f"- {item}" for item in values if isinstance(item, str))])
        lines.extend(["", "以上为 MindCare 题库中的类型描述，不代表对个人的确定判断。"])
        return AssessmentReply("\n".join(lines), state, stop_reason="mbti_description")
    if action in {"start", "restart"}:
        if action == "start" and state is not None and state["status"] in {"active", "paused", "ready"}:
            action = "resume"
        else:
            page = call(QUESTION_TOOL, {"offset": 0, "limit": 12})
            state = validate_snapshot({
                "schema_version": 1, "test_id": uuid4().hex, "status": "active",
                "total": page.get("total"), "questionnaire_version": page.get("questionnaire_version"),
                "scoring_version": page.get("scoring_version"), "answers": {},
                "questions": page.get("questions"), "options": page.get("options"), "result": None,
            })
            return AssessmentReply(_question_text(state, NOTICE + "\n\n"), state, stop_reason="mbti_started")
    if state is None:
        return AssessmentReply("当前会话还没有 MBTI 测试，请先发送“开始 MBTI 测试”。", status="needs_clarification", stop_reason="mbti_not_started")
    if action == "cancel":
        state["status"] = "cancelled"
        return AssessmentReply("已退出 MBTI 测试。要重新作答，请发送“重新开始 MBTI 测试”。", state, stop_reason="mbti_cancelled")
    if action == "pause" and state["status"] in {"active", "paused", "ready"}:
        state["status"] = "paused"
        return AssessmentReply(f"已暂停，保留了 {len(state['answers'])}/{state['total']} 道答案。发送“继续 MBTI 测试”恢复。", state, stop_reason="mbti_paused")
    if state["status"] == "cancelled":
        return AssessmentReply("这次测试已退出。请发送“重新开始 MBTI 测试”创建新测试。", state, "needs_clarification", "mbti_cancelled")
    if state["status"] == "completed":
        return AssessmentReply(_result_text(state["result"]), state, stop_reason="mbti_completed")
    if action == "answer":
        if state["status"] != "active":
            return AssessmentReply("请先发送“继续 MBTI 测试”，再回答当前题目。", state, "needs_clarification", "mbti_not_active")
        text = question.strip()
        tagged = _ANSWER_RE.fullmatch(text)
        if text.casefold().startswith("mbti答题 "):
            if tagged is None or tagged.group(1).lower() != state["test_id"] or int(tagged.group(2)) != len(state["answers"]):
                return AssessmentReply(_question_text(state, "这是旧题目或重复提交，未修改答案。\n\n"), state, "needs_clarification", "mbti_stale_answer")
            score = int(tagged.group(3))
        else:
            match = _SCORE_RE.fullmatch(text)
            score = int(match.group(1)) if match else _LABELS.get(text)
        if score is None or len(state["answers"]) >= len(state["questions"]):
            return AssessmentReply("请选择当前题目的 1—5 分，或发送“继续 MBTI 测试”加载题目。", state, "needs_clarification", "mbti_invalid_answer")
        state["answers"][str(len(state["answers"]))] = score
    elif action == "resume":
        state["status"] = "active"
    if len(state["answers"]) == state["total"]:
        state["status"] = "ready"
        try:
            scored = call(SCORE_TOOL, {"answers": state["answers"], "questionnaire_version": state["questionnaire_version"]})
            _validate_result(scored, state)
        except (AssessmentToolError, ValueError):
            return AssessmentReply("48 道答案已收集，评分暂未完成。请稍后发送“查看 MBTI 结果”；若题库版本发生变化，需要重新测试。", state, "needs_clarification", "mbti_scoring_pending")
        state["result"] = scored
        state["status"] = "completed"
        return AssessmentReply(_result_text(scored), state, stop_reason="mbti_completed")
    if action == "result":
        return AssessmentReply(_question_text(state, f"当前已答 {len(state['answers'])}/{state['total']} 题，请完成剩余题目。\n\n"), state, "needs_clarification", "mbti_incomplete")
    if len(state["answers"]) == len(state["questions"]):
        try:
            _load_page(state, call)
        except (AssessmentToolError, ValueError):
            # Persist the just-submitted answer, but do not invent the next question.
            state["questions"] = state["questions"][:len(state["answers"])]
            return AssessmentReply(f"已记录 {len(state['answers'])}/{state['total']} 道答案，下一批题目暂不可用。请稍后发送“继续 MBTI 测试”；若题库更新，请重新测试。", state, "needs_clarification", "mbti_page_pending")
    return AssessmentReply(_question_text(state), validate_snapshot(state))
