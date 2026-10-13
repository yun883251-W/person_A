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
_TARGET = r"(?:mbti(?:人格|性格)?(?:测试|自评|问卷)?|(?:人格|性格)(?:测试|自评|问卷))"
_TOOL_TOPIC_RE = re.compile(r"mbti|人格测试|性格测试|\bmcp\b|工具|mcp__|(?:score_mbti_answers|get_mbti_\w+)", re.I)
_TEACHING_RE = re.compile(
    r"怎么|如何|怎样|为什么|是什么|什么是|教我|原理|算法|代码|源码|编程|实现|开发|设计|"
    r"功能|按钮|接口|函数|教程|示例|例子|样例|\b(?:how|why|what|tutorial|example|implement|code)\b", re.I,
)
_NEGATED_CONTROL_RE = re.compile(
    r"(?:不要|别|不用|不想|不需要|无需|不许|先不|暂不|没有要求|不是要).{0,20}"
    r"(?:开始|做|测|重做|重测|继续|恢复|暂停|退出|取消|查看|调用|执行|运行)|"
    r"\b(?:do\s+not|don't|never)\s+(?:start|run|call|execute|resume|cancel)\b", re.I,
)
_QUOTED_RE = re.compile(r"```[\s\S]*?```|`[^`]*`|“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』|\"[^\"]*\"")
_CLAUSE_RE = re.compile(
    r"[，,；;。\n]+|然后|之后|随后|接着|同时|另外|并且|"
    r"(?:并|和|后|再)(?=(?:帮我|请|给我|解释|介绍|比较|开始))"
)
_SELF_DESCRIPTION_RE = re.compile(
    r"(?:介绍|解释|解读|描述|说说|了解)(?:一下|下)?(?:我的|我这次的|本次的?)"
    r"(?:mbti)?(?:测试)?(?:人格|性格)?(?:类型|结果)(?:的特点)?", re.I,
)
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


def _command_text(question: str) -> str:
    text = re.sub(r"\s+", "", question.strip()).casefold().strip("。.!！?？")
    text = re.sub(r"^(?:(?:请|麻烦|帮我|给我|我想要|我想|我要|能不能|可以|先|再))*", "", text)
    return re.sub(r"(?:吧|好吗|可以吗|行吗|吗)$", "", text)


def _control_action(question: str, has_state: bool) -> str | None:
    """Match a complete command, never an arbitrary action word in a sentence."""
    text = _command_text(question)
    if re.fullmatch(rf"(?:重新(?:开始|做|测)?|重做|重测)(?:一次|一个|一份)?{_TARGET}", text):
        return "restart"
    if re.fullmatch(rf"(?:(?:开始(?:做|进行)?|开启|做(?:个|一下|一个)?|测(?:一下)?|进行)(?:一次|一个|一份)?)?{_TARGET}", text):
        return "start"
    for action, verbs in (("cancel", r"退出|结束|取消"), ("pause", r"暂停|先停一下"),
                          ("resume", r"继续|恢复")):
        if re.fullmatch(rf"(?:{verbs}){_TARGET}", text):
            return action
        if has_state and re.fullmatch(rf"(?:{verbs})(?:测试|答题)?", text):
            return action
    if has_state and text == "下一题":
        return "resume"
    if re.fullmatch(rf"(?:查看|看看|查询|显示)?(?:我的|本次)?{_TARGET}(?:结果|得分|评分)", text):
        return "result"
    if has_state and re.fullmatch(r"(?:查看|看看|查询|显示)?(?:我的|本次)?(?:测试)?(?:结果|得分|评分)", text):
        return "result"
    return None


def _is_description_request(question: str) -> bool:
    text = _command_text(question)
    if _SELF_DESCRIPTION_RE.fullmatch(text):
        return True
    code = r"[ei][sn][tf][jp]"
    return re.fullmatch(
        rf"(?:(?:介绍|解释|了解|描述|解读|说说)(?:一下|下)?(?:mbti)?{code}"
        r"(?:的)?(?:人格|类型)?(?:的)?(?:特点|特征|性格|结果|优势|优缺点|含义)?|"
        rf"{code}(?:是什么(?:类型)?|有什么特点|有哪些特点|有什么优缺点|的特点|的含义))",
        text,
    ) is not None


def is_tool_instruction_question(question: str) -> bool:
    """Whether this asks about implementing/using tools instead of running them.

    The planner can reuse this guard. A separate explicit MBTI command in another
    clause makes the request mixed; request_clarification handles that boundary.
    Quoted command examples cannot become executable clauses.
    """
    for quoted in _QUOTED_RE.finditer(question):
        fragment = quoted.group(0)
        inner = fragment[1:-1].strip()
        # A quoted parameter ("INTJ") or formatted tool identifier is not an
        # instructional example. A quoted whole command must never execute.
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", inner):
            continue
        if (
            _control_action(inner, False) is not None
            or re.search(r"调用|执行|运行|\b(?:run|call|execute|invoke)\b", inner, re.I)
            or fragment.startswith("```") and _TOOL_TOPIC_RE.search(question)
        ):
            return True
    has_tool_topic = _TOOL_TOPIC_RE.search(question) or re.search(
        r"调用|执行|运行|\b(?:run|call|execute|invoke)\b", question, re.I,
    )
    if not has_tool_topic or not _TEACHING_RE.search(question):
        return False
    unquoted = _QUOTED_RE.sub("", question)
    for part in _CLAUSE_RE.split(unquoted):
        if not part.strip():
            continue
        if _control_action(part, False) is not None:
            return False
        if not _TEACHING_RE.search(part) and re.match(
            r"^(?:(?:请|麻烦|帮我|先|再)\s*)*(?:调用|执行|运行|(?:call|run|execute|invoke)\b)",
            part.strip(), re.I,
        ):
            return False
    return True


def request_clarification(question: str, state: Mapping[str, Any] | None) -> str | None:
    """Return a fixed clarification for requests that must not advance the test."""
    text = question.strip()
    has_state = state is not None
    if is_tool_instruction_question(text):
        return None
    if _NEGATED_CONTROL_RE.search(text) and (
        _TOOL_TOPIC_RE.search(text) or has_state or re.search(r"调用|执行|运行|\b(?:run|call|execute)\b", text, re.I)
    ):
        return "本轮没有执行测试或工具操作。请直接说明你想了解的内容，或单独发送需要执行的操作。"
    unquoted = _QUOTED_RE.sub("", text)
    parts = tuple(part.strip() for part in _CLAUSE_RE.split(unquoted) if part.strip())
    # An interactive questionnaire cannot finish as one step of a compound turn.
    # Ask explicitly instead of silently dropping the user's remaining objectives.
    if len(parts) > 1 and any(
        _control_action(part, has_state) is not None or _is_description_request(part)
        for part in parts
    ):
        return "这条消息同时包含 MBTI 操作和其他要求。请将测试操作单独发送；知识问题可以另发，已有测试进度会保留。"
    if (state is None or state.get("status") not in {"active", "paused", "ready"}) and (
        _SCORE_RE.fullmatch(text) or text in _LABELS
    ):
        return "当前会话没有可作答的 MBTI 测试。请先发送“开始 MBTI 测试”，或补充这个数字所指的问题。"
    return None


def _description_type(question: str, state: Mapping[str, Any] | None) -> str | None:
    match = _TYPE_RE.search(question)
    if match is not None:
        return match.group(1).upper()
    if state is not None and state.get("status") == "completed":
        result = state.get("result")
        code = result.get("type") if isinstance(result, Mapping) else None
        if isinstance(code, str) and _TYPE_RE.fullmatch(code):
            return code.upper()
    return None


def request_action(question: str, state: Mapping[str, Any] | None) -> str | None:
    """Recognize complete controls without advancing state or guessing personality.

    A knowledge-topic switch returns None and leaves the snapshot untouched.
    Numeric replies in paused/ready states reach handle_request's explicit guard;
    they never implicitly resume a test or overwrite a completed result.
    """
    text = question.strip()
    ongoing = state is not None and state.get("status") in {"active", "paused", "ready"}
    if text.casefold().startswith("mbti答题 "):
        return "answer"
    if request_clarification(text, state) is not None or is_tool_instruction_question(text):
        return None
    if _QUOTED_RE.search(text):
        return None
    if _is_description_request(text):
        return "describe"
    action = _control_action(text, state is not None)
    if action is not None:
        return action
    if ongoing and (_SCORE_RE.fullmatch(text) or text in _LABELS):
        return "answer"
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
        type_code = _description_type(question, state)
        if type_code is None:
            return AssessmentReply(
                "当前会话还没有已完成的测试结果。请先完成测试，或提供要了解的四字母类型，例如 INTJ。",
                state, "needs_clarification", "mbti_description_without_result",
            )
        response = call(DESCRIPTION_TOOL, {"type_code": type_code})
        description = response.get("description")
        if not isinstance(description, dict):
            raise AssessmentToolError("mbti_invalid_description")
        lines = [str(description.get("name") or type_code)]
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
