"""画像白名单与输入校验；仅保存用户手动设置的字段。"""
from __future__ import annotations

FIELDS = {
    "python_level": 100, "learning_goal": 400, "preferred_style": 300,
    "development_environment": 300, "learning_progress": 400, "preferred_language": 80,
}


def validate_fields(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) - set(FIELDS):
        raise ValueError("画像包含不支持的字段")
    result = {}
    for key, text in value.items():
        if type(text) is not str or len(text) > FIELDS[key]:
            raise ValueError(f"{key} 超过长度限制或不是文本")
        if any(ord(c) < 32 and c not in "\n\t" or 0x7f <= ord(c) <= 0x9f or 0xd800 <= ord(c) <= 0xdfff for c in text):
            raise ValueError("画像不能包含控制字符")
        if text.strip():
            result[key] = text.strip()
    return result
