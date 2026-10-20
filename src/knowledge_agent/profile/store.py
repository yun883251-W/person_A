"""独立SQLite文件，短连接与乐观版本校验；固定本机用户，不提供任意user_id访问。"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .models import validate_fields


class ProfileConflictError(ValueError):
    pass


def profile_path() -> Path:
    from knowledge_agent.config import get_session_store_settings
    session_path = get_session_store_settings().db_path
    return session_path.with_name(session_path.stem + ".profile.db")


def _empty() -> dict:
    return {"user_id": "local", "enabled": True, "fields": {}, "revision": 0,
            "updated_at": None, "source": "manual", "field_metadata": {}}


def _decode(row: tuple | None, metadata: dict | None = None) -> dict:
    if row is None:
        return _empty()
    fields = validate_fields(json.loads(row[1]))
    return {"user_id": "local", "enabled": bool(row[0]),
            "fields": fields, "revision": row[2],
            "updated_at": row[3], "source": "manual",
            # 旧记录只有整份画像时间，不能冒充字段更新时间。
            "field_metadata": {key: (metadata or {}).get(key, {
                "source": "legacy_manual", "updated_at": None,
            }) for key in fields}}


def _metadata(db) -> dict:
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='profile_field_metadata'").fetchone() is None:
        return {}
    return {key: {"source": source, "updated_at": updated}
            for key, source, updated in db.execute(
                "SELECT field_name, source, updated_at FROM profile_field_metadata WHERE user_id='local'")}


def load_profile(path: Path | None = None) -> dict:
    path = path or profile_path()
    if not path.exists():
        return _empty()
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
        db.execute("BEGIN")
        # 首次写入的建表事务尚未提交时，读取空画像即可。
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='learning_profile'").fetchone() is None:
            return _empty()
        return _decode(db.execute("SELECT enabled, fields_json, revision, updated_at FROM learning_profile WHERE user_id='local'").fetchone(), _metadata(db))


def save_profile(fields: dict, enabled: bool, expected_revision: int, path: Path | None = None,
                 *, confirmed_fields: tuple[str, ...] | list[str] = ()) -> dict:
    fields = validate_fields(fields)
    if set(confirmed_fields) - set(fields):
        raise ValueError("只能确认非空画像字段")
    if type(enabled) is not bool or type(expected_revision) is not int or not 0 <= expected_revision <= 9223372036854775806:
        raise ValueError("画像开关或版本无效")
    path = path or profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=5)) as db:
        with db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS learning_profile (user_id TEXT PRIMARY KEY, enabled INTEGER NOT NULL, fields_json TEXT NOT NULL, revision INTEGER NOT NULL, updated_at TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS profile_field_metadata (user_id TEXT NOT NULL, field_name TEXT NOT NULL, source TEXT NOT NULL, updated_at TEXT, PRIMARY KEY(user_id, field_name))")
            old = _decode(db.execute("SELECT enabled, fields_json, revision, updated_at FROM learning_profile WHERE user_id='local'").fetchone(), _metadata(db))
            if old["revision"] != expected_revision:
                raise ProfileConflictError("画像已在其他页面更新，请重新加载后再修改")
            updated = datetime.now(timezone.utc).isoformat()
            revision = expected_revision + 1
            metadata = {}
            for key, value in fields.items():
                metadata[key] = (
                    {"source": "manual", "updated_at": updated}
                    if old["fields"].get(key) != value or key in confirmed_fields
                    else old["field_metadata"][key]
                )
            # 删除字段时同步删除元信息；无历史副本，开关或无关字段修改不刷新时间。
            db.execute("DELETE FROM profile_field_metadata WHERE user_id='local'")
            db.executemany("INSERT INTO profile_field_metadata VALUES (?, ?, ?, ?)", [
                ("local", key, value["source"], value["updated_at"]) for key, value in metadata.items()
            ])
            db.execute("INSERT OR REPLACE INTO learning_profile VALUES (?, ?, ?, ?, ?)",
                       ("local", int(enabled), json.dumps(fields, ensure_ascii=False), revision, updated))
    return {"user_id": "local", "enabled": enabled, "fields": fields,
            "revision": revision, "updated_at": updated, "source": "manual",
            "field_metadata": metadata}
