from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from knowledge_agent.agent.dialogue.contracts import (
    BusinessFocus, BusinessPhase, BusinessType, SessionBusinessState,
)
from knowledge_agent.session.store import (
    BusinessStateConflictError, DATABASE_SCHEMA_VERSION, SQLiteSessionStore,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_version_two_database_migrates_without_rebuilding(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA user_version = 2;
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, title TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE turns (
            turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            turn_no INTEGER NOT NULL CHECK (turn_no > 0), question TEXT NOT NULL,
            retrieval_question TEXT NOT NULL, messages_json TEXT NOT NULL,
            result_json TEXT NOT NULL, serialization_version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            UNIQUE (session_id, turn_no)
        );
        CREATE TABLE session_summaries (
            session_id TEXT PRIMARY KEY, summary TEXT NOT NULL,
            covered_turn_id TEXT NOT NULL, covered_turn_no INTEGER NOT NULL,
            summary_version INTEGER NOT NULL, updated_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (covered_turn_id) REFERENCES turns(turn_id) ON DELETE CASCADE
        );
        """
    )
    now = _timestamp()
    connection.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)", ("legacy", "kept", now, now))
    connection.commit()
    connection.close()

    store = SQLiteSessionStore(path)
    store.initialize()

    assert store.get_session("legacy").title == "kept"
    assert store.load_business_state("legacy") is None
    assert store._require_connection().execute("PRAGMA user_version").fetchone()[0] == DATABASE_SCHEMA_VERSION


def test_business_state_compare_and_set_rejects_stale_revision(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "chat.db")
    store.initialize()
    store.create_session("session")
    first = SessionBusinessState(
        session_id="session", revision=1,
        focus=BusinessFocus(BusinessType.KNOWLEDGE, BusinessPhase.ACTIVE),
        updated_at=_timestamp(),
    )
    store.save_business_state("session", first, expected_revision=0)
    stale = SessionBusinessState(
        session_id="session", revision=1,
        focus=BusinessFocus(BusinessType.KNOWLEDGE, BusinessPhase.ACTIVE),
        updated_at=_timestamp(),
    )
    with pytest.raises(BusinessStateConflictError):
        store.save_business_state("session", stale, expected_revision=0)
