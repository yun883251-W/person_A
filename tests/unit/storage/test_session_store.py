from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from knowledge_agent.application import KnowledgeAgentApplication
from knowledge_agent.schemas import AnswerResult
from knowledge_agent.session_store import (
    DATABASE_SCHEMA_VERSION,
    DuplicateSessionError,
    DuplicateTurnError,
    SQLiteSessionStore,
    SessionStoreLockedError,
    SessionStorePathError,
    SessionStoreSchemaError,
    SessionStoreWriteError,
    StoredTurn,
)


def _turn(turn_id: str, question: str = "问题") -> StoredTurn:
    answer = f"回答：{question}"
    return StoredTurn(
        turn_id=turn_id,
        turn_no=0,
        question=question,
        retrieval_question=question,
        messages=(HumanMessage(content=question), AIMessage(content=answer)),
        result=AnswerResult(question=question, answer=answer, status="success"),
    )


def test_initialize_and_crud_are_persistent_and_ordered(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "chat.db"
    store = SQLiteSessionStore(db_path)
    store.initialize()
    store.initialize()

    first = store.create_session("first", "第一个会话")
    second = store.create_session("second")
    assert store.get_session("first") == first
    assert {item.session_id for item in store.list_sessions()} == {"first", "second"}
    with pytest.raises(DuplicateSessionError):
        store.create_session("first")

    saved_first = store.append_turn("first", _turn("turn-1", "一"))
    saved_second = store.append_turn("first", _turn("turn-2", "二"))
    assert (saved_first.turn_no, saved_second.turn_no) == (1, 2)
    assert [turn.turn_id for turn in store.load_turns("first")] == [
        "turn-1",
        "turn-2",
    ]

    assert store.clear_session("first") is True
    assert store.load_turns("first") == ()
    store.append_turn("second", _turn("turn-3"))
    assert store.delete_session("second") is True
    assert store.get_session("second") is None
    assert store.delete_session("missing") is False

    store.close()
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 0


def test_turn_id_retries_are_idempotent_but_conflicts_fail(tmp_path: Path) -> None:
    with SQLiteSessionStore(tmp_path / "chat.db") as store:
        store.create_session("one")
        store.create_session("two")
        original = _turn("same-id", "原问题")

        first = store.append_turn("one", original)
        retried = store.append_turn("one", original)
        assert retried == first
        assert len(store.load_turns("one")) == 1

        with pytest.raises(DuplicateTurnError):
            store.append_turn("one", _turn("same-id", "不同问题"))
        with pytest.raises(DuplicateTurnError):
            store.append_turn("two", original)


def test_failed_append_rolls_back_the_whole_turn(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    with SQLiteSessionStore(db_path) as store:
        store.create_session("session")
        connection = store._require_connection()
        connection.execute(
            """
            CREATE TRIGGER fail_session_update
            BEFORE UPDATE OF updated_at ON sessions
            BEGIN
                SELECT RAISE(ABORT, 'injected failure');
            END
            """
        )

        with pytest.raises(SessionStoreWriteError, match="changes were not saved"):
            store.append_turn("session", _turn("rolled-back"))
        assert store.load_turns("session") == ()


def test_locked_database_is_reported_without_partial_write(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    with SQLiteSessionStore(db_path, lock_timeout=0.01, max_write_retries=0) as store:
        store.create_session("session")
        blocker = sqlite3.connect(db_path, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(SessionStoreLockedError):
                store.append_turn("session", _turn("blocked"))
        finally:
            blocker.rollback()
            blocker.close()
        assert store.load_turns("session") == ()


def test_invalid_path_and_schema_version_are_rejected(tmp_path: Path) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("x", encoding="utf-8")
    with pytest.raises(SessionStorePathError):
        SQLiteSessionStore(parent_file / "chat.db").initialize()

    incompatible = tmp_path / "incompatible.db"
    with sqlite3.connect(incompatible) as connection:
        connection.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION + 1}")
    with pytest.raises(SessionStoreSchemaError):
        SQLiteSessionStore(incompatible).initialize()


def test_application_restart_preserves_isolates_clears_and_deletes_sessions(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "chat.db"
    first_store = SQLiteSessionStore(db_path)
    first_app = KnowledgeAgentApplication(
        model=object(), retriever=object(), session_store=first_store
    )
    first_app.start_session("first")
    first_store.append_turn("first", _turn("first-1", "第一问"))
    first_store.append_turn("first", _turn("first-2", "继续追问"))
    first_app.end_session()
    assert [turn.turn_id for turn in first_store.load_turns("first")] == [
        "first-1",
        "first-2",
    ]
    first_app.close()

    second_store = SQLiteSessionStore(db_path)
    second_app = KnowledgeAgentApplication(
        model=object(), retriever=object(), session_store=second_store
    )
    assert second_app.resume_session("first") == "first"
    assert [turn.question for turn in second_app.get_session_history()] == [
        "第一问",
        "继续追问",
    ]
    second_app.end_session()
    second_app.resume_session("first")
    assert len(second_app.get_session_history()) == 2
    second_app.end_session()

    second_app.start_session("other")
    second_store.append_turn("other", _turn("other-1", "隔离问题"))
    second_app.end_session()
    second_app.resume_session("first")
    second_app.clear_session()
    second_app.end_session()
    assert second_store.load_turns("first") == ()
    assert [turn.turn_id for turn in second_store.load_turns("other")] == ["other-1"]
    assert second_app.delete_session("other") is True
    second_app.close()

    with SQLiteSessionStore(db_path) as final_store:
        assert final_store.load_turns("first") == ()
        assert final_store.get_session("other") is None
