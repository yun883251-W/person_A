"""SQLite 会话存储模块（待开发）。

模块功能：
1. 在本地保存会话信息和完整问答轮，程序退出后仍可保留聊天记录。
2. 按会话 ID 恢复历史，支持查看会话列表、分页读取历史、清空和删除会话。
3. 保存问题、实际检索问题、必要的模型与工具消息、回答、引用和结果状态。
4. 为 Agent 提供完整轮次的存取接口，为 CLI 提供会话管理所需的数据。

存储方案：Python sqlite3 + SQLite，默认数据库位置为：
PROJECT_ROOT / "data/sessions/chat.db"。
规划 sessions（会话信息）和 turns（完整问答轮）两张表。
数据库由后续显式初始化操作创建；导入模块时不连接数据库或创建目录。

职责边界：本模块只管理会话数据，模型调用、检索及来源校验仍由 Agent 负责。
知识库向量索引继续存放在 data/index；会话数据库单独放在 data/sessions。
数据库保留已提交的完整历史，送入模型的最近轮数由 AgentConfig.max_history_turns 控制。

当前仅包含功能说明和 TODO，没有数据库操作、业务实现或测试结果。
由用户按 TODO 01—15 开发，再集中执行 TODO 16—19 的验证。
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from .schemas import AnswerResult, AnswerSource, RouteSummary
from .agent.trace import trace_from_dict, trace_to_dict


# 第一阶段：存储层及应用接入开发。

DATABASE_SCHEMA_VERSION = 1
MESSAGE_SERIALIZATION_VERSION = 1


class SessionStoreError(RuntimeError):
    """Base class for safe, application-facing persistence failures."""

    operation = "storage"
    retryable = False


class SessionStorePathError(SessionStoreError):
    operation = "open"


class SessionStoreLockedError(SessionStoreError):
    operation = "write"
    retryable = True


class SessionStoreSchemaError(SessionStoreError):
    operation = "open"


class SessionStoreCorruptionError(SessionStoreError):
    operation = "read"


class SessionStoreReadError(SessionStoreError):
    operation = "read"


class SessionStoreWriteError(SessionStoreError):
    operation = "write"


class SessionNotFoundError(SessionStoreError):
    operation = "lookup"


class DuplicateSessionError(SessionStoreError):
    operation = "create"


class DuplicateTurnError(SessionStoreError):
    operation = "write"


def _storage_error(
    exc: BaseException, *, operation: str, write: bool = False
) -> SessionStoreError:
    """Translate low-level failures without exposing paths or stored content."""
    if isinstance(exc, SessionStoreError):
        return exc
    if isinstance(exc, OSError):
        return SessionStorePathError("Session storage path is not writable")

    code = getattr(exc, "sqlite_errorcode", None)
    primary_code = code & 0xFF if isinstance(code, int) else None
    message = str(exc).lower()
    if primary_code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or any(
        marker in message for marker in ("database is locked", "database table is locked")
    ):
        detail = "changes were not saved; " if write else ""
        return SessionStoreLockedError(
            f"Session storage is busy; {detail}the operation may be retried"
        )
    if primary_code in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB) or any(
        marker in message for marker in ("malformed", "not a database")
    ):
        return SessionStoreCorruptionError("Session storage is corrupt")
    if primary_code in (
        sqlite3.SQLITE_CANTOPEN,
        sqlite3.SQLITE_READONLY,
        sqlite3.SQLITE_PERM,
    ):
        return SessionStorePathError("Session storage path is not writable")
    if write:
        return SessionStoreWriteError(
            f"Session storage {operation} failed; changes were not saved"
        )
    return SessionStoreReadError(f"Session storage {operation} failed")


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Persisted session metadata."""

    session_id: str
    title: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class StoredTurn:
    """One complete conversation turn restored from durable storage."""

    turn_id: str
    turn_no: int
    question: str
    retrieval_question: str
    messages: tuple[BaseMessage, ...]
    result: AnswerResult


@runtime_checkable
class SessionStore(Protocol):
    """Persistence boundary shared by the Agent and CLI.

    Implementations manage session metadata and atomically committed, complete
    turns only.  They must not invoke models, embeddings, or retrieval services.
    """

    def initialize(self) -> None:
        """Explicitly prepare the backing store."""
        ...

    def create_session(
        self, session_id: str | None = None, title: str | None = None
    ) -> SessionRecord:
        """Create a session, raising a conflict if ``session_id`` exists."""
        ...

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Return a session, or ``None`` when it does not exist."""
        ...

    def list_sessions(
        self, *, limit: int = 50, offset: int = 0
    ) -> Sequence[SessionRecord]:
        """Return a page of session metadata."""
        ...

    def append_turn(self, session_id: str, turn: StoredTurn) -> StoredTurn:
        """Atomically persist one complete turn and return its stored form."""
        ...

    def load_turns(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[StoredTurn]:
        """Load complete turns for a session in chronological order."""
        ...

    def load_context_turns(
        self, session_id: str, *, max_history_turns: int
    ) -> Sequence[StoredTurn]:
        """Load the most recent complete turns in chronological order."""
        ...

    def load_latest_assessment(self, session_id: str) -> dict[str, Any] | None:
        """Restore the latest assessment snapshot, independent of chat limits."""
        ...

    def clear_session(self, session_id: str) -> bool:
        """Delete a session's turns while retaining its metadata."""
        ...

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all of its turns."""
        ...

    def close(self) -> None:
        """Release owned resources; repeated calls must be safe."""
        ...

class SQLiteSessionStore:
    """Own one lazily opened SQLite connection for synchronous use."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        lock_timeout: float = 5.0,
        max_write_retries: int = 2,
    ) -> None:
        if not math.isfinite(lock_timeout) or lock_timeout <= 0:
            raise ValueError("lock_timeout must be a finite positive number")
        if (
            isinstance(max_write_retries, bool)
            or not isinstance(max_write_retries, int)
            or max_write_retries < 0
        ):
            raise ValueError("max_write_retries must be a non-negative integer")

        self.db_path = Path(db_path)
        self.lock_timeout = float(lock_timeout)
        self.max_write_retries = max_write_retries
        self._connection: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Open and configure the database connection, if necessary."""
        if self._connection is not None:
            return

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.db_path,
                timeout=self.lock_timeout,
                isolation_level=None,
            )
        except (OSError, sqlite3.Error) as exc:
            raise _storage_error(exc, operation="open", write=True) from exc
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
            if schema_version == 0:
                connection.execute(
                    """
                    CREATE TABLE sessions (
                        session_id TEXT PRIMARY KEY,
                        title TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE turns (
                        turn_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        turn_no INTEGER NOT NULL CHECK (turn_no > 0),
                        question TEXT NOT NULL,
                        retrieval_question TEXT NOT NULL,
                        messages_json TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        serialization_version INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                            ON DELETE CASCADE,
                        UNIQUE (session_id, turn_no)
                    )
                    """
                )
                connection.execute(
                    f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}"
                )
            elif schema_version != DATABASE_SCHEMA_VERSION:
                raise SessionStoreSchemaError(
                    "Unsupported session database schema version: "
                    f"expected {DATABASE_SCHEMA_VERSION}, got {schema_version}"
                )
            connection.commit()
        except BaseException as exc:
            if connection.in_transaction:
                connection.rollback()
            connection.close()
            if isinstance(exc, SessionStoreError):
                raise
            if isinstance(exc, (OSError, sqlite3.Error)):
                raise _storage_error(exc, operation="initialize", write=True) from exc
            raise
        self._connection = connection

    def _require_connection(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is None:
            raise RuntimeError("SQLiteSessionStore is not initialized")
        return connection

    @contextmanager
    def _transaction(
        self, *, write: bool = False, operation: str = "transaction"
    ) -> Iterator[sqlite3.Connection]:
        """Run a transaction with explicit commit and exception rollback."""
        connection = self._require_connection()
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        except sqlite3.Error as exc:
            raise _storage_error(exc, operation=operation, write=write) from exc
        try:
            yield connection
        except BaseException as exc:
            connection.rollback()
            if isinstance(exc, SessionStoreError):
                raise
            if isinstance(exc, sqlite3.Error):
                raise _storage_error(exc, operation=operation, write=write) from exc
            raise
        else:
            try:
                connection.commit()
            except BaseException as exc:
                connection.rollback()
                if isinstance(exc, sqlite3.Error):
                    raise _storage_error(exc, operation=operation, write=write) from exc
                raise

    def close(self) -> None:
        """Close the owned connection; repeated calls are safe."""
        connection = self._connection
        if connection is None:
            return

        self._connection = None
        try:
            if connection.in_transaction:
                connection.rollback()
        finally:
            connection.close()

    def __enter__(self) -> SQLiteSessionStore:
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def create_session(
        self, session_id: str | None = None, title: str | None = None
    ) -> SessionRecord:
        """Persist metadata for a new, empty session."""
        if session_id is None:
            session_id = uuid.uuid4().hex
        elif not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if title is not None and not isinstance(title, str):
            raise ValueError("title must be a string or None")

        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            with self._transaction(write=True, operation="create") as connection:
                connection.execute(
                    """
                    INSERT INTO sessions (session_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, title, timestamp, timestamp),
                )
        except SessionStoreWriteError as exc:
            cause = exc.__cause__
            if isinstance(cause, sqlite3.IntegrityError):
                raise DuplicateSessionError("Session ID already exists") from exc
            raise
        return SessionRecord(session_id, title, timestamp, timestamp)

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Return metadata for exactly one session ID."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        try:
            row = self._require_connection().execute(
                """
                SELECT session_id, title, created_at, updated_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise _storage_error(exc, operation="lookup") from exc
        if row is None:
            return None
        return SessionRecord(
            session_id=row["session_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_sessions(
        self, *, limit: int = 50, offset: int = 0
    ) -> Sequence[SessionRecord]:
        """Return session metadata ordered from most recently updated."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        try:
            rows = self._require_connection().execute(
                """
                SELECT session_id, title, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC, session_id ASC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        except sqlite3.Error as exc:
            raise _storage_error(exc, operation="list") from exc
        return tuple(
            SessionRecord(
                session_id=row["session_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        )

    def append_turn(self, session_id: str, turn: StoredTurn) -> StoredTurn:
        """Atomically append a complete turn, making retries idempotent."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(turn, StoredTurn):
            raise TypeError("turn must be a StoredTurn")
        if not isinstance(turn.turn_id, str) or not turn.turn_id.strip():
            raise ValueError("turn.turn_id must be a non-empty string")
        if not isinstance(turn.question, str):
            raise ValueError("turn.question must be a string")
        if not isinstance(turn.retrieval_question, str):
            raise ValueError("turn.retrieval_question must be a string")

        messages_json = _serialize_messages(turn.messages)
        result_json = _serialize_result(turn.result)
        for attempt in range(self.max_write_retries + 1):
            try:
                return self._append_turn_once(
                    session_id, turn, messages_json=messages_json, result_json=result_json
                )
            except SessionStoreLockedError:
                if attempt >= self.max_write_retries:
                    raise
                time.sleep(min(0.05 * (2**attempt), 0.2))
        raise AssertionError("unreachable")

    def _append_turn_once(
        self,
        session_id: str,
        turn: StoredTurn,
        *,
        messages_json: str,
        result_json: str,
    ) -> StoredTurn:
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._transaction(write=True, operation="save") as connection:
            existing = connection.execute(
                """
                SELECT turn_id, session_id, turn_no, question,
                       retrieval_question, messages_json, result_json,
                       serialization_version
                FROM turns
                WHERE turn_id = ?
                """,
                (turn.turn_id,),
            ).fetchone()
            if existing is not None:
                if existing["session_id"] != session_id:
                    raise DuplicateTurnError("Turn ID already belongs to another session")
                if (
                    existing["question"] != turn.question
                    or existing["retrieval_question"] != turn.retrieval_question
                    or existing["messages_json"] != messages_json
                    or existing["result_json"] != result_json
                    or existing["serialization_version"]
                    != MESSAGE_SERIALIZATION_VERSION
                ):
                    raise DuplicateTurnError("Turn ID already exists with different content")
                return StoredTurn(
                    turn_id=existing["turn_id"],
                    turn_no=existing["turn_no"],
                    question=existing["question"],
                    retrieval_question=existing["retrieval_question"],
                    messages=_deserialize_messages(existing["messages_json"]),
                    result=_deserialize_result(existing["result_json"]),
                )

            next_turn_no = connection.execute(
                """
                SELECT CASE
                           WHEN EXISTS (
                               SELECT 1 FROM sessions WHERE session_id = ?
                           )
                           THEN COALESCE((
                               SELECT MAX(turn_no) + 1
                               FROM turns
                               WHERE session_id = ?
                           ), 1)
                       END
                """,
                (session_id, session_id),
            ).fetchone()[0]
            if next_turn_no is None:
                raise SessionNotFoundError("Session does not exist")

            connection.execute(
                """
                INSERT INTO turns (
                    turn_id, session_id, turn_no, question,
                    retrieval_question, messages_json, result_json,
                    serialization_version, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.turn_id,
                    session_id,
                    next_turn_no,
                    turn.question,
                    turn.retrieval_question,
                    messages_json,
                    result_json,
                    MESSAGE_SERIALIZATION_VERSION,
                    timestamp,
                ),
            )
            connection.execute(
                """
                UPDATE sessions
                SET updated_at = ?
                WHERE session_id = ?
                """,
                (timestamp, session_id),
            )

        return StoredTurn(
            turn_id=turn.turn_id,
            turn_no=next_turn_no,
            question=turn.question,
            retrieval_question=turn.retrieval_question,
            messages=_deserialize_messages(messages_json),
            result=_deserialize_result(result_json),
        )

    def load_turns(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[StoredTurn]:
        """Load a chronological page of all committed turns for display."""
        self._validate_load_arguments(session_id, limit=limit, offset=offset)
        connection = self._require_connection()
        try:
            self._require_session(connection, session_id)
            if limit is None:
                rows = connection.execute(
                    """
                    SELECT turn_id, turn_no, question, retrieval_question,
                           messages_json, result_json, serialization_version
                    FROM turns
                    WHERE session_id = ?
                    ORDER BY turn_no ASC
                    LIMIT -1 OFFSET ?
                    """,
                    (session_id, offset),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT turn_id, turn_no, question, retrieval_question,
                           messages_json, result_json, serialization_version
                    FROM turns
                    WHERE session_id = ?
                    ORDER BY turn_no ASC
                    LIMIT ? OFFSET ?
                    """,
                    (session_id, limit, offset),
                ).fetchall()
        except SessionStoreError:
            raise
        except sqlite3.Error as exc:
            raise _storage_error(exc, operation="load") from exc
        return tuple(self._restore_turn(row) for row in rows)

    def load_context_turns(
        self, session_id: str, *, max_history_turns: int
    ) -> Sequence[StoredTurn]:
        """Load only the newest complete turns, ordered oldest to newest."""
        if (
            isinstance(max_history_turns, bool)
            or not isinstance(max_history_turns, int)
            or max_history_turns <= 0
        ):
            raise ValueError("max_history_turns must be a positive integer")
        connection = self._require_connection()
        try:
            self._require_session(connection, session_id)
            rows = connection.execute(
                """
                SELECT turn_id, turn_no, question, retrieval_question,
                       messages_json, result_json, serialization_version
                FROM (
                    SELECT turn_id, turn_no, question, retrieval_question,
                           messages_json, result_json, serialization_version
                    FROM turns
                    WHERE session_id = ?
                    ORDER BY turn_no DESC
                    LIMIT ?
                )
                ORDER BY turn_no ASC
                """,
                (session_id, max_history_turns),
            ).fetchall()
        except SessionStoreError:
            raise
        except sqlite3.Error as exc:
            raise _storage_error(exc, operation="load") from exc
        return tuple(self._restore_turn(row) for row in rows)

    def load_latest_assessment(self, session_id: str) -> dict[str, Any] | None:
        """Restore the newest committed snapshot for this session only.

        Assessment state shares the turn transaction with the answer. Searching
        separately from the bounded model history lets a paused test survive
        later knowledge questions without another table or schema migration.
        Corrupt snapshots must fail explicitly instead of restarting the test or
        reviving an older snapshot whose answers may already have been changed.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        connection = self._require_connection()
        try:
            self._require_session(connection, session_id)
            rows = connection.execute(
                """
                SELECT result_json, serialization_version
                FROM turns
                WHERE session_id = ? AND result_json LIKE ?
                ORDER BY turn_no DESC
                """,
                (session_id, '%"assessment_state":%'),
            )
            try:
                for row in rows:
                    payload = _load_json_object(row["result_json"], "result_json")
                    # LIKE is only a candidate filter; an unrelated nested field
                    # must never be interpreted as the top-level snapshot.
                    if "assessment_state" not in payload:
                        continue
                    if row["serialization_version"] != MESSAGE_SERIALIZATION_VERSION:
                        raise SessionStoreCorruptionError(
                            "Stored assessment uses an unsupported serialization version"
                        )
                    result = _deserialize_result(row["result_json"])
                    return deepcopy(result.assessment_state)
            finally:
                rows.close()
        except SessionStoreError:
            raise
        except sqlite3.Error as exc:
            raise _storage_error(exc, operation="load assessment") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionStoreCorruptionError(
                "Stored assessment data is corrupt"
            ) from exc
        return None

    def clear_session(self, session_id: str) -> bool:
        """Delete all turns for a session while retaining its metadata."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")

        with self._transaction(write=True, operation="clear") as connection:
            updated = connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (datetime.now(timezone.utc).isoformat(), session_id),
            )
            if updated.rowcount == 0:
                return False
            connection.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and rely on the foreign key cascade for its turns."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")

        with self._transaction(write=True, operation="delete") as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
        return cursor.rowcount > 0

    @staticmethod
    def _validate_load_arguments(
        session_id: str, *, limit: int | None, offset: int
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")

    @staticmethod
    def _require_session(connection: sqlite3.Connection, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        exists = connection.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if exists is None:
            raise SessionNotFoundError("Session does not exist")

    @staticmethod
    def _restore_turn(row: sqlite3.Row) -> StoredTurn:
        try:
            version = row["serialization_version"]
            if version != MESSAGE_SERIALIZATION_VERSION:
                raise SessionStoreCorruptionError(
                    "Stored turn uses an unsupported serialization version"
                )
            return StoredTurn(
                turn_id=row["turn_id"],
                turn_no=row["turn_no"],
                question=row["question"],
                retrieval_question=row["retrieval_question"],
                messages=_deserialize_messages(row["messages_json"]),
                result=_deserialize_result(row["result_json"]),
            )
        except SessionStoreError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionStoreCorruptionError("Stored turn data is corrupt") from exc

_ANSWER_STATUSES = frozenset(
    {
        "success",
        "insufficient_evidence",
        "needs_clarification",
        "out_of_scope",
        "service_error",
        "limit_reached",
    }
)


def _load_json_object(value: str, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a JSON string")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate field {key!r}")
            result[key] = item
        return result

    try:
        decoded = json.loads(value, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return decoded


def _require_fields(
    value: dict[str, Any], required: set[str], optional: set[str], label: str
) -> None:
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)!r}")
        if extra:
            details.append(f"unexpected {sorted(extra)!r}")
        raise ValueError(f"Invalid {label}: {', '.join(details)}")


def _require_string(value: Any, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        suffix = " a non-empty string" if nonempty else " a string"
        raise ValueError(f"{label} must be{suffix}")
    return value


def _validate_content(value: Any, label: str) -> str | list[Any]:
    if not isinstance(value, (str, list)):
        raise ValueError(f"{label} must be a string or list")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid JSON content") from exc
    return value


def _messages_to_json(messages: Sequence[BaseMessage]) -> str:
    """Serialize system-free, complete LangChain message history."""
    encoded: list[dict[str, Any]] = []
    requested: set[str] = set()
    pending: set[str] = set()

    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        if pending and not isinstance(message, ToolMessage):
            raise ValueError("Message history contains incomplete tool calls")
        content = _validate_content(message.content, "message content")
        message_id = message.id
        if message_id is not None:
            message_id = _require_string(message_id, "message id", nonempty=True)
        if isinstance(message, HumanMessage):
            item: dict[str, Any] = {"type": "human", "content": content}
            if message_id is not None:
                item["id"] = message_id
            encoded.append(item)
        elif isinstance(message, AIMessage):
            calls: list[dict[str, Any]] = []
            for call in message.tool_calls:
                call_id = _require_string(call.get("id"), "tool call id", nonempty=True)
                name = _require_string(call.get("name"), "tool call name", nonempty=True)
                args = call.get("args")
                if not isinstance(args, dict):
                    raise ValueError("tool call args must be an object")
                if call_id in requested:
                    raise ValueError(f"Duplicate tool call id: {call_id}")
                try:
                    json.dumps(args, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise ValueError("tool call args are not JSON serializable") from exc
                requested.add(call_id)
                pending.add(call_id)
                calls.append({"id": call_id, "name": name, "args": args})
            item = {"type": "ai", "content": content, "tool_calls": calls}
            if message_id is not None:
                item["id"] = message_id
            encoded.append(item)
        elif isinstance(message, ToolMessage):
            call_id = _require_string(
                message.tool_call_id, "tool result call id", nonempty=True
            )
            if call_id not in pending:
                raise ValueError(f"Unmatched tool result call id: {call_id}")
            pending.remove(call_id)
            item = {
                "type": "tool",
                "content": content,
                "tool_call_id": call_id,
                "status": message.status,
            }
            if message_id is not None:
                item["id"] = message_id
            if message.name is not None:
                item["name"] = _require_string(message.name, "tool result name")
            encoded.append(item)
        else:
            raise ValueError(f"Unsupported message type: {type(message).__name__}")

    if pending:
        raise ValueError("Message history contains incomplete tool calls")
    return json.dumps(
        {"version": MESSAGE_SERIALIZATION_VERSION, "messages": encoded},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _messages_from_json(value: str) -> tuple[BaseMessage, ...]:
    """Restore and validate complete LangChain message history."""
    root = _load_json_object(value, "messages_json")
    _require_fields(root, {"version", "messages"}, set(), "messages_json")
    if root["version"] != MESSAGE_SERIALIZATION_VERSION:
        raise ValueError(f"Unsupported message serialization version: {root['version']!r}")
    items = root["messages"]
    if not isinstance(items, list):
        raise ValueError("messages_json.messages must be a list")

    restored: list[BaseMessage] = []
    requested: set[str] = set()
    pending: set[str] = set()
    for index, item in enumerate(items):
        label = f"messages_json.messages[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        message_type = item.get("type")
        if pending and message_type != "tool":
            raise ValueError("Message history contains incomplete tool calls")
        message_id = item.get("id")
        if message_id is not None:
            message_id = _require_string(message_id, f"{label}.id", nonempty=True)
        if message_type == "human":
            _require_fields(item, {"type", "content"}, {"id"}, label)
            restored.append(
                HumanMessage(
                    content=_validate_content(item["content"], label), id=message_id
                )
            )
        elif message_type == "ai":
            _require_fields(item, {"type", "content", "tool_calls"}, {"id"}, label)
            calls = item["tool_calls"]
            if not isinstance(calls, list):
                raise ValueError(f"{label}.tool_calls must be a list")
            restored_calls: list[dict[str, Any]] = []
            for call_index, call in enumerate(calls):
                call_label = f"{label}.tool_calls[{call_index}]"
                if not isinstance(call, dict):
                    raise ValueError(f"{call_label} must be an object")
                _require_fields(call, {"id", "name", "args"}, set(), call_label)
                call_id = _require_string(call["id"], f"{call_label}.id", nonempty=True)
                name = _require_string(call["name"], f"{call_label}.name", nonempty=True)
                if not isinstance(call["args"], dict):
                    raise ValueError(f"{call_label}.args must be an object")
                if call_id in requested:
                    raise ValueError(f"Duplicate tool call id: {call_id}")
                requested.add(call_id)
                pending.add(call_id)
                restored_calls.append(
                    {"id": call_id, "name": name, "args": call["args"], "type": "tool_call"}
                )
            restored.append(
                AIMessage(
                    content=_validate_content(item["content"], label),
                    tool_calls=restored_calls,
                    id=message_id,
                )
            )
        elif message_type == "tool":
            _require_fields(
                item,
                {"type", "content", "tool_call_id", "status"},
                {"name", "id"},
                label,
            )
            call_id = _require_string(
                item["tool_call_id"], f"{label}.tool_call_id", nonempty=True
            )
            if call_id not in pending:
                raise ValueError(f"Unmatched tool result call id: {call_id}")
            status = item["status"]
            if status not in {"success", "error"}:
                raise ValueError(f"{label}.status must be 'success' or 'error'")
            name = item.get("name")
            if name is not None:
                name = _require_string(name, f"{label}.name")
            pending.remove(call_id)
            restored.append(
                ToolMessage(
                    content=_validate_content(item["content"], label),
                    tool_call_id=call_id,
                    name=name,
                    status=status,
                    id=message_id,
                )
            )
        else:
            raise ValueError(f"Unsupported message type in {label}: {message_type!r}")

    if pending:
        raise ValueError("Message history contains incomplete tool calls")
    return tuple(restored)


def _result_to_json(result: AnswerResult) -> str:
    """Serialize an answer and validated optional execution snapshots."""
    sources = [
        {
            "chunk_id": source.chunk_id,
            "source_path": source.source_path,
            "heading_path": list(source.heading_path),
            "start_line": source.start_line,
            "end_line": source.end_line,
            "content_hash": source.content_hash,
            "index_version": source.index_version,
            "retrieval_score": source.retrieval_score,
        }
        for source in result.sources
    ]
    payload = {
        "version": MESSAGE_SERIALIZATION_VERSION,
        "question": result.question,
        "answer": result.answer,
        "status": result.status,
        "sources": sources,
        "index_version": result.index_version,
        "model_calls": result.model_calls,
        "tool_calls": result.tool_calls,
        "stop_reason": result.stop_reason,
    }
    if result.trace is not None:
        payload["trace"] = trace_to_dict(result.trace)
    if result.routing is not None:
        payload["routing"] = {
            "route": result.routing.route,
            "intent": result.routing.intent,
            "decision_source": result.routing.decision_source,
            "reason_code": result.routing.reason_code,
        }
    if result.assessment_state is not None:
        from .agent.assessment import validate_snapshot

        payload["assessment_state"] = validate_snapshot(result.assessment_state)
    if result.plan is not None:
        from .agent.planner import validate_plan_summary

        payload["plan"] = validate_plan_summary(result.plan)
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _result_from_json(value: str) -> AnswerResult:
    """Restore an AnswerResult after validating every persisted field."""
    root = _load_json_object(value, "result_json")
    fields = {
        "version",
        "question",
        "answer",
        "status",
        "sources",
        "index_version",
        "model_calls",
        "tool_calls",
        "stop_reason",
    }
    _require_fields(
        root, fields, {"trace", "routing", "assessment_state", "plan"}, "result_json"
    )
    if root["version"] != MESSAGE_SERIALIZATION_VERSION:
        raise ValueError(f"Unsupported result serialization version: {root['version']!r}")
    question = _require_string(root["question"], "result_json.question")
    answer = _require_string(root["answer"], "result_json.answer")
    status = root["status"]
    if not isinstance(status, str) or status not in _ANSWER_STATUSES:
        raise ValueError(f"Invalid answer status: {status!r}")
    index_version = root["index_version"]
    if index_version is not None:
        index_version = _require_string(index_version, "result_json.index_version")
    stop_reason = root["stop_reason"]
    if stop_reason is not None:
        stop_reason = _require_string(stop_reason, "result_json.stop_reason")
    for field in ("model_calls", "tool_calls"):
        count = root[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"result_json.{field} must be a non-negative integer")
    source_items = root["sources"]
    if not isinstance(source_items, list):
        raise ValueError("result_json.sources must be a list")

    sources: list[AnswerSource] = []
    source_fields = {
        "chunk_id",
        "source_path",
        "heading_path",
        "start_line",
        "end_line",
        "content_hash",
        "index_version",
        "retrieval_score",
    }
    for index, item in enumerate(source_items):
        label = f"result_json.sources[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        _require_fields(item, source_fields, set(), label)
        heading_path = item["heading_path"]
        if not isinstance(heading_path, list) or not all(
            isinstance(part, str) for part in heading_path
        ):
            raise ValueError(f"{label}.heading_path must be a list of strings")
        for field in ("start_line", "end_line"):
            number = item[field]
            if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                raise ValueError(f"{label}.{field} must be a positive integer")
        score = item["retrieval_score"]
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError(f"{label}.retrieval_score must be a finite number or null")
        sources.append(
            AnswerSource(
                chunk_id=_require_string(item["chunk_id"], f"{label}.chunk_id"),
                source_path=_require_string(item["source_path"], f"{label}.source_path"),
                heading_path=tuple(heading_path),
                start_line=item["start_line"],
                end_line=item["end_line"],
                content_hash=_require_string(item["content_hash"], f"{label}.content_hash"),
                index_version=_require_string(
                    item["index_version"], f"{label}.index_version"
                ),
                retrieval_score=float(score) if score is not None else None,
            )
        )
    trace = None
    if "trace" in root and root["trace"] is not None:
        try:
            trace = trace_from_dict(root["trace"])
        except (TypeError, ValueError):
            pass
    routing = None
    if "routing" in root and root["routing"] is not None:
        item = root["routing"]
        if not isinstance(item, dict):
            raise ValueError("result_json.routing must be an object")
        routing_fields = {"route", "intent", "decision_source", "reason_code"}
        _require_fields(item, routing_fields, set(), "result_json.routing")
        route = _require_string(item["route"], "result_json.routing.route")
        intent = _require_string(item["intent"], "result_json.routing.intent")
        decision_source = _require_string(
            item["decision_source"], "result_json.routing.decision_source"
        )
        if route not in {"knowledge", "mcp", "direct"}:
            raise ValueError("Invalid result_json.routing.route")
        if intent not in {
            "knowledge", "mcp_tool", "greeting", "scope",
            "needs_clarification", "out_of_scope",
        }:
            raise ValueError("Invalid result_json.routing.intent")
        if decision_source not in {"rules", "model", "fallback"}:
            raise ValueError("Invalid result_json.routing.decision_source")
        routing = RouteSummary(
            route=route,
            intent=intent,
            decision_source=decision_source,
            reason_code=_require_string(
                item["reason_code"], "result_json.routing.reason_code"
            ),
        )
    assessment_state = None
    if root.get("assessment_state") is not None:
        from .agent.assessment import validate_snapshot

        assessment_state = validate_snapshot(root["assessment_state"])
    plan = None
    if root.get("plan") is not None:
        from .agent.planner import validate_plan_summary

        plan = validate_plan_summary(root["plan"])
    return AnswerResult(
        question=question,
        answer=answer,
        status=status,
        sources=tuple(sources),
        index_version=index_version,
        model_calls=root["model_calls"],
        tool_calls=root["tool_calls"],
        stop_reason=stop_reason,
        trace=trace,
        routing=routing,
        assessment_state=assessment_state,
        plan=plan,
    )


# Stable helper names used by the persistence operations implemented below.
_serialize_messages = _messages_to_json
_deserialize_messages = _messages_from_json
_serialize_result = _result_to_json
_deserialize_result = _result_from_json
