"""SQLite persistence for TechNest conversation turns."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "technest_support.db"
)


def initialize_database(
    db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> bool:
    """Create the conversations table when it does not already exist."""
    path = Path(db_path)
    connection: sqlite3.Connection | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_question TEXT NOT NULL,
                assistant_response TEXT NOT NULL DEFAULT '',
                retrieved_sources TEXT NOT NULL DEFAULT '[]',
                escalation_status TEXT NOT NULL DEFAULT 'not_required',
                escalation_reason TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_session_id "
            "ON conversations(session_id)"
        )
        connection.commit()
        return True
    except (OSError, sqlite3.Error):
        return False
    finally:
        if connection is not None:
            connection.close()


def save_conversation(
    session_id: str,
    user_question: str,
    assistant_response: str,
    retrieved_sources: list[dict[str, Any]] | None = None,
    escalation_status: str = "not_required",
    escalation_reason: str = "",
    db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int | None:
    """Save one question/answer turn and return its row ID, or None on error."""
    if not initialize_database(db_path):
        return None

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.execute(
            """
            INSERT INTO conversations (
                session_id, timestamp, user_question, assistant_response,
                retrieved_sources, escalation_status, escalation_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                datetime.now(timezone.utc).isoformat(),
                user_question,
                assistant_response,
                json.dumps(retrieved_sources or [], ensure_ascii=False, default=str),
                escalation_status,
                escalation_reason,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return None
    finally:
        if connection is not None:
            connection.close()


def update_conversation(
    conversation_id: int,
    session_id: str,
    assistant_response: str,
    escalation_status: str,
    escalation_reason: str = "",
    db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> bool:
    """Update a pending escalation after a human has supplied the final reply."""
    if not initialize_database(db_path):
        return False

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.execute(
            """
            UPDATE conversations
            SET assistant_response = ?, escalation_status = ?, escalation_reason = ?
            WHERE id = ? AND session_id = ?
            """,
            (
                assistant_response,
                escalation_status,
                escalation_reason,
                conversation_id,
                session_id,
            ),
        )
        connection.commit()
        return cursor.rowcount == 1
    except (OSError, sqlite3.Error):
        return False
    finally:
        if connection is not None:
            connection.close()


def _row_to_record(row: tuple[Any, ...]) -> dict[str, Any]:
    """Convert an SQL row to a convenient Python record."""
    try:
        sources = json.loads(row[5])
    except (json.JSONDecodeError, TypeError):
        sources = []
    return {
        "id": row[0],
        "session_id": row[1],
        "timestamp": row[2],
        "user_question": row[3],
        "assistant_response": row[4],
        "retrieved_sources": sources,
        "escalation_status": row[6],
        "escalation_reason": row[7],
    }


def get_conversation_history(
    session_id: str,
    db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return all saved turns for one session in chronological order."""
    if not initialize_database(db_path):
        return []

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(db_path)
        rows = connection.execute(
            """
            SELECT id, session_id, timestamp, user_question, assistant_response,
                   retrieved_sources, escalation_status, escalation_reason
            FROM conversations
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
        return [_row_to_record(row) for row in rows]
    except (OSError, sqlite3.Error):
        return []
    finally:
        if connection is not None:
            connection.close()


def get_recent_conversations(
    limit: int = 20,
    db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return the most recent saved turns across sessions."""
    if limit <= 0 or not initialize_database(db_path):
        return []

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(db_path)
        rows = connection.execute(
            """
            SELECT id, session_id, timestamp, user_question, assistant_response,
                   retrieved_sources, escalation_status, escalation_reason
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_record(row) for row in rows]
    except (OSError, sqlite3.Error):
        return []
    finally:
        if connection is not None:
            connection.close()


def get_all_conversations(
    db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return stored conversation rows for aggregate dashboard calculations."""
    if not initialize_database(db_path):
        return []

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(db_path)
        rows = connection.execute(
            """
            SELECT id, session_id, timestamp, user_question, assistant_response,
                   retrieved_sources, escalation_status, escalation_reason
            FROM conversations
            ORDER BY id ASC
            """
        ).fetchall()
        return [_row_to_record(row) for row in rows]
    except (OSError, sqlite3.Error):
        return []
    finally:
        if connection is not None:
            connection.close()


def clear_session(
    session_id: str,
    db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> bool:
    """Delete a session's history only when explicitly requested by a caller."""
    if not initialize_database(db_path):
        return False

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(db_path)
        connection.execute(
            "DELETE FROM conversations WHERE session_id = ?", (session_id,)
        )
        connection.commit()
        return True
    except (OSError, sqlite3.Error):
        return False
    finally:
        if connection is not None:
            connection.close()
