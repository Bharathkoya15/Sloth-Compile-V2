"""
database.py
Lightweight, thread-safe SQLite persistence layer for Sloth-Compile V2.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from config import DB_PATH


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Provide a transactional SQLite connection configured with WAL mode
    for safe concurrent access between the background watcher and Streamlit.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize the SQLite database and create the `fixes` table if it does not exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS fixes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        original_code TEXT NOT NULL,
        fixed_code TEXT NOT NULL,
        diff_html TEXT NOT NULL,
        status TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        model_used TEXT NOT NULL
    );
    """
    create_index_sql = """
    CREATE INDEX IF NOT EXISTS idx_fixes_timestamp ON fixes (id DESC);
    """
    with get_db_connection() as conn:
        conn.execute(create_table_sql)
        conn.execute(create_index_sql)


def insert_fix(
    filename: str,
    original_code: str,
    fixed_code: str,
    diff_html: str,
    status: str,
    model_used: str,
    timestamp: str | None = None,
) -> int:
    """
    Insert a new processed file record into the `fixes` table.

    Returns:
        int: The newly inserted row ID.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

    insert_sql = """
    INSERT INTO fixes (
        filename,
        original_code,
        fixed_code,
        diff_html,
        status,
        timestamp,
        model_used
    ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    with get_db_connection() as conn:
        cursor = conn.execute(
            insert_sql,
            (
                filename,
                original_code,
                fixed_code,
                diff_html,
                status,
                timestamp,
                model_used,
            ),
        )
        return int(cursor.lastrowid or 0)


def get_recent_fixes(limit: int = 50) -> list[dict[str, Any]]:
    """
    Retrieve the most recent code fix records ordered newest first.

    Args:
        limit: Maximum number of rows to return.

    Returns:
        list[dict[str, Any]]: List of dictionaries representing rows from `fixes`.
    """
    query_sql = """
    SELECT
        id,
        filename,
        original_code,
        fixed_code,
        diff_html,
        status,
        timestamp,
        model_used
    FROM fixes
    ORDER BY id DESC
    LIMIT ?;
    """
    with get_db_connection() as conn:
        cursor = conn.execute(query_sql, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]