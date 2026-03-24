"""
SQLite database utilities for the multi-agent framework.

Provides safe query execution, schema introspection, and sandbox management
(creating temporary copies of production databases with alterations applied).
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

# DML keywords that are forbidden in the read-only query path
_WRITE_KEYWORDS = frozenset(
    {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE"}
)


def get_db_path(db_id: str, db_base_dir: Path | None = None) -> Path:
    """Return the path to the original SQLite file for *db_id*."""
    base = db_base_dir or config.DB_BASE_DIR
    return base / db_id / f"{db_id}.sqlite"


def run_select_query(db_path: Path, sql: str) -> list[dict[str, Any]]:
    """
    Execute a read-only SELECT query and return rows as a list of dicts.

    Args:
        db_path: Path to the SQLite database file.
        sql: A SELECT statement to execute.

    Returns:
        List of row dicts (column-name → value).

    Raises:
        ValueError: If the SQL looks like a write operation.
        sqlite3.Error: On any database error.
    """
    first_token = sql.strip().split()[0].upper() if sql.strip() else ""
    if first_token in _WRITE_KEYWORDS:
        raise ValueError(
            f"Only SELECT queries are permitted in this context; got: {sql[:80]!r}"
        )

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_ddl(db_path: Path) -> str:
    """Return all CREATE TABLE statements from the database schema."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        )
        return "\n\n".join(row[0] for row in cursor.fetchall())
    finally:
        conn.close()


def create_altered_sandbox(
    db_id: str,
    altering_sql: str,
    sandbox_dir: Path | None = None,
    db_base_dir: Path | None = None,
) -> Path:
    """
    Create a temporary copy of the original database with *altering_sql* applied.

    This gives the UserAgent an authentic post-alteration database to query.

    Args:
        db_id: Database identifier.
        altering_sql: DML statement(s) that were applied to corrupt the database.
        sandbox_dir: Directory where the sandbox copy will be placed.
        db_base_dir: Root directory containing per-database folders.

    Returns:
        Path to the sandbox SQLite file.

    Raises:
        FileNotFoundError: If the original database does not exist.
        sqlite3.Error: If applying *altering_sql* fails.
    """
    original = get_db_path(db_id, db_base_dir)
    if not original.exists():
        raise FileNotFoundError(f"Original database not found: {original}")

    _sandbox_dir = sandbox_dir or config.SANDBOX_DIR
    _sandbox_dir.mkdir(parents=True, exist_ok=True)

    sandbox_name = f"{db_id}_{uuid.uuid4().hex[:8]}.sqlite"
    sandbox_path = _sandbox_dir / sandbox_name
    shutil.copy2(str(original), str(sandbox_path))
    logger.debug("Created sandbox copy: %s", sandbox_path)

    conn = sqlite3.connect(str(sandbox_path), timeout=30)
    try:
        conn.executescript(altering_sql)
        conn.commit()
        logger.debug("Applied altering SQL to sandbox: %s", sandbox_path)
    except Exception:
        conn.rollback()
        conn.close()
        sandbox_path.unlink(missing_ok=True)
        raise
    finally:
        conn.close()

    return sandbox_path


def destroy_sandbox(sandbox_path: Path) -> None:
    """Delete a sandbox database file."""
    if sandbox_path.exists():
        sandbox_path.unlink()
        logger.debug("Destroyed sandbox: %s", sandbox_path)
