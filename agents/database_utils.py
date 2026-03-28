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


def compare_databases(db_path1: Path, db_path2: Path) -> tuple[bool, str]:
    """
    Compare two SQLite databases for identical content across all tables.

    Row comparison is order-insensitive — only set equality matters.

    Args:
        db_path1: Path to the first SQLite database (treated as reference).
        db_path2: Path to the second SQLite database to compare against.

    Returns:
        ``(True, "")`` if every table has the same rows in both databases,
        or ``(False, description)`` with a human-readable diff summary.
    """

    def _get_tables(conn: sqlite3.Connection) -> list[str]:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]

    def _get_rows(conn: sqlite3.Connection, table: str) -> frozenset[tuple]:
        cursor = conn.execute(f'SELECT * FROM "{table}"')
        return frozenset(tuple(row) for row in cursor.fetchall())

    conn1 = sqlite3.connect(str(db_path1), timeout=30)
    conn2 = sqlite3.connect(str(db_path2), timeout=30)
    try:
        tables1 = set(_get_tables(conn1))
        tables2 = set(_get_tables(conn2))

        if tables1 != tables2:
            missing = tables1 - tables2
            extra = tables2 - tables1
            parts = []
            if missing:
                parts.append(f"tables missing from second db: {sorted(missing)}")
            if extra:
                parts.append(f"extra tables in second db: {sorted(extra)}")
            return False, "; ".join(parts)

        diffs: list[str] = []
        for table in sorted(tables1):
            rows1 = _get_rows(conn1, table)
            rows2 = _get_rows(conn2, table)
            if rows1 != rows2:
                only_in_1 = rows1 - rows2
                only_in_2 = rows2 - rows1
                diffs.append(
                    f"table '{table}': {len(only_in_1)} row(s) only in original, "
                    f"{len(only_in_2)} row(s) only in candidate"
                )

        if diffs:
            return False, "; ".join(diffs)
        return True, ""
    finally:
        conn1.close()
        conn2.close()


def compute_structured_diff(
    original_db_path: Path,
    altered_db_path: Path,
) -> dict[str, Any]:
    """
    Compute a structured diff between two SQLite databases, identifying
    specific row-level changes using primary keys where available.

    For each table with differences, returns the affected original and altered
    rows. Uses primary keys to match rows when possible; falls back to
    full-row equality comparison when PKs are unavailable.

    Returns:
        Dict with structure::

            {
                "tables": {
                    "TableName": {
                        "primary_keys": ["col1", ...],
                        "columns": ["col1", "col2", ...],
                        "original_records": [...],  # rows from original that differ/are missing
                        "altered_records": [...],    # corresponding altered rows
                    },
                    ...
                }
            }
    """
    conn_orig = sqlite3.connect(str(original_db_path), timeout=30)
    conn_orig.row_factory = sqlite3.Row
    conn_alt = sqlite3.connect(str(altered_db_path), timeout=30)
    conn_alt.row_factory = sqlite3.Row
    try:
        # Get table lists
        orig_tables = set(
            row[0] for row in conn_orig.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        )
        alt_tables = set(
            row[0] for row in conn_alt.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        )
        common_tables = sorted(orig_tables & alt_tables)
        logger.debug("compute_structured_diff: %d common tables", len(common_tables))

        result_tables: dict[str, Any] = {}

        for table in common_tables:
            # Identify primary key columns via PRAGMA
            pragma = conn_orig.execute(f'PRAGMA table_info("{table}")').fetchall()
            columns = [row[1] for row in pragma]
            pk_cols = [row[1] for row in pragma if row[5] > 0]  # pk field > 0

            if not pk_cols:
                logger.warning(
                    "Table '%s' has no primary key; using full-row comparison", table
                )

            # Fetch all rows as dicts
            orig_rows = [dict(r) for r in conn_orig.execute(f'SELECT * FROM "{table}"').fetchall()]
            alt_rows = [dict(r) for r in conn_alt.execute(f'SELECT * FROM "{table}"').fetchall()]

            original_records: list[dict[str, Any]] = []
            altered_records: list[dict[str, Any]] = []

            if pk_cols:
                # Index rows by PK tuple
                def _pk(row: dict) -> tuple:
                    return tuple(row[c] for c in pk_cols)

                orig_by_pk = {_pk(r): r for r in orig_rows}
                alt_by_pk = {_pk(r): r for r in alt_rows}

                # Rows only in original (missing from altered)
                for pk, row in orig_by_pk.items():
                    if pk not in alt_by_pk:
                        original_records.append(row)
                    elif row != alt_by_pk[pk]:
                        # Same PK but different values (changed row)
                        original_records.append(row)
                        altered_records.append(alt_by_pk[pk])

                # Rows only in altered (added)
                for pk, row in alt_by_pk.items():
                    if pk not in orig_by_pk:
                        altered_records.append(row)
            else:
                # No PK — fall back to frozenset comparison
                orig_set = frozenset(tuple(sorted(r.items())) for r in orig_rows)
                alt_set = frozenset(tuple(sorted(r.items())) for r in alt_rows)

                only_orig = orig_set - alt_set
                only_alt = alt_set - orig_set

                original_records = [dict(t) for t in only_orig]
                altered_records = [dict(t) for t in only_alt]

            if original_records or altered_records:
                result_tables[table] = {
                    "primary_keys": pk_cols,
                    "columns": columns,
                    "original_records": original_records,
                    "altered_records": altered_records,
                }
                logger.debug(
                    "Table '%s': %d original-only, %d altered-only record(s)",
                    table, len(original_records), len(altered_records),
                )

        return {"tables": result_tables}
    finally:
        conn_orig.close()
        conn_alt.close()


def format_diff_as_text(structured_diff: dict[str, Any]) -> str:
    """
    Format a structured diff (from :func:`compute_structured_diff`) as
    human-readable text tables suitable for inclusion in an LLM prompt.

    Each affected table is presented with its original and altered records
    in a markdown-style table format.
    """
    tables = structured_diff.get("tables", {})
    if not tables:
        return "No differences detected between original and current database."

    sections: list[str] = []

    for table_name, info in tables.items():
        columns = info["columns"]
        original_records = info["original_records"]
        altered_records = info["altered_records"]

        lines: list[str] = [f"Table: {table_name}", ""]

        if original_records:
            lines.append("Original Records:")
            lines.append(_format_table(columns, original_records))
            lines.append("")

        if altered_records:
            lines.append("Altered Records:")
            lines.append(_format_table(columns, altered_records))
            lines.append("")

        lines.append("Note: Record order within each table is not significant.")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _format_table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Format rows as a markdown-style text table."""
    # Compute column widths
    col_widths = {c: len(str(c)) for c in columns}
    for row in rows:
        for c in columns:
            val = "NULL" if row.get(c) is None else str(row.get(c, ""))
            col_widths[c] = max(col_widths[c], len(val))

    # Header
    header = "| " + " | ".join(str(c).ljust(col_widths[c]) for c in columns) + " |"
    separator = "|-" + "-|-".join("-" * col_widths[c] for c in columns) + "-|"

    # Rows
    data_lines: list[str] = []
    for row in rows:
        vals = []
        for c in columns:
            val = "NULL" if row.get(c) is None else str(row.get(c, ""))
            vals.append(val.ljust(col_widths[c]))
        data_lines.append("| " + " | ".join(vals) + " |")

    return "\n".join([header, separator] + data_lines)
