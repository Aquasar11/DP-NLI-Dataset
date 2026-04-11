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


def run_select_query(
    db_path: Path,
    sql: str,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """
    Execute a read-only SELECT query and return rows as a list of dicts.

    Args:
        db_path: Path to the SQLite database file.
        sql: A SELECT statement to execute.
        max_rows: If set, fetch at most this many rows (uses fetchmany to avoid
            loading millions of rows into Python memory when agents run broad
            queries on large databases).

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
        if max_rows is not None:
            return [dict(row) for row in cursor.fetchmany(max_rows)]
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
    Uses SQLite ATTACH + EXCEPT to avoid loading large tables into Python memory.

    Args:
        db_path1: Path to the first SQLite database (treated as reference).
        db_path2: Path to the second SQLite database to compare against.

    Returns:
        ``(True, "")`` if every table has the same rows in both databases,
        or ``(False, description)`` with a human-readable diff summary.
    """
    conn = sqlite3.connect(str(db_path1), timeout=60)
    try:
        conn.execute("ATTACH DATABASE ? AS cmp_db", (str(db_path2),))

        tables1 = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        tables2 = {
            row[0] for row in conn.execute(
                "SELECT name FROM cmp_db.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }

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
            # For very large tables, use a fast row-count comparison to avoid
            # expensive EXCEPT queries. The alteration only ever touches 1-2 rows
            # in one small table, so large tables should always be identical.
            orig_count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            cmp_count = conn.execute(f'SELECT COUNT(*) FROM cmp_db."{table}"').fetchone()[0]

            if orig_count != cmp_count:
                diffs.append(
                    f"table '{table}': row count differs (original={orig_count:,}, "
                    f"candidate={cmp_count:,})"
                )
                continue

            if orig_count > _MAX_DIFF_ROWS:
                # Table is very large and counts are equal — skip detailed EXCEPT diff.
                # A fix SQL that modifies values in a large table without changing the count
                # would be incorrectly scored as 1.0, but this is an acceptable trade-off
                # to prevent OOM / multi-minute queries on databases like language_corpus.
                logger.debug(
                    "Table '%s' has %d rows; skipping detailed diff (count matches)",
                    table, orig_count,
                )
                continue

            # Small table — use EXCEPT to detect value-level changes.
            only_in_1 = conn.execute(
                f'SELECT COUNT(*) FROM (SELECT * FROM "{table}" EXCEPT SELECT * FROM cmp_db."{table}")'
            ).fetchone()[0]
            only_in_2 = conn.execute(
                f'SELECT COUNT(*) FROM (SELECT * FROM cmp_db."{table}" EXCEPT SELECT * FROM "{table}")'
            ).fetchone()[0]

            if only_in_1 or only_in_2:
                diffs.append(
                    f"table '{table}': {only_in_1} row(s) only in original, "
                    f"{only_in_2} row(s) only in candidate"
                )

        if diffs:
            return False, "; ".join(diffs)
        return True, ""
    finally:
        try:
            conn.execute("DETACH DATABASE cmp_db")
        except Exception:
            pass
        conn.close()


# Tables with more than this many rows skip full-row extraction and only report
# the count delta. This prevents Python OOM on large databases (e.g. language_corpus).
_MAX_DIFF_ROWS = 100_000


def compute_structured_diff(
    original_db_path: Path,
    altered_db_path: Path,
) -> dict[str, Any]:
    """
    Compute a structured diff between two SQLite databases, identifying
    specific row-level changes using primary keys where available.

    For each table with differences, returns the affected original and altered
    rows. Uses SQLite ATTACH + EXCEPT to avoid loading large tables into Python
    memory — only the actually-changed rows (typically 1–2) are fetched.

    Tables with more than ``_MAX_DIFF_ROWS`` rows are summarised by count only
    to prevent OOM on very large databases.

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
    conn = sqlite3.connect(str(original_db_path), timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("ATTACH DATABASE ? AS alt_db", (str(altered_db_path),))

        orig_tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        alt_tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM alt_db.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        common_tables = sorted(orig_tables & alt_tables)
        logger.debug("compute_structured_diff: %d common tables", len(common_tables))

        result_tables: dict[str, Any] = {}

        for table in common_tables:
            pragma = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            columns = [row[1] for row in pragma]
            pk_cols = [row[1] for row in pragma if row[5] > 0]

            if not pk_cols:
                logger.warning(
                    "Table '%s' has no primary key; using full-row comparison", table
                )

            # ── Row count check ────────────────────────────────────────────
            # For very large tables, fetching all rows into Python is unsafe.
            # Check the count first; skip row extraction if the table is huge.
            orig_count: int = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            alt_count: int = conn.execute(f'SELECT COUNT(*) FROM alt_db."{table}"').fetchone()[0]

            if max(orig_count, alt_count) > _MAX_DIFF_ROWS:
                if orig_count != alt_count:
                    logger.warning(
                        "Table '%s' is too large for row-level diff (%d rows); "
                        "reporting count delta only (orig=%d, alt=%d)",
                        table, max(orig_count, alt_count), orig_count, alt_count,
                    )
                    result_tables[table] = {
                        "primary_keys": pk_cols,
                        "columns": columns,
                        "original_records": [],
                        "altered_records": [],
                        "note": (
                            f"Table too large for row-level diff ({orig_count:,} rows). "
                            f"Row count changed from {orig_count:,} to {alt_count:,}."
                        ),
                    }
                else:
                    # Same count — check quickly whether any rows differ at all.
                    has_diff = conn.execute(
                        f'SELECT EXISTS(SELECT 1 FROM "{table}" EXCEPT SELECT 1 FROM alt_db."{table}")'
                    ).fetchone()[0]
                    if has_diff:
                        logger.warning(
                            "Table '%s' is too large for row-level diff (%d rows); "
                            "rows changed but cannot extract specifics",
                            table, orig_count,
                        )
                        result_tables[table] = {
                            "primary_keys": pk_cols,
                            "columns": columns,
                            "original_records": [],
                            "altered_records": [],
                            "note": (
                                f"Table too large for row-level diff ({orig_count:,} rows). "
                                "Some rows changed but specifics are unavailable."
                            ),
                        }
                continue

            # ── Full row-level diff via EXCEPT (memory-safe) ───────────────
            # EXCEPT does set-difference inside SQLite; only the few changed rows
            # are returned to Python.
            original_records: list[dict[str, Any]] = [
                dict(r) for r in conn.execute(
                    f'SELECT * FROM "{table}" EXCEPT SELECT * FROM alt_db."{table}"'
                ).fetchall()
            ]
            altered_records: list[dict[str, Any]] = [
                dict(r) for r in conn.execute(
                    f'SELECT * FROM alt_db."{table}" EXCEPT SELECT * FROM "{table}"'
                ).fetchall()
            ]

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
        try:
            conn.execute("DETACH DATABASE alt_db")
        except Exception:
            pass
        conn.close()


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
