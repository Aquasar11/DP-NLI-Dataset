"""
Database manager for sandbox creation, querying, and schema introspection.

Handles SQLite database operations: executing queries, creating isolated sandbox
copies for alteration, and retrieving schema information.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import config
from models import TableSchema

logger = logging.getLogger(__name__)


class QueryResult:
    """Container for a SQL query result."""

    def __init__(self, rows: list[dict[str, Any]], columns: list[str]):
        self.rows = rows
        self.columns = columns

    def __len__(self) -> int:
        return len(self.rows)

    def __bool__(self) -> bool:
        return len(self.rows) > 0

    def __repr__(self) -> str:
        return f"QueryResult(columns={self.columns}, rows={len(self.rows)})"


class DatabaseManager:
    """Manages SQLite database operations for the pipeline."""

    def __init__(
        self,
        db_dir: Path | None = None,
        sandbox_dir: Path | None = None,
        tables_json_path: Path | None = None,
    ):
        self.db_dir = db_dir or config.BIRD_DB_DIR
        self.sandbox_dir = sandbox_dir or config.SANDBOX_DIR
        self.tables_json_path = tables_json_path or config.BIRD_TABLES_JSON
        self._tables_cache: dict[str, TableSchema] | None = None

        # Ensure sandbox directory exists
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    # ── Path Resolution ────────────────────────────────────────────────────

    def get_db_path(self, db_id: str) -> Path:
        """Resolve the path to the original SQLite database for a given db_id."""
        return self.db_dir / db_id / f"{db_id}.sqlite"

    # ── Query Execution ────────────────────────────────────────────────────

    def execute_query(self, db_path: Path, sql: str) -> QueryResult:
        """
        Execute a SELECT query and return the results as a list of dicts.

        Args:
            db_path: Path to the SQLite database file.
            sql: SQL query to execute.

        Returns:
            QueryResult with rows (list of dicts) and column names.

        Raises:
            sqlite3.Error: If the query fails.
        """
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = [dict(row) for row in cursor.fetchall()]
            return QueryResult(rows=rows, columns=columns)
        finally:
            conn.close()

    def execute_alter(self, db_path: Path, alter_sql: str) -> None:
        """
        Execute one or more data-altering SQL statements (DELETE, UPDATE, INSERT)
        on the given database.

        The alter_sql may contain multiple statements separated by semicolons.

        Args:
            db_path: Path to the SQLite database file (typically a sandbox copy).
            alter_sql: SQL statement(s) to execute.

        Raises:
            sqlite3.Error: If any statement fails (transaction is rolled back).
        """
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            conn.executescript(alter_sql)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Sandbox Management ─────────────────────────────────────────────────

    def create_sandbox(self, db_id: str) -> Path:
        """
        Create an isolated copy of the original database for safe alteration.

        Args:
            db_id: The database identifier.

        Returns:
            Path to the sandbox copy.
        """
        original = self.get_db_path(db_id)
        if not original.exists():
            raise FileNotFoundError(f"Original database not found: {original}")

        sandbox_name = f"{db_id}_{uuid.uuid4().hex[:8]}.sqlite"
        sandbox_path = self.sandbox_dir / sandbox_name
        shutil.copy2(str(original), str(sandbox_path))
        logger.debug("Created sandbox: %s", sandbox_path)
        return sandbox_path

    def destroy_sandbox(self, sandbox_path: Path) -> None:
        """Delete a sandbox database file."""
        if sandbox_path.exists():
            sandbox_path.unlink()
            logger.debug("Destroyed sandbox: %s", sandbox_path)

    def cleanup_all_sandboxes(self) -> None:
        """Remove all sandbox files (e.g., on startup or shutdown)."""
        if self.sandbox_dir.exists():
            for f in self.sandbox_dir.glob("*.sqlite"):
                f.unlink()
            logger.info("Cleaned up all sandbox files")

    # ── Schema Introspection ───────────────────────────────────────────────

    def _load_tables_json(self) -> dict[str, TableSchema]:
        """Load and cache the train_tables.json file."""
        if self._tables_cache is not None:
            return self._tables_cache

        with open(self.tables_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._tables_cache = {}
        for entry in data:
            schema = TableSchema(**entry)
            self._tables_cache[schema.db_id] = schema
        return self._tables_cache

    def get_table_schema(self, db_id: str) -> TableSchema | None:
        """Get the schema information for a database from train_tables.json."""
        tables = self._load_tables_json()
        return tables.get(db_id)

    def get_ddl(self, db_id: str) -> str:
        """
        Get the CREATE TABLE DDL statements for all tables in a database
        by reading directly from the SQLite file.

        Returns a string with all CREATE TABLE statements separated by newlines.
        """
        db_path = self.get_db_path(db_id)
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
            )
            ddl_statements = [row[0] for row in cursor.fetchall()]
            return "\n\n".join(ddl_statements)
        finally:
            conn.close()

    def get_table_names(self, db_id: str) -> list[str]:
        """Get all table names in a database."""
        db_path = self.get_db_path(db_id)
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_column_info(self, db_path: Path, table_name: str) -> list[dict[str, Any]]:
        """
        Get column information for a specific table using PRAGMA table_info.

        Returns a list of dicts with keys: cid, name, type, notnull, dflt_value, pk
        """
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            cursor = conn.execute(f"PRAGMA table_info(`{table_name}`)")
            columns = ["cid", "name", "type", "notnull", "dflt_value", "pk"]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_sample_rows(
        self, db_path: Path, table_name: str, limit: int = 3
    ) -> list[dict[str, Any]]:
        """Get a few sample rows from a table for context in prompts."""
        result = self.execute_query(db_path, f"SELECT * FROM `{table_name}` LIMIT {limit}")
        return result.rows
