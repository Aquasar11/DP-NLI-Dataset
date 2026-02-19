import sqlite3
import sqlglot
from sqlglot import expressions as exp
from typing import List, Tuple


def extract_foreign_keys(conn: sqlite3.Connection) -> List[Tuple[str, str]]:
    """
    Extracts all declared FK constraints from the SQLite DB connection.
    Returns a list of (from_table.column, to_table.column) tuples.
    """
    cursor = conn.cursor()
    foreign_keys = []

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        try:
            cursor.execute(f"PRAGMA foreign_key_list(`{table}`)")
            for row in cursor.fetchall():
                from_col = f"{table}.{row[3]}"
                to_col = f"{row[2]}.{row[4]}"
                foreign_keys.append((from_col, to_col))
        except Exception as e:
            print(f"[WARN] Failed to extract FK from table {table}: {e}")

    return foreign_keys


def get_column_types(conn: sqlite3.Connection) -> dict:
    """
    Return a dict {table.column: type} for all tables in DB.
    """
    types = {}
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"PRAGMA table_info(`{table}`)")
        for row in cursor.fetchall():
            col = row[1]
            col_type = row[2].upper()
            types[f"{table}.{col}"] = col_type

    return types

def extract_join_columns(sql_query):
    try:
        parsed = sqlglot.parse_one(sql_query)
    except Exception as e:
        print(f"Error parsing SQL: {e}")
        return []

    alias_to_table = {}

    # Map aliases to table names
    for table in parsed.find_all(exp.Table):
        alias = table.alias_or_name
        table_name = table.name
        if alias:
            alias_to_table[alias] = table_name
        else:
            alias_to_table[table_name] = table_name

    join_column_pairs = []

    # Traverse the AST to find join expressions
    for join in parsed.find_all(exp.Join):
        on_condition = join.args.get("on")
        if on_condition:
            # Handle compound conditions (e.g., AND)
            conditions = [on_condition]
            if isinstance(on_condition, exp.And):
                conditions = on_condition.flatten()

            for condition in conditions:
                if isinstance(condition, exp.EQ):
                    left = condition.left
                    right = condition.right
                    if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                        left_table_alias = left.table
                        right_table_alias = right.table
                        left_table = alias_to_table.get(left_table_alias, left_table_alias)
                        right_table = alias_to_table.get(right_table_alias, right_table_alias)
                        left_col = f"{left_table}.{left.name}" if left_table else left.name
                        right_col = f"{right_table}.{right.name}" if right_table else right.name
                        join_column_pairs.append((left_col, right_col))
    return join_column_pairs
