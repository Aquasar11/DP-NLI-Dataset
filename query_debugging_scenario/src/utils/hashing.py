from datasketch import MinHash
from dotenv import load_dotenv

import os

load_dotenv(override=True)
NUM_PERMUTATIONS = int(os.getenv("NUM_PERMUTATIONS", 128))

def get_table_columns(conn):
    """Returns a dict of {table_name: [column_names]}"""
    cursor = conn.cursor()
    table_columns = {}
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    for table in tables:
        cursor.execute(f"PRAGMA table_info(`{table}`);")
        columns = [r[1] for r in cursor.fetchall()]
        table_columns[table] = columns
    return table_columns


def generate_minhash_for_column(conn, table, column, num_perm=NUM_PERMUTATIONS, max_values=None):
    """Generates a MinHash signature for the given table.column."""
    m = MinHash(num_perm=num_perm)
    cursor = conn.cursor()
    count_rows = 0
    try:
        query = f"SELECT DISTINCT `{column}` FROM `{table}` WHERE `{column}` IS NOT NULL;"
        if max_values:
            query += f" LIMIT {max_values}"
        cursor.execute(query)
        for row in cursor:
            count_rows += 1
            val = str(row[0]).encode("utf8")
            m.update(val)
    except Exception as e:
        print(f"[WARN] Failed to hash {table}.{column}: {e}")
        return None
    # skip low cardinality columns
    if count_rows < 5:
        print(f"[WARN] Skipping {table}.{column} due to low cardinality ({count_rows} unique values)")
        return None
    return m


def generate_minhash_for_db(conn):
    """Generates a MinHash signature for each column in the database."""
    table_columns = get_table_columns(conn)
    minhashes = {}
    for table, columns in table_columns.items():
        for column in columns:
            print(f"Generating MinHash for {table}.{column}...")
            minhash = generate_minhash_for_column(conn, table, column)
            if minhash:
                minhashes[f"{table}.{column}"] = minhash
    return minhashes
