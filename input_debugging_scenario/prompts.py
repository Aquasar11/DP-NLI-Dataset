"""
Prompt templates for LLM interactions.

Two-step prompting:
  Step 1 – Generate altering SQL + explanation (validated before proceeding)
  Step 2 – Generate follow-up question, gold explanation, and gold fix
"""

from __future__ import annotations

import json
from typing import Any

from models import AlterationType


def _format_rows(rows: list[dict[str, Any]], max_rows: int = 30) -> str:
    """Format result rows as a readable table-like string."""
    if not rows:
        return "(empty result set)"
    truncated = rows[:max_rows]
    lines = [json.dumps(row, ensure_ascii=False, default=str) for row in truncated]
    result = "\n".join(lines)
    if len(rows) > max_rows:
        result += f"\n... ({len(rows) - max_rows} more rows omitted)"
    return result


def _format_targeted(
    targeted_records: list[dict[str, Any]],
    alteration_type: AlterationType,
) -> str:
    """Describe the targeted records and what should happen to them."""
    parts = []
    for i, rec in enumerate(targeted_records):
        parts.append(f"  Record {i + 1}: {json.dumps(rec, ensure_ascii=False, default=str)}")

    target_desc = "\n".join(parts)

    if alteration_type == AlterationType.DELETE:
        action = (
            "DELETE the above record(s) entirely from the database so they no longer "
            "appear in the query result.\n"
            'In the response, set "target_columns" to ["all"].'
        )
    else:
        action = (
            "MODIFY one or more columns of the above record(s) so that they no "
            "longer satisfy the query conditions and disappear from the result.\n"
            "IMPORTANT: Carefully analyze the gold SQL query — look at the WHERE, "
            "JOIN, HAVING, and other conditions. Choose column(s) that are used "
            "in those conditions (not just the columns in the SELECT clause). "
            "Changing a column that only appears in the SELECT will NOT remove "
            "the record from the result. Instead, change a column that the query "
            "filters on (e.g., a column in the WHERE clause, a JOIN key, or a "
            "column used in a subquery condition).\n"
            "Set the chosen column(s) to NULL, a default value, or a value that "
            "breaks the query conditions.\n"
            'In the response, list the column(s) you modified in "target_columns".'
        )
    return target_desc, action


# ── Step 1: Alteration SQL Prompt ──────────────────────────────────────────────

ALTERATION_SYSTEM_PROMPT = """\
You are a database expert. Your task is to write SQL statements that alter a \
database so that specific records disappear from a query's result.

RULES:
1. Only output valid SQLite-compatible SQL (DELETE or UPDATE statements).
2. The alteration must ONLY affect the targeted records — no other rows in the \
query result should be added, removed, or changed.
3. Be precise: use primary keys or unique identifiers when possible to target \
exact rows.
4. If the query involves JOINs, identify which table's row needs to change.
5. For UPDATE (modify) operations, change column values so the row no longer \
matches the WHERE, JOIN, or HAVING conditions of the gold query.
6. Multiple SQL statements should be separated by semicolons.

OUTPUT FORMAT — respond with valid JSON only:
{
  "altering_sql": "<SQL statement(s)>",
  "target_columns": ["<column1>", "<column2>"],
  "explanation": "<why this alteration removes the targeted records from the result>"
}\
"""


def build_alteration_prompt(
    *,
    gold_sql: str,
    db_ddl: str,
    gold_result: list[dict[str, Any]],
    targeted_records: list[dict[str, Any]],
    alteration_type: AlterationType,
) -> str:
    """Build the user prompt for Step 1: generating the altering SQL."""
    target_desc, action = _format_targeted(
        targeted_records, alteration_type,
    )

    return f"""\
## Database Schema (DDL)
```sql
{db_ddl}
```

## Gold SQL Query
```sql
{gold_sql}
```

## Current Query Result ({len(gold_result)} rows)
{_format_rows(gold_result)}

## Targeted Record(s) to Remove from Result
{target_desc}

## Required Action
{action}

Write the SQL statement(s) that will alter the database so that the targeted \
record(s) no longer appear when the gold SQL query is re-executed. \
The rest of the query result must remain unchanged.

Respond with JSON only.\
"""


# ── Step 1 Retry Prompt ───────────────────────────────────────────────────────

def build_retry_prompt(
    *,
    previous_altering_sql: str,
    previous_explanation: str,
    error_message: str,
    gold_sql: str,
    db_ddl: str,
    gold_result: list[dict[str, Any]],
    altered_result: list[dict[str, Any]],
    targeted_records: list[dict[str, Any]],
    alteration_type: AlterationType,
) -> str:
    """Build a retry prompt when the previous alteration failed validation."""
    target_desc, action = _format_targeted(
        targeted_records, alteration_type,
    )

    return f"""\
Your previous alteration SQL did NOT produce the correct result. Please fix it.

## Database Schema (DDL)
```sql
{db_ddl}
```

## Gold SQL Query
```sql
{gold_sql}
```

## Original Query Result ({len(gold_result)} rows)
{_format_rows(gold_result)}

## Targeted Record(s) to Remove
{target_desc}

## Required Action
{action}

## Your Previous Attempt
```sql
{previous_altering_sql}
```
Previous explanation: {previous_explanation}

## Validation Error
{error_message}

## Result After Your Previous Alteration ({len(altered_result)} rows)
{_format_rows(altered_result)}

Please provide a CORRECTED altering SQL. Think carefully about which table and \
which exact row(s) need to be altered, using primary keys or unique identifiers.

Respond with JSON only:
{{
  "altering_sql": "<corrected SQL>",
  "target_columns": ["<column1>", "<column2>"],
  "explanation": "<why this corrected alteration works>"
}}\
"""


# ── Step 2: Follow-up Question & Explanation Prompt ────────────────────────────

FOLLOWUP_SYSTEM_PROMPT = """\
You are a database debugging expert helping a user understand why a SQL query \
returned unexpected results due to corrupted/modified data.

Given:
- A natural language question about a database
- The correct SQL query
- The correct (gold) result
- The altered (wrong) result caused by data modification
- An explanation of what data was changed

Generate:
1. A natural follow-up question the user would ask when seeing the wrong output
2. A gold explanation that identifies the data corruption
3. A SQL fix to restore the database to its correct state

OUTPUT FORMAT — respond with valid JSON only:
{
  "follow_up_question": "<question>",
  "gold_explanation": "<explanation>",
  "gold_fix": "<SQL to reverse the alteration>"
}\
"""


def build_followup_prompt(
    *,
    question: str,
    evidence: str,
    gold_sql: str,
    gold_result: list[dict[str, Any]],
    altered_result: list[dict[str, Any]],
    alteration_type: AlterationType,
    targeted_records: list[dict[str, Any]],
    altering_sql: str,
    alteration_explanation: str,
) -> str:
    """Build the user prompt for Step 2: generating follow-up Q&A."""
    return f"""\
## Natural Language Question
{question}

## Evidence / Hints
{evidence}

## Gold SQL Query
```sql
{gold_sql}
```

## Correct (Gold) Result ({len(gold_result)} rows)
{_format_rows(gold_result)}

## Altered (Wrong) Result ({len(altered_result)} rows)
{_format_rows(altered_result)}

## What Was Changed
- Alteration type: {alteration_type.value}
- Targeted records: {json.dumps(targeted_records, ensure_ascii=False, default=str)}
- Altering SQL: {altering_sql}
- Explanation: {alteration_explanation}

Based on the above, generate:
1. A follow-up question a user would naturally ask when seeing the wrong output \
instead of the correct one (e.g., "Why is X missing?" or "Why does the count \
show Y instead of Z?")
2. A gold explanation that precisely identifies the data issue
3. A SQL fix (INSERT or UPDATE) that reverses the alteration

Respond with JSON only.\
"""


# ── Step 3: Fix SQL Retry Prompt ───────────────────────────────────────────────

FIX_RETRY_SYSTEM_PROMPT = """\
You are a database expert. Your task is to write SQL statements that REVERSE \
a previous database alteration, restoring the database to its original state.

RULES:
1. Only output valid SQLite-compatible SQL (INSERT or UPDATE statements).
2. The fix must restore EXACTLY the original query result — no extra or missing rows.
3. Be precise: use the correct table, column names, and values from the original data.
4. If the alteration was a DELETE, use INSERT to restore the deleted row(s).
5. If the alteration was an UPDATE, use UPDATE to restore the original column values.
6. Multiple SQL statements should be separated by semicolons.

OUTPUT FORMAT — respond with valid JSON only:
{
  "gold_fix": "<SQL statement(s) to reverse the alteration>",
  "explanation": "<why this fix restores the original state>"
}\
"""


def build_fix_retry_prompt(
    *,
    gold_sql: str,
    db_ddl: str,
    gold_result: list[dict[str, Any]],
    altering_sql: str,
    alteration_explanation: str,
    previous_fix_sql: str,
    fix_execution_error: str | None,
    result_after_fix: list[dict[str, Any]],
    result_mismatch_details: str,
) -> str:
    """Build a retry prompt when the previous fix SQL failed verification."""
    error_section = ""
    if fix_execution_error:
        error_section = f"""\n## Execution Error\n{fix_execution_error}\n"""

    return f"""\
Your previous fix SQL did NOT correctly restore the database to its original state. \
Please provide a corrected fix.

## Database Schema (DDL)
```sql
{db_ddl}
```

## Gold SQL Query
```sql
{gold_sql}
```

## Original (Correct) Query Result ({len(gold_result)} rows)
{_format_rows(gold_result)}

## Alteration That Was Applied
```sql
{altering_sql}
```
Explanation: {alteration_explanation}

## Your Previous Fix Attempt
```sql
{previous_fix_sql}
```
{error_section}
## Result After Your Fix ({len(result_after_fix)} rows)
{_format_rows(result_after_fix)}

## Why It Failed
{result_mismatch_details}

Please provide a CORRECTED fix SQL that will restore the database so that the \
gold SQL query returns EXACTLY the original result.

Respond with JSON only:
{{
  "gold_fix": "<corrected fix SQL>",
  "explanation": "<why this corrected fix works>"
}}\
"""
