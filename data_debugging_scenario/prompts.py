"""
Prompt templates for LLM interactions.

Two-step prompting:
  Step 1 – Generate altering SQL + explanation (validated before proceeding)
  Step 2 – Generate follow-up question
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
            "MODIFY one or more columns of the above record(s) using UPDATE "
            "statements so that the targeted record(s) disappear from the result.\n"
            "CRITICAL: You MUST use UPDATE statements. NEVER use DELETE when the "
            "required action is MODIFY. Do NOT switch to DELETE under any "
            "circumstances.\n"
            "Carefully analyze the gold SQL query — look at the WHERE, JOIN, "
            "HAVING, and other conditions.\n"
            "COLUMN SELECTION PRIORITY (follow this order):\n"
            "1. BEST: Change attribute columns used in WHERE/JOIN/HAVING "
            "conditions (e.g., score, age, status, amount, date, color, salary, "
            "category) to a value that breaks the condition.\n"
            "2. GOOD: If no non-identifier condition columns exist, change "
            "attribute columns in the SELECT clause to a different value "
            "(this changes the record's returned values so the original "
            "targeted record no longer appears).\n"
            "3. LAST RESORT: Only if no other columns can be changed, modify "
            "identifier/primary key columns used in WHERE conditions.\n"
            "Use primary keys or unique identifiers in the WHERE clause of your "
            "UPDATE statement to target the correct row.\n"
            "Set the chosen column(s) to a value that breaks the query "
            "conditions or changes the output (e.g., a value outside a range, "
            "a different category, a different numeric value).\n"
            'In the response, list the column(s) you modified in "target_columns".\n\n'
        ) + MODIFY_EXAMPLES
    return target_desc, action


MODIFY_EXAMPLES = """\
Examples of correct MODIFY behavior:

Example 1 — Gold SQL: SELECT name FROM employees WHERE age > 30 AND salary > 50000
  GOOD: UPDATE employees SET age = 25 WHERE employee_id = 42;
  BAD:  UPDATE employees SET employee_id = NULL WHERE employee_id = 42;
  BAD:  DELETE FROM employees WHERE employee_id = 42;
  Why: age is the filtering attribute — changing it removes the row from the result. \
Never use DELETE when told to MODIFY. Never change identifiers when condition columns exist.

Example 2 — Gold SQL: SELECT T1.name FROM students AS T1 JOIN scores AS T2 ON T1.id = T2.student_id WHERE T2.score >= 90
  GOOD: UPDATE scores SET score = 50 WHERE student_id = 101;
  BAD:  UPDATE scores SET student_id = NULL WHERE student_id = 101;
  Why: score is the attribute the query filters on — lowering it excludes the row. \
Changing student_id breaks the JOIN but also corrupts referential integrity.

Example 3 — Gold SQL: SELECT StandardCost FROM ProductCostHistory WHERE ProductID = 847
  GOOD: UPDATE ProductCostHistory SET StandardCost = 0.0 WHERE ProductID = 847;
  BAD:  DELETE FROM ProductCostHistory WHERE ProductID = 847;
  BAD:  UPDATE ProductCostHistory SET ProductID = -1 WHERE ProductID = 847;
  Why: The only WHERE condition is on ProductID (an identifier). Instead of deleting \
or changing the PK, change the SELECT attribute StandardCost — this changes the \
returned value so the original targeted record disappears from the result.

Example 4 — Gold SQL: SELECT product_name FROM products WHERE category = 'Electronics' AND price < 1000
  GOOD: UPDATE products SET category = 'Furniture' WHERE product_id = 7;
  BAD:  UPDATE products SET product_name = NULL WHERE product_id = 7;
  Why: category is the condition column — changing it to a different value removes the row \
from the filtered result.\
"""


# ── Step 1: Alteration SQL Prompt ──────────────────────────────────────────────

ALTERATION_SYSTEM_PROMPT = """\
You are a database expert. Your task is to write SQL statements that alter a \
database so that specific records disappear from a query's result.

RULES:
1. Only output valid SQLite-compatible SQL (DELETE or UPDATE statements).
2. NEVER switch between operation types. If the required action says MODIFY, \
you MUST use UPDATE — never DELETE. If it says DELETE, use DELETE.
3. In WHERE clauses, use primary keys or unique identifiers to target exact rows.
4. For UPDATE (modify) operations — column selection priority:
   a. BEST: Modify attribute columns in the gold query's WHERE, JOIN, HAVING, \
or aggregate conditions (e.g., score, age, status, amount, date, color, salary).
   b. GOOD: If no non-identifier condition columns exist, modify attribute \
columns in the SELECT clause to change the returned values.
   c. LAST RESORT: Only if no other columns work, modify identifier/primary \
key columns used in conditions.
5. If the query involves JOINs, identify which table's row needs to change.
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


# ── Step 1: Aggregate Alteration Prompt ───────────────────────────────────────

AGGREGATE_ALTERATION_SYSTEM_PROMPT = """\
You are a database expert. Your task is to write SQL statements that alter \
underlying table data so that an aggregate SQL query returns a DIFFERENT value \
than it currently does.

RULES:
1. Only output valid SQLite-compatible SQL (DELETE or UPDATE statements).
2. NEVER switch between operation types. If the required action says MODIFY, \
you MUST use UPDATE — never DELETE. If it says DELETE, use DELETE.
3. Your alteration must change the output of the aggregate query — the new \
aggregate result must differ from the original one.
4. Aim to modify or delete approximately the requested number of underlying rows.
5. In WHERE clauses, use primary keys or unique identifiers to target exact \
rows. For UPDATE statements, prefer modifying attribute columns (score, amount, \
date, status, etc.) over identifiers. Only change identifiers as a last resort.
6. If the query involves JOINs, identify which table's rows need to change.
7. Multiple SQL statements should be separated by semicolons.

OUTPUT FORMAT — respond with valid JSON only:
{
  "altering_sql": "<SQL statement(s)>",
  "target_columns": ["<column1>", "<column2>"],
  "explanation": "<why this alteration changes the aggregate result>"
}\
"""


def build_aggregate_alteration_prompt(
    *,
    gold_sql: str,
    db_ddl: str,
    gold_result: list[dict[str, Any]],
    alteration_type: AlterationType,
    num_targets: int,
) -> str:
    """Build the user prompt for Step 1 when the gold query is an aggregate."""
    if alteration_type == AlterationType.DELETE:
        action = (
            f"DELETE approximately {num_targets} underlying row(s) from the relevant "
            f"table(s) so that the aggregate result changes.\n"
            'In the response, set "target_columns" to ["all"].'
        )
    else:
        action = (
            f"MODIFY approximately {num_targets} underlying row(s) — change column "
            "value(s) so that those rows are excluded from, or change the value of, "
            "the aggregate computation.\n"
            "IMPORTANT: Analyze the gold SQL carefully — look at WHERE clauses, JOIN "
            "conditions, and which columns feed into the aggregate function. Modify "
            "attribute columns (score, amount, date, status, etc.) that affect whether "
            "a row is counted/summed/averaged. NEVER change primary key or identifier "
            "columns.\n"
            'In the response, list the modified column(s) in "target_columns".\n\n'
        ) + MODIFY_EXAMPLES

    return f"""\
## Database Schema (DDL)
```sql
{db_ddl}
```

## Gold SQL Query (Aggregate)
```sql
{gold_sql}
```

## Current Aggregate Result
{_format_rows(gold_result)}

## Required Action
{action}

Write the SQL statement(s) that will alter the database so that when the gold \
SQL query is re-executed, it returns a DIFFERENT aggregate value than the one \
shown above.

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
which exact row(s) need to be altered. Use primary keys in the WHERE clause to \
target rows, but modify attribute columns (not identifiers) in the SET clause.

Respond with JSON only:
{{
  "altering_sql": "<corrected SQL>",
  "target_columns": ["<column1>", "<column2>"],
  "explanation": "<why this corrected alteration works>"
}}\
"""


def build_aggregate_retry_prompt(
    *,
    previous_altering_sql: str,
    previous_explanation: str,
    error_message: str,
    gold_sql: str,
    db_ddl: str,
    gold_result: list[dict[str, Any]],
    altered_result: list[dict[str, Any]],
    alteration_type: AlterationType,
    num_targets: int,
) -> str:
    """Build a retry prompt when a previous aggregate alteration failed validation."""
    if alteration_type == AlterationType.DELETE:
        action = (
            f"DELETE approximately {num_targets} underlying row(s) from the relevant "
            f"table(s) so that the aggregate result changes.\n"
            'In the response, set "target_columns" to ["all"].'
        )
    else:
        action = (
            f"MODIFY approximately {num_targets} underlying row(s) — change column "
            "value(s) so that those rows are excluded from, or change the value of, "
            "the aggregate computation.\n"
            "IMPORTANT: Analyze the gold SQL carefully — look at WHERE clauses, JOIN "
            "conditions, and which columns feed into the aggregate function. "
            "Modify attribute columns (score, amount, date, etc.), NOT primary keys "
            "or identifiers.\n"
            'In the response, list the modified column(s) in "target_columns".\n\n'
        ) + MODIFY_EXAMPLES

    return f"""\
Your previous alteration SQL did NOT change the aggregate result. Please fix it.

## Database Schema (DDL)
```sql
{db_ddl}
```

## Gold SQL Query (Aggregate)
```sql
{gold_sql}
```

## Original Aggregate Result
{_format_rows(gold_result)}

## Required Action
{action}

## Your Previous Attempt
```sql
{previous_altering_sql}
```
Previous explanation: {previous_explanation}

## Validation Error
{error_message}

## Result After Your Previous Alteration
{_format_rows(altered_result)}

Please provide a CORRECTED altering SQL that causes the aggregate query to \
return a DIFFERENT value than the original.

Respond with JSON only:
{{
  "altering_sql": "<corrected SQL>",
  "target_columns": ["<column1>", "<column2>"],
  "explanation": "<why this corrected alteration changes the aggregate result>"
}}\
"""


# ── Step 2: Follow-up Question ────────────────────────────

FOLLOWUP_SYSTEM_PROMPT = """\
You are a database expert. Given a natural language question, its correct SQL \
query, the expected result, and the actual (wrong) result the user is seeing, \
you will generate one thing.

── Field instructions ──────────────────────────────────────────────────────

1. follow_up_question
   A natural question the user would ask when they notice the wrong output \
instead of the expected one (e.g. "Why is X missing?" or "Why does the count \
show Y instead of Z?").

OUTPUT FORMAT — respond with valid JSON only:
{
  "follow_up_question": "<question>",
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

## Current (Wrong) Result ({len(altered_result)} rows)
{_format_rows(altered_result)}

Generate the response as described in the system prompt.

Respond with JSON only.\
"""
