"""
Validation logic for verifying that a database alteration produced
the intended result: targeted records removed, everything else unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from models import AlterationType, ValidationResult

logger = logging.getLogger(__name__)


def _normalize_value(v: Any) -> Any:
    """Normalize a value for comparison (handle float/int equivalence, etc.)."""
    if v is None:
        return None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize all values in a row for comparison."""
    return {k: _normalize_value(v) for k, v in row.items()}


def _row_matches(row_a: dict[str, Any], row_b: dict[str, Any]) -> bool:
    """Check if two rows are equivalent (same keys and values after normalization)."""
    na = _normalize_row(row_a)
    nb = _normalize_row(row_b)
    if set(na.keys()) != set(nb.keys()):
        return False
    return all(na[k] == nb[k] for k in na)


def _row_in_list(row: dict[str, Any], row_list: list[dict[str, Any]]) -> bool:
    """Check if a row is present in a list of rows."""
    return any(_row_matches(row, r) for r in row_list)


def _row_to_key(row: dict[str, Any]) -> str:
    """Convert a row to a hashable string key for set operations."""
    normalized = _normalize_row(row)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)


def validate_alteration(
    gold_result: list[dict[str, Any]],
    altered_result: list[dict[str, Any]],
    targeted_records: list[dict[str, Any]],
    alteration_type: AlterationType,
) -> ValidationResult:
    """
    Validate that the alteration correctly removed targeted records.

    Only hard requirement: all targeted records must be absent from the altered
    result.  Side-effects — other records disappearing or new records appearing —
    are permitted and logged as informational observations only.

    Checks:
    1. (HARD) All targeted records are ABSENT from the altered result.
    2. (INFO) Non-targeted records that were unintentionally removed — logged but
       does NOT cause failure.
    3. (INFO) New records that appeared — logged but does NOT cause failure.

    Args:
        gold_result: The original query result before alteration.
        altered_result: The query result after alteration.
        targeted_records: The records that should have been removed.
        alteration_type: DELETE or MODIFY.

    Returns:
        ValidationResult with is_valid and error details.
    """
    errors: list[str] = []

    # ── Check 1 (HARD): Targeted records must be absent ────────────────────
    still_present = []
    missing_targeted = []
    for target in targeted_records:
        if _row_in_list(target, altered_result):
            still_present.append(target)
        else:
            missing_targeted.append(target)

    if still_present:
        errors.append(
            f"{len(still_present)} targeted record(s) still present in altered result: "
            f"{json.dumps(still_present, ensure_ascii=False, default=str)}"
        )

    # ── Check 2 (INFO): Track non-targeted records that went missing ────────
    non_targeted_gold = [
        row for row in gold_result if not _row_in_list(row, targeted_records)
    ]

    unintended_missing = []
    for row in non_targeted_gold:
        if not _row_in_list(row, altered_result):
            unintended_missing.append(row)

    if unintended_missing:
        logger.info(
            "%d non-targeted record(s) were also removed from the result "
            "(allowed — relaxed validation): %s",
            len(unintended_missing),
            json.dumps(unintended_missing[:5], ensure_ascii=False, default=str),
        )

    # ── Check 3 (INFO): Track unexpected new records ────────────────────────
    gold_keys = {_row_to_key(r) for r in gold_result}
    new_records = [
        row for row in altered_result if _row_to_key(row) not in gold_keys
    ]
    if new_records:
        logger.info(
            "%d new record(s) appeared in altered result that weren't in gold "
            "(allowed — relaxed validation): %s",
            len(new_records),
            json.dumps(new_records[:5], ensure_ascii=False, default=str),
        )

    # ── Assemble result ────────────────────────────────────────────────────
    is_valid = len(errors) == 0
    error_message = " | ".join(errors) if errors else None

    if is_valid:
        logger.debug("Validation passed: %d targeted records removed", len(missing_targeted))
    else:
        logger.debug("Validation failed: %s", error_message)

    return ValidationResult(
        is_valid=is_valid,
        error_message=error_message,
        missing_targeted=missing_targeted,
        still_present_targeted=still_present,
        unintended_missing=unintended_missing,
    )


def validate_alteration_aggregate(
    gold_result: list[dict[str, Any]],
    altered_result: list[dict[str, Any]],
) -> ValidationResult:
    """
    Simplified validation for aggregate queries (COUNT, AVG, SUM, etc.)
    that return a single-row scalar result.

    Just checks that the result actually changed — we can't do row-level
    matching for aggregates.
    """
    if not gold_result or not altered_result:
        return ValidationResult(
            is_valid=False,
            error_message="Empty result set (gold or altered)",
        )

    gold_key = _row_to_key(gold_result[0])
    altered_key = _row_to_key(altered_result[0])

    if gold_key == altered_key:
        return ValidationResult(
            is_valid=False,
            error_message=(
                "Aggregate result did not change after alteration. "
                f"Gold: {gold_result[0]} | Altered: {altered_result[0]}"
            ),
        )

    return ValidationResult(
        is_valid=True,
        missing_targeted=[gold_result[0]],
    )


def is_aggregate_query(sql: str) -> bool:
    """
    Heuristic to detect if a SQL query is an aggregate query
    (returns a single scalar/row rather than a set of records).
    """
    sql_upper = sql.upper().strip()

    # Check for aggregate functions without GROUP BY
    agg_functions = ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]
    has_agg = any(func in sql_upper for func in agg_functions)
    has_group_by = "GROUP BY" in sql_upper

    # If it has aggregates but no GROUP BY, it's likely a scalar aggregate
    if has_agg and not has_group_by:
        return True

    return False
