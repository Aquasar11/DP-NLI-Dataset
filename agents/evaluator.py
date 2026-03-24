"""
Evaluator — scores the AnswerAgent's prediction against ground truth.

Scoring methodology:
  - ``exact_match``: SQL strings are identical after whitespace/case normalization.
  - ``semantic_match``: Applying the predicted SQL to a fresh copy of the original
    database produces the same gold-query result as the ground-truth alteration.
  - ``base_score``: 1.0 if either match is True, else 0.0.
  - ``final_score``: base_score − (QUESTION_PENALTY × questions_asked), clamped to [0, 1].
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from database_utils import create_altered_sandbox, destroy_sandbox, run_select_query
from models import AnswerResult, DatasetRecord, EvaluationResult

import config

logger = logging.getLogger(__name__)


def _normalize_sql(sql: str) -> str:
    """Lowercase and collapse whitespace for a rough SQL comparison."""
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def _run_gold_on_altered(
    db_id: str,
    altering_sql: str,
    gold_sql: str,
    sandbox_dir: Path | None = None,
    db_base_dir: Path | None = None,
) -> list[dict[str, Any]] | None:
    """
    Apply *altering_sql* to a fresh sandbox and run *gold_sql* on it.

    Returns the result rows, or ``None`` if any step fails.
    """
    sandbox_path: Path | None = None
    try:
        sandbox_path = create_altered_sandbox(
            db_id, altering_sql,
            sandbox_dir=sandbox_dir,
            db_base_dir=db_base_dir,
        )
        return run_select_query(sandbox_path, gold_sql)
    except Exception as exc:
        logger.warning("semantic_match sandbox failed (%s): %s", altering_sql[:60], exc)
        return None
    finally:
        if sandbox_path is not None:
            destroy_sandbox(sandbox_path)


def evaluate(
    record: DatasetRecord,
    answer: AnswerResult,
    question_penalty: float | None = None,
    sandbox_dir: Path | None = None,
    db_base_dir: Path | None = None,
) -> EvaluationResult:
    """
    Evaluate the AnswerAgent's prediction for one dataset record.

    Args:
        record: The ground-truth dataset record.
        answer: The AnswerAgent's output.
        question_penalty: Score deducted per question asked (defaults to config value).
        sandbox_dir: Override for sandbox directory.
        db_base_dir: Override for database root directory.

    Returns:
        :class:`EvaluationResult` with all metrics filled in.
    """
    penalty = question_penalty if question_penalty is not None else config.QUESTION_PENALTY

    ground_truth = record.altering_sql
    predicted = answer.predicted_altering_sql
    error: str | None = None

    # ── Exact match ──────────────────────────────────────────────────────────
    exact_match = _normalize_sql(predicted) == _normalize_sql(ground_truth)

    # ── Semantic match ───────────────────────────────────────────────────────
    semantic_match = False
    if not exact_match:
        try:
            # Ground-truth effect: what gold_sql returns after ground_truth alteration
            # (should equal record.altered_result — used as reference)
            predicted_result = _run_gold_on_altered(
                record.db_id, predicted, record.gold_sql,
                sandbox_dir=sandbox_dir,
                db_base_dir=db_base_dir,
            )
            if predicted_result is not None:
                semantic_match = predicted_result == record.altered_result
        except Exception as exc:
            error = str(exc)
            logger.warning(
                "[Evaluator] semantic match error for record=%d: %s", record.id, exc
            )

    # ── Scoring ──────────────────────────────────────────────────────────────
    base_score = 1.0 if (exact_match or semantic_match) else 0.0
    total_penalty = round(penalty * answer.questions_asked, 4)
    final_score = round(max(0.0, base_score - total_penalty), 4)

    logger.info(
        "[Evaluator] record=%d  exact=%s  semantic=%s  questions=%d  "
        "base=%.2f  penalty=%.2f  final=%.4f",
        record.id, exact_match, semantic_match, answer.questions_asked,
        base_score, total_penalty, final_score,
    )

    return EvaluationResult(
        record_id=record.id,
        db_id=record.db_id,
        ground_truth_sql=ground_truth,
        predicted_sql=predicted,
        exact_match=exact_match,
        semantic_match=semantic_match,
        questions_asked=answer.questions_asked,
        question_penalty=total_penalty,
        base_score=base_score,
        final_score=final_score,
        error=error,
    )
