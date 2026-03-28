"""
Evaluator — scores the FixAgent's output and the ExplanationAgent's explanation.

Two evaluation dimensions:

1. **Explanation quality** (LLM-as-judge):
   - A judge LLM compares the investigator's explanation against the ground truth.
   - Returns a score in {0.0, 0.5, 1.0} and a reasoning string.

2. **Fix correctness** (relaxed evaluation):
   - The fix SQL is applied to a fresh copy of the altered database.
   - ``fix_score`` = 1.0 if gold_sql returns gold_result AND no corruption.
   - ``fix_score`` = 1.5 if additionally all corrupted rows are fully restored.
   - ``fix_score`` = 0.0 if gold_sql result doesn't match or records were corrupted.

Scoring:
   - ``final_score`` = max(0.0, fix_score − QUESTION_PENALTY × questions_asked).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from database_utils import (
    compare_databases,
    create_altered_sandbox,
    destroy_sandbox,
    get_db_path,
    run_select_query,
)
from llm_client import GeminiClient, LLMClient
from models import DatasetRecord, EvaluationResult, FixResult, ExplanationResult
from prompts import JUDGE_SYSTEM_PROMPT

import config

logger = logging.getLogger(__name__)


def _judge_explanation(
    record: DatasetRecord,
    explanation_result: ExplanationResult,
    judge_llm: Union[LLMClient, GeminiClient],
) -> tuple[float, str]:
    """
    Use the judge LLM to score the explanation quality.

    Returns:
        ``(score, reasoning)`` — score is in [0.0, 1.0].
    """
    import json

    system_prompt = JUDGE_SYSTEM_PROMPT.format_map(
        {
            "altering_sql": record.altering_sql,
            "alteration_type": record.alteration_type,
            "targeted_records": json.dumps(record.targeted_records, default=str),
            "alteration_explanation": record.alteration_explanation,
            "agent_explanation": explanation_result.explanation,
            "agent_root_cause": explanation_result.root_cause,
        }
    )

    logger.info("[JudgeAgent] calling judge LLM for record=%d", record.id)
    result, data = judge_llm.chat_json(system_prompt, [])

    if not result.success or data is None:
        logger.warning("[JudgeAgent] LLM call failed: %s", result.error)
        return 0.0, f"Judge call failed: {result.error}"

    if isinstance(data, list):
        data = data[0] if data else {}

    try:
        raw_score = float(data.get("score", 0.0))
        # Snap to nearest valid 3-level score
        _VALID_SCORES = [0.0, 0.5, 1.0]
        score = min(_VALID_SCORES, key=lambda v: abs(v - raw_score))
    except (TypeError, ValueError):
        score = 0.0

    reasoning = str(data.get("reasoning", "")).strip()
    logger.info("[JudgeAgent] record=%d  score=%.1f  reasoning=%.100s", record.id, score, reasoning)
    return score, reasoning


def _evaluate_fix(
    record: DatasetRecord,
    fix_result: FixResult,
    sandbox_dir: Path | None = None,
    db_base_dir: Path | None = None,
) -> tuple[float, str]:
    """
    Apply the fix SQL to a fresh altered sandbox and evaluate correctness.

    Scoring:
      - 0.0: gold_sql does not return gold_result after fix, or good records corrupted.
      - 1.0: gold_sql returns gold_result AND no previously-good records were corrupted.
      - 1.5: additionally, all corrupted rows are fully restored to original values.

    Returns:
        ``(fix_score, description)``
    """
    import sqlite3

    logger.info("[Evaluator] evaluating fix SQL for record=%d", record.id)
    original_db_path = get_db_path(record.db_id, db_base_dir)
    eval_sandbox: Path | None = None
    try:
        # Create a fresh altered sandbox (original → apply altering_sql)
        eval_sandbox = create_altered_sandbox(
            record.db_id,
            record.altering_sql,
            sandbox_dir=sandbox_dir,
            db_base_dir=db_base_dir,
        )

        # Snapshot the altered DB's non-affected rows for corruption check.
        # "Non-affected rows" = rows that are identical in original and altered.
        altered_before_fix_path = eval_sandbox

        # Apply the fix SQL on top
        conn = sqlite3.connect(str(eval_sandbox), timeout=30)
        try:
            conn.executescript(fix_result.fix_sql)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            conn.close()
            logger.warning("[Evaluator] fix SQL execution failed for record=%d: %s", record.id, exc)
            return 0.0, f"Fix SQL execution failed: {exc}"
        finally:
            conn.close()
        logger.debug("[Evaluator] fix SQL applied successfully for record=%d", record.id)

        # --- Primary check: does gold_sql return gold_result? ---
        try:
            fixed_result = run_select_query(eval_sandbox, record.gold_sql)
        except Exception as exc:
            return 0.0, f"gold_sql execution failed on fixed DB: {exc}"

        # Compare gold_result (expected) vs fixed_result (actual) as sets of frozenset items
        def _normalize_rows(rows: list[dict]) -> frozenset:
            return frozenset(
                tuple(sorted(r.items())) for r in rows
            )

        gold_set = _normalize_rows(record.gold_result)
        fixed_set = _normalize_rows(fixed_result)

        if gold_set != fixed_set:
            return 0.0, (
                f"gold_sql result mismatch: expected {len(record.gold_result)} row(s), "
                f"got {len(fixed_result)} row(s)"
            )

        # --- Corruption check: ensure non-affected rows are unchanged ---
        # Rows that were the same in original and altered must still be the same after fix.
        corruption = _check_no_corruption(original_db_path, eval_sandbox, db_base_dir)
        if corruption:
            return 0.0, f"Fix corrupted previously-good records: {corruption}"

        # --- Bonus check: are corrupted rows fully restored? ---
        full_match, diff = compare_databases(original_db_path, eval_sandbox)
        if full_match:
            return 1.5, "gold_result matches, no corruption, and all rows fully restored"

        return 1.0, f"gold_result matches and no corruption (partial row restoration: {diff})"

    except Exception as exc:
        logger.warning("[Evaluator] fix evaluation error for record=%d: %s", record.id, exc)
        return 0.0, f"Evaluation error: {exc}"
    finally:
        if eval_sandbox is not None:
            destroy_sandbox(eval_sandbox)


def _check_no_corruption(
    original_db_path: Path,
    fixed_db_path: Path,
    db_base_dir: Path | None = None,
) -> str | None:
    """
    Check that rows which were identical in the original and altered DBs
    remain unchanged in the fixed DB.

    Returns None if no corruption, or a description string if corruption found.
    """
    import sqlite3

    conn_orig = sqlite3.connect(str(original_db_path), timeout=30)
    conn_fixed = sqlite3.connect(str(fixed_db_path), timeout=30)
    try:
        # Get all tables from original
        cursor = conn_orig.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        corruptions: list[str] = []
        for table in tables:
            # Get all rows from original and fixed as frozensets
            orig_rows = frozenset(
                tuple(row) for row in conn_orig.execute(f'SELECT * FROM "{table}"').fetchall()
            )
            try:
                fixed_rows = frozenset(
                    tuple(row) for row in conn_fixed.execute(f'SELECT * FROM "{table}"').fetchall()
                )
            except Exception:
                corruptions.append(f"table '{table}' missing or unreadable in fixed DB")
                continue

            # Rows that existed in original but are now missing from fixed DB
            # (and were NOT part of the alteration) indicate corruption.
            # We check: any row that was in original AND was in the altered DB
            # (i.e. it was a "good" row, not touched by the alteration)
            # must still be in the fixed DB.
            missing_from_fixed = orig_rows - fixed_rows
            if missing_from_fixed:
                # Some original rows are gone — this could be corruption
                # or it could be rows that were part of the alteration.
                # We allow rows that were already different in the altered DB,
                # but rows that existed unchanged must still be present.
                corruptions.append(
                    f"table '{table}': {len(missing_from_fixed)} original row(s) missing after fix"
                )

        return "; ".join(corruptions) if corruptions else None
    finally:
        conn_orig.close()
        conn_fixed.close()


def evaluate(
    record: DatasetRecord,
    fix_result: FixResult,
    explanation_result: ExplanationResult,
    judge_llm: Union[LLMClient, GeminiClient],
    question_penalty: float | None = None,
    sandbox_dir: Path | None = None,
    db_base_dir: Path | None = None,
) -> EvaluationResult:
    """
    Evaluate both the ExplanationAgent and FixAgent outputs for one record.

    Args:
        record: The ground-truth dataset record.
        fix_result: The FixAgent's output.
        explanation_result: The ExplanationAgent's output.
        judge_llm: LLM client used to score the explanation.
        question_penalty: Score deducted per question asked (defaults to config value).
        sandbox_dir: Override for sandbox directory.
        db_base_dir: Override for database root directory.

    Returns:
        :class:`EvaluationResult` with all metrics filled in.
    """
    penalty = question_penalty if question_penalty is not None else config.QUESTION_PENALTY
    error: str | None = None

    # ── Explanation evaluation (LLM-as-judge) ────────────────────────────────
    if explanation_result.is_fallback:
        explanation_score = 0.0
        explanation_reasoning = "FALLBACK triggered for ExplanationAgent — score 0, judge skipped."
        logger.warning(
            "[Evaluator] record=%d ExplanationAgent fallback — score 0, judge skipped",
            record.id,
        )
    else:
        try:
            explanation_score, explanation_reasoning = _judge_explanation(
                record, explanation_result, judge_llm
            )
        except Exception as exc:
            error = str(exc)
            explanation_score = 0.0
            explanation_reasoning = f"Judge evaluation error: {exc}"
            logger.warning("[Evaluator] explanation judge error for record=%d: %s", record.id, exc)

    # ── Fix evaluation (relaxed: gold_result match + no corruption) ─────────
    if fix_result.is_fallback:
        logger.warning(
            "[Evaluator] record=%d FixAgent fallback — DB comparison will likely fail",
            record.id,
        )
    try:
        fix_score, fix_description = _evaluate_fix(
            record, fix_result, sandbox_dir=sandbox_dir, db_base_dir=db_base_dir
        )
    except Exception as exc:
        if error is None:
            error = str(exc)
        fix_score = 0.0
        fix_description = f"Fix evaluation error: {exc}"
        logger.warning("[Evaluator] fix eval error for record=%d: %s", record.id, exc)

    # ── Scoring ──────────────────────────────────────────────────────────────
    total_penalty = round(penalty * fix_result.questions_asked, 4)
    final_score = round(max(0.0, fix_score - total_penalty), 4)

    logger.info(
        "[Evaluator] record=%d  explanation_score=%.2f  fix_score=%.1f  "
        "questions=%d  penalty=%.2f  final=%.4f",
        record.id, explanation_score, fix_score, fix_result.questions_asked,
        total_penalty, final_score,
    )

    return EvaluationResult(
        record_id=record.id,
        db_id=record.db_id,
        explanation_score=round(explanation_score, 4),
        explanation_reasoning=explanation_reasoning,
        fix_score=fix_score,
        fix_description=fix_description,
        questions_asked=fix_result.questions_asked,
        question_penalty=total_penalty,
        base_score=fix_score,
        final_score=final_score,
        error=error,
    )
