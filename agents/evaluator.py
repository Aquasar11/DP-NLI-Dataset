"""
Evaluator — scores the FixAgent's output and the ExplanationAgent's explanation.

Two evaluation dimensions:

1. **Explanation quality** (LLM-as-judge):
   - A judge LLM compares the investigator's explanation against the ground truth.
   - Returns a score in [0.0, 1.0] and a reasoning string.

2. **Fix correctness** (database comparison):
   - The fix SQL is applied to a fresh copy of the altered database.
   - The result is compared table-by-table against the original database.
   - ``db_match`` is True only if every table has identical rows.

Scoring:
   - ``base_score`` = 1.0 if ``db_match``, else 0.0.
   - ``final_score`` = max(0.0, base_score − QUESTION_PENALTY × questions_asked).
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

    result, data = judge_llm.chat_json(system_prompt, [])

    if not result.success or data is None:
        logger.warning("[Evaluator] judge LLM call failed: %s", result.error)
        return 0.0, f"Judge call failed: {result.error}"

    if isinstance(data, list):
        data = data[0] if data else {}

    try:
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
    except (TypeError, ValueError):
        score = 0.0

    reasoning = str(data.get("reasoning", "")).strip()
    return score, reasoning


def _evaluate_fix(
    record: DatasetRecord,
    fix_result: FixResult,
    sandbox_dir: Path | None = None,
    db_base_dir: Path | None = None,
) -> tuple[bool, str]:
    """
    Apply the fix SQL to a fresh altered sandbox and compare against the original DB.

    Returns:
        ``(db_match, diff_description)``
    """
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

        # Now apply the fix SQL on top
        import sqlite3
        conn = sqlite3.connect(str(eval_sandbox), timeout=30)
        try:
            conn.executescript(fix_result.fix_sql)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            conn.close()
            return False, f"Fix SQL execution failed: {exc}"
        finally:
            conn.close()

        # Compare the repaired sandbox against the original database
        match, diff = compare_databases(original_db_path, eval_sandbox)
        return match, diff

    except Exception as exc:
        logger.warning("[Evaluator] fix evaluation error for record=%d: %s", record.id, exc)
        return False, f"Evaluation error: {exc}"
    finally:
        if eval_sandbox is not None:
            destroy_sandbox(eval_sandbox)


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
    try:
        explanation_score, explanation_reasoning = _judge_explanation(
            record, explanation_result, judge_llm
        )
    except Exception as exc:
        error = str(exc)
        explanation_score = 0.0
        explanation_reasoning = f"Judge evaluation error: {exc}"
        logger.warning("[Evaluator] explanation judge error for record=%d: %s", record.id, exc)

    # ── Fix evaluation (DB comparison) ───────────────────────────────────────
    try:
        db_match, db_diff = _evaluate_fix(
            record, fix_result, sandbox_dir=sandbox_dir, db_base_dir=db_base_dir
        )
    except Exception as exc:
        if error is None:
            error = str(exc)
        db_match = False
        db_diff = f"Fix evaluation error: {exc}"
        logger.warning("[Evaluator] fix eval error for record=%d: %s", record.id, exc)

    # ── Scoring ──────────────────────────────────────────────────────────────
    base_score = 1.0 if db_match else 0.0
    total_penalty = round(penalty * fix_result.questions_asked, 4)
    final_score = round(max(0.0, base_score - total_penalty), 4)

    logger.info(
        "[Evaluator] record=%d  explanation_score=%.2f  db_match=%s  "
        "questions=%d  base=%.2f  penalty=%.2f  final=%.4f",
        record.id, explanation_score, db_match, fix_result.questions_asked,
        base_score, total_penalty, final_score,
    )

    return EvaluationResult(
        record_id=record.id,
        db_id=record.db_id,
        explanation_score=round(explanation_score, 4),
        explanation_reasoning=explanation_reasoning,
        db_match=db_match,
        db_diff=db_diff if not db_match else "",
        questions_asked=fix_result.questions_asked,
        question_penalty=total_penalty,
        base_score=base_score,
        final_score=final_score,
        error=error,
    )
