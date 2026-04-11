"""
Evaluator — scores the FixAgent's output and the ExplanationAgent's explanation.

Four evaluation metrics:

1. **Alteration type accuracy** (systematic):
   - Compares predicted type ("deletion"/"modification") to the ground truth.
   - Returns 0 or 1.

2. **Explanation quality** (LLM-as-judge):
   - A judge LLM evaluates the investigator's explanation against the ground truth.
   - Returns a score in {0.0, 0.5, 1.0} and a reasoning string.

3. **Gold result score** (DB evaluation):
   - The fix SQL is applied to a fresh altered sandbox.
   - ``gold_result_score`` = 1.0 if gold_sql returns gold_result after fix, else 0.0.

4. **Full restore score** (DB evaluation):
   - ``full_restore_score`` = 1.0 if the fixed DB is identical to the original.

Scoring:
   - ``final_score`` = max(0.0, gold_result_score − QUESTION_PENALTY × questions_asked).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from database_utils import (
    _MAX_DIFF_ROWS,
    compare_databases,
    create_altered_sandbox,
    destroy_sandbox,
    get_db_path,
    run_select_query,
)
from llm_client import GeminiClient, LLMClient
from models import DatasetRecord, EvaluationResult, FixResult, ExplanationResult
from prompts import JUDGE_SYSTEM_PROMPT
from sample_logger import PipelineLogger

import config

logger = logging.getLogger(__name__)


def _evaluate_alteration_type(
    record: DatasetRecord,
    explanation_result: ExplanationResult,
) -> float:
    """
    Systematically compare the predicted alteration type against the ground truth.

    Normalises both values:
      - "deletion" / "delete"                    → "delete"
      - "modification" / "modify" / "update"     → "modify"

    Returns 1.0 if they match, 0.0 otherwise.
    """
    def _normalize(t: str) -> str:
        t = t.lower().strip()
        if t in ("deletion", "delete"):
            return "delete"
        if t in ("modification", "modify", "update"):
            return "modify"
        return t

    predicted = _normalize(explanation_result.alteration_type)
    ground_truth = _normalize(record.alteration_type)
    score = 1.0 if predicted == ground_truth else 0.0
    logger.info(
        "[Evaluator] record=%d alteration_type: predicted=%r  ground_truth=%r  score=%.1f",
        record.id, predicted, ground_truth, score,
    )
    return score


def _judge_explanation(
    record: DatasetRecord,
    explanation_result: ExplanationResult,
    judge_llm: Union[LLMClient, GeminiClient],
    pipeline_logger: PipelineLogger | None = None,
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
            "targeted_records": json.dumps(record.targeted_records, default=str),
            "alteration_explanation": record.alteration_explanation,
            "agent_explanation": explanation_result.explanation,
        }
    )

    logger.info(
        "[JudgeAgent] record=%d INPUT: explanation=%.200s | ground_truth_type=%s | "
        "ground_truth_explanation=%.100s",
        record.id,
        explanation_result.explanation,
        record.alteration_type,
        record.alteration_explanation,
    )
    logger.info("[JudgeAgent] calling judge LLM for record=%d", record.id)
    result, data = judge_llm.chat_json(system_prompt, [])

    if pipeline_logger:
        pipeline_logger.log_llm_call(
            agent="Judge",
            step="judge",
            system_prompt=system_prompt,
            messages=[],
            raw_response=result.content if result.content else None,
            parsed_response=data,
            success=result.success,
            error=result.error,
            duration_seconds=result.duration_seconds,
        )

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
    logger.info(
        "[JudgeAgent] record=%d OUTPUT: score=%.1f | reasoning=%.300s",
        record.id, score, reasoning,
    )
    return score, reasoning


def _evaluate_fix(
    record: DatasetRecord,
    fix_result: FixResult,
    sandbox_dir: Path | None = None,
    db_base_dir: Path | None = None,
) -> tuple[float, float, str, list]:
    """
    Apply the fix SQL to a fresh altered sandbox and evaluate correctness.

    Scoring:
      - gold_result_score = 1.0 if gold_sql returns gold_result after fix, else 0.0.
      - full_restore_score = 1.0 if the database is fully identical to the original, else 0.0.

    Returns:
        ``(gold_result_score, full_restore_score, description, actual_result_after_fix)``
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
        logger.info("[Evaluator] record=%d applying fix SQL on sandbox", record.id)
        conn = sqlite3.connect(str(eval_sandbox), timeout=30)
        try:
            conn.executescript(fix_result.fix_sql)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            conn.close()
            logger.warning("[Evaluator] record=%d fix SQL execution failed: %s", record.id, exc)
            return 0.0, 0.0, f"Fix SQL execution failed: {exc}", []
        finally:
            conn.close()
        logger.info("[Evaluator] record=%d fix SQL applied successfully", record.id)

        # --- Primary check: does gold_sql return gold_result? ---
        try:
            fixed_result = run_select_query(eval_sandbox, record.gold_sql)
        except Exception as exc:
            logger.warning("[Evaluator] record=%d gold_sql failed on fixed DB: %s", record.id, exc)
            return 0.0, 0.0, f"gold_sql execution failed on fixed DB: {exc}", []

        # Compare gold_result (expected) vs fixed_result (actual) as sets of frozenset items
        def _normalize_rows(rows: list[dict]) -> frozenset:
            return frozenset(
                tuple(sorted(r.items())) for r in rows
            )

        gold_set = _normalize_rows(record.gold_result)
        fixed_set = _normalize_rows(fixed_result)

        if gold_set != fixed_set:
            logger.info(
                "[Evaluator] record=%d gold_sql result mismatch: expected %d row(s), got %d row(s). "
                "Expected: %s | Actual: %s",
                record.id,
                len(record.gold_result),
                len(fixed_result),
                str(record.gold_result)[:300],
                str(fixed_result)[:300],
            )
            return 0.0, 0.0, (
                f"gold_sql result mismatch: expected {len(record.gold_result)} row(s), "
                f"got {len(fixed_result)} row(s)"
            ), fixed_result

        # --- Corruption check: logged but not a hard gate (captured in full_restore_score) ---
        corruption = _check_no_corruption(original_db_path, eval_sandbox, db_base_dir)
        if corruption:
            logger.info(
                "[Evaluator] record=%d corruption check found issues: %s",
                record.id, corruption,
            )
        else:
            logger.info(
                "[Evaluator] record=%d corruption check passed — no previously-good rows lost",
                record.id,
            )

        # --- Full restore check: is DB identical to original? ---
        full_match, diff = compare_databases(original_db_path, eval_sandbox)
        if full_match:
            logger.info("[Evaluator] record=%d full restore check: DB fully matches original", record.id)
            return 1.0, 1.0, "gold_result matches and DB fully restored to original", fixed_result

        logger.info(
            "[Evaluator] record=%d full restore check: partial restore. Diff summary: %s",
            record.id, str(diff)[:300],
        )
        return 1.0, 0.0, f"gold_result matches but DB not fully restored (diff: {diff})", fixed_result

    except Exception as exc:
        logger.warning("[Evaluator] fix evaluation error for record=%d: %s", record.id, exc)
        return 0.0, 0.0, f"Evaluation error: {exc}", []
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

    Uses SQLite ATTACH + EXCEPT to avoid loading large tables into Python memory.

    Returns None if no corruption, or a description string if corruption found.
    """
    import sqlite3

    conn = sqlite3.connect(str(original_db_path), timeout=60)
    try:
        conn.execute("ATTACH DATABASE ? AS fixed_db", (str(fixed_db_path),))

        tables = [
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]

        corruptions: list[str] = []
        for table in tables:
            # Fast count check first — avoids expensive EXCEPT on large unchanged tables
            try:
                orig_count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                fixed_count = conn.execute(f'SELECT COUNT(*) FROM fixed_db."{table}"').fetchone()[0]
            except Exception:
                corruptions.append(f"table '{table}' missing or unreadable in fixed DB")
                continue

            if orig_count != fixed_count:
                corruptions.append(
                    f"table '{table}': row count changed from {orig_count:,} to {fixed_count:,} after fix"
                )
                continue

            if orig_count > _MAX_DIFF_ROWS:
                # Table is large and count matches — skip EXCEPT to avoid multi-minute query.
                # The alteration only ever touches 1-2 rows in small tables, so this is safe.
                logger.debug(
                    "_check_no_corruption: skipping detailed check for large table '%s' (%d rows)",
                    table, orig_count,
                )
                continue

            # Count rows in original that are not in fixed (EXCEPT on disk — no Python memory spike)
            try:
                missing_count = conn.execute(
                    f'SELECT COUNT(*) FROM (SELECT * FROM "{table}" EXCEPT SELECT * FROM fixed_db."{table}")'
                ).fetchone()[0]
            except Exception:
                corruptions.append(f"table '{table}' unreadable during EXCEPT check")
                continue

            if missing_count:
                corruptions.append(
                    f"table '{table}': {missing_count} original row(s) missing after fix"
                )

        return "; ".join(corruptions) if corruptions else None
    finally:
        try:
            conn.execute("DETACH DATABASE fixed_db")
        except Exception:
            pass
        conn.close()


def quick_fix_check(
    record: DatasetRecord,
    fix_result: FixResult,
    sandbox_dir: Path | None = None,
    db_base_dir: Path | None = None,
) -> tuple[float, float, list, str]:
    """
    Apply the fix SQL to a fresh sandbox and return metrics for retry feedback.

    Returns:
        ``(gold_result_score, full_restore_score, actual_result_after_fix, description)``
    """
    gold_result_score, full_restore_score, description, actual_result = _evaluate_fix(
        record, fix_result, sandbox_dir=sandbox_dir, db_base_dir=db_base_dir
    )
    return gold_result_score, full_restore_score, actual_result, description


def evaluate(
    record: DatasetRecord,
    fix_result: FixResult,
    explanation_result: ExplanationResult,
    judge_llm: Union[LLMClient, GeminiClient],
    question_penalty: float | None = None,
    explanation_query_penalty: float | None = None,
    fix_query_penalty: float | None = None,
    ask_question_penalty: float | None = None,
    sandbox_dir: Path | None = None,
    db_base_dir: Path | None = None,
    pipeline_logger: PipelineLogger | None = None,
) -> EvaluationResult:
    """
    Evaluate both the ExplanationAgent and FixAgent outputs for one record.

    Args:
        record: The ground-truth dataset record.
        fix_result: The FixAgent's output.
        explanation_result: The ExplanationAgent's output.
        judge_llm: LLM client used to score the explanation.
        question_penalty: Deprecated. Use ask_question_penalty instead.
        explanation_query_penalty: Penalty per run_query call by ExplanationAgent.
        fix_query_penalty: Penalty per run_query call by FixAgent.
        ask_question_penalty: Penalty per ask_question call by FixAgent.
        sandbox_dir: Override for sandbox directory.
        db_base_dir: Override for database root directory.

    Returns:
        :class:`EvaluationResult` with all metrics filled in.
    """
    # Resolve per-tool penalties (ask_question_penalty falls back to legacy question_penalty)
    _ask_q_penalty = (
        ask_question_penalty
        if ask_question_penalty is not None
        else (question_penalty if question_penalty is not None else config.ASK_QUESTION_PENALTY)
    )
    _expl_q_penalty = (
        explanation_query_penalty if explanation_query_penalty is not None
        else config.EXPLANATION_QUERY_PENALTY
    )
    _fix_q_penalty = (
        fix_query_penalty if fix_query_penalty is not None
        else config.FIX_QUERY_PENALTY
    )
    error: str | None = None

    # ── Alteration type evaluation (systematic comparison) ───────────────────
    if explanation_result.is_fallback:
        alteration_type_score = 0.0
        logger.warning(
            "[Evaluator] record=%d ExplanationAgent fallback — alteration_type_score=0",
            record.id,
        )
    else:
        alteration_type_score = _evaluate_alteration_type(record, explanation_result)

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
                record, explanation_result, judge_llm, pipeline_logger=pipeline_logger
            )
        except Exception as exc:
            error = str(exc)
            explanation_score = 0.0
            explanation_reasoning = f"Judge evaluation error: {exc}"
            logger.warning("[Evaluator] explanation judge error for record=%d: %s", record.id, exc)

    # ── Fix evaluation ───────────────────────────────────────────────────────
    if fix_result.is_fallback:
        logger.warning(
            "[Evaluator] record=%d FixAgent fallback — DB comparison will likely fail",
            record.id,
        )
    try:
        gold_result_score, full_restore_score, fix_description, _actual_result = _evaluate_fix(
            record, fix_result, sandbox_dir=sandbox_dir, db_base_dir=db_base_dir
        )
    except Exception as exc:
        if error is None:
            error = str(exc)
        gold_result_score = 0.0
        full_restore_score = 0.0
        fix_description = f"Fix evaluation error: {exc}"
        logger.warning("[Evaluator] fix eval error for record=%d: %s", record.id, exc)

    # ── Scoring ──────────────────────────────────────────────────────────────
    expl_penalty = round(_expl_q_penalty * explanation_result.query_turns, 4)
    fix_q_penalty_total = round(_fix_q_penalty * fix_result.query_turns, 4)
    ask_q_penalty_total = round(_ask_q_penalty * fix_result.questions_asked, 4)
    total_penalty = round(expl_penalty + fix_q_penalty_total + ask_q_penalty_total, 4)
    final_score = round(max(0.0, gold_result_score - total_penalty), 4)

    breakdown = {
        "explanation_query_penalty": expl_penalty,
        "fix_query_penalty": fix_q_penalty_total,
        "ask_question_penalty": ask_q_penalty_total,
    }

    logger.info(
        "[Evaluator] record=%d  alteration_type_score=%.1f  explanation_score=%.2f  "
        "gold_result_score=%.1f  full_restore_score=%.1f  "
        "expl_queries=%d  fix_queries=%d  questions=%d  "
        "penalty=%.4f (expl=%.4f fix_q=%.4f ask_q=%.4f)  final=%.4f",
        record.id, alteration_type_score, explanation_score,
        gold_result_score, full_restore_score,
        explanation_result.query_turns, fix_result.query_turns, fix_result.questions_asked,
        total_penalty, expl_penalty, fix_q_penalty_total, ask_q_penalty_total, final_score,
    )

    return EvaluationResult(
        record_id=record.id,
        db_id=record.db_id,
        alteration_type_score=alteration_type_score,
        explanation_score=round(explanation_score, 4),
        explanation_reasoning=explanation_reasoning,
        gold_result_score=gold_result_score,
        full_restore_score=full_restore_score,
        fix_description=fix_description,
        questions_asked=fix_result.questions_asked,
        explanation_query_turns=explanation_result.query_turns,
        fix_query_turns=fix_result.query_turns,
        tool_penalty_breakdown=breakdown,
        tool_penalty=total_penalty,
        question_penalty=total_penalty,
        base_score=gold_result_score,
        final_score=final_score,
        error=error,
    )
