"""
Runner — orchestrates the three-agent pipeline for a single dataset record.

Flow for each record:
  1. Create altered sandbox database and compute diff for UserAgent.
  2. ExplanationAgent investigates via direct DB queries (autonomous, no user interaction).
  3. FixAgent receives the explanation and generates SQL to restore the database.
  4. Evaluator judges the explanation (LLM-as-judge) and the fix (relaxed DB comparison).
  5. Sandbox is cleaned up.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from database_utils import (
    compute_structured_diff,
    create_altered_sandbox,
    destroy_sandbox,
    format_diff_as_text,
    get_db_path,
)
import json

from evaluator import evaluate, quick_fix_check
from explanation_agent import ExplanationAgent
from fix_agent import FixAgent
from llm_client import GeminiClient, LLMClient
from models import DatasetRecord, RunResult
from sample_logger import PipelineLogger, SampleLog
from user_agent import UserAgent

import config

logger = logging.getLogger(__name__)


def run_record(
    record: DatasetRecord,
    user_llm: Union[LLMClient, GeminiClient],
    explanation_llm: Union[LLMClient, GeminiClient],
    fix_llm: Union[LLMClient, GeminiClient],
    judge_llm: Union[LLMClient, GeminiClient],
    db_base_dir: Path | None = None,
    sandbox_dir: Path | None = None,
    max_explanation_turns: int | None = None,
    max_fix_turns: int | None = None,
    question_penalty: float | None = None,
    max_fix_retries: int | None = None,
    explanation_query_penalty: float | None = None,
    fix_query_penalty: float | None = None,
    ask_question_penalty: float | None = None,
) -> tuple[RunResult, SampleLog]:
    """
    Run the full three-agent pipeline for one dataset record.

    Returns:
        Tuple of :class:`RunResult` and :class:`SampleLog` with full event log.
    """
    t_start = time.perf_counter()
    timestamp_start = datetime.now(timezone.utc).isoformat()
    pipeline_logger = PipelineLogger()

    logger.info(
        "━━━ Processing record id=%d  db=%s ━━━",
        record.id, record.db_id,
    )

    # Verify the database file exists before starting
    db_path = get_db_path(record.db_id, db_base_dir)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database '{record.db_id}' not found at: {db_path}"
        )

    # ── Create sandbox and compute diff ────────────────────────────────────
    sandbox_path = create_altered_sandbox(
        record.db_id,
        record.altering_sql,
        sandbox_dir=sandbox_dir,
        db_base_dir=db_base_dir,
    )
    logger.info("Created sandbox for record=%d at %s", record.id, sandbox_path)

    original_db_path = get_db_path(record.db_id, db_base_dir)
    structured_diff = compute_structured_diff(original_db_path, sandbox_path)
    diff_text = format_diff_as_text(structured_diff)
    diff_table_count = len(structured_diff.get("tables", {}))
    logger.info(
        "Computed DB diff for record=%d (%d table(s) with differences)",
        record.id, diff_table_count,
    )

    # ── Initialize UserAgent (no SQL access, receives diff as text) ────────
    user_agent = UserAgent(
        record=record,
        llm=user_llm,
        diff_text=diff_text,
        pipeline_logger=pipeline_logger,
    )

    # Prepare SampleLog shell
    sample_log = SampleLog(
        record_id=record.id,
        db_id=record.db_id,
        question=record.question,
        evidence=record.evidence,
        gold_sql=record.gold_sql,
        gold_result=record.gold_result,
        alteration_type=record.alteration_type,
        altering_sql=record.altering_sql,
        altered_result=record.altered_result,
        alteration_explanation=record.alteration_explanation,
        follow_up_question=record.follow_up_question,
        sandbox_path=str(sandbox_path),
        diff_tables_count=diff_table_count,
        diff_text=diff_text,
        timestamp_start=timestamp_start,
    )

    _max_fix_retries = max_fix_retries if max_fix_retries is not None else config.MAX_FIX_RETRIES

    try:
        # ── Step 1: ExplanationAgent ─────────────────────────────────────────
        explanation_agent = ExplanationAgent(
            record=record,
            llm=explanation_llm,
            altered_db_path=sandbox_path,
            max_turns=max_explanation_turns,
            pipeline_logger=pipeline_logger,
        )
        explanation = explanation_agent.run()
        logger.info(
            "ExplanationAgent done  turns=%d  query_turns=%d  alteration_type=%s",
            explanation.turns_used, explanation.query_turns, explanation.alteration_type,
        )
        sample_log.explanation_result = explanation.model_dump()

        # ── Step 2: FixAgent ──────────────────────────────────────────────────
        _effective_question_penalty = ask_question_penalty if ask_question_penalty is not None else question_penalty
        fix_agent = FixAgent(
            record=record,
            explanation=explanation,
            llm=fix_llm,
            max_turns=max_fix_turns,
            question_penalty=_effective_question_penalty,
            pipeline_logger=pipeline_logger,
            altered_db_path=sandbox_path,
        )
        fix = fix_agent.run(user_agent)
        logger.info(
            "FixAgent done  questions_asked=%d  query_turns=%d  fix_sql=%s",
            fix.questions_asked, fix.query_turns, fix.fix_sql,
        )

        # ── Retry if first fix fails the gold check ───────────────────────────
        if _max_fix_retries > 0 and not fix.is_fallback:
            gold_pre, _, actual_after_fix, _ = quick_fix_check(
                record=record,
                fix_result=fix,
                sandbox_dir=sandbox_dir,
                db_base_dir=db_base_dir,
            )
            if gold_pre == 0.0:
                logger.info(
                    "FixAgent initial fix failed gold check for record=%d — attempting retry 1/%d",
                    record.id, _max_fix_retries,
                )
                retry_context = (
                    "Your previous fix SQL was INCORRECT.\n\n"
                    f"Previous fix SQL:\n{fix.fix_sql}\n\n"
                    f"After applying it, running the gold SQL returned:\n"
                    f"{json.dumps(actual_after_fix, default=str)}\n\n"
                    f"But the expected result is:\n"
                    f"{json.dumps(record.gold_result, default=str)}\n\n"
                    "Please produce a new, corrected fix SQL."
                )
                fix_agent_retry = FixAgent(
                    record=record,
                    explanation=explanation,
                    llm=fix_llm,
                    max_turns=max_fix_turns,
                    question_penalty=_effective_question_penalty,
                    pipeline_logger=pipeline_logger,
                    altered_db_path=sandbox_path,
                )
                fix_retry = fix_agent_retry.run(user_agent, retry_context=retry_context)
                fix = fix_retry.model_copy(update={"retry_count": 1})
                logger.info(
                    "FixAgent retry complete for record=%d  fix_sql=%s",
                    record.id, fix.fix_sql,
                )

        sample_log.fix_result = fix.model_dump()

        # ── Step 3: Evaluator ─────────────────────────────────────────────────
        evaluation = evaluate(
            record=record,
            fix_result=fix,
            explanation_result=explanation,
            judge_llm=judge_llm,
            question_penalty=question_penalty,
            explanation_query_penalty=explanation_query_penalty,
            fix_query_penalty=fix_query_penalty,
            ask_question_penalty=ask_question_penalty,
            sandbox_dir=sandbox_dir,
            db_base_dir=db_base_dir,
            pipeline_logger=pipeline_logger,
        )
        sample_log.evaluation_result = evaluation.model_dump()

    except Exception as exc:
        sample_log.status = "error"
        sample_log.error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Record id=%d pipeline error: %s", record.id, exc, exc_info=True,
        )
        return None, sample_log

    finally:
        destroy_sandbox(sandbox_path)
        logger.info("Sandbox cleaned up for record=%d", record.id)

        # Finalize sample log
        elapsed = round(time.perf_counter() - t_start, 2)
        sample_log.total_duration_seconds = elapsed
        sample_log.events = [asdict(e) for e in pipeline_logger.events]

    logger.info(
        "Record id=%d done in %.1fs — gold_result_score=%.1f  full_restore_score=%.1f  "
        "alteration_type_score=%.1f  explanation_score=%.2f  "
        "tool_penalty=%.4f  final_score=%.4f  retry_count=%d",
        record.id, elapsed,
        evaluation.gold_result_score, evaluation.full_restore_score,
        evaluation.alteration_type_score, evaluation.explanation_score,
        evaluation.tool_penalty, evaluation.final_score, fix.retry_count,
    )

    run_result = RunResult(
        record_id=record.id,
        db_id=record.db_id,
        explanation=explanation,
        fix=fix,
        evaluation=evaluation,
    )
    return run_result, sample_log
