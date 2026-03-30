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
from evaluator import evaluate
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
            "ExplanationAgent done  turns=%d  root_cause=%s",
            explanation.turns_used, explanation.root_cause,
        )
        sample_log.explanation_result = explanation.model_dump()

        # ── Step 2: FixAgent ──────────────────────────────────────────────────
        fix_agent = FixAgent(
            record=record,
            explanation=explanation,
            llm=fix_llm,
            max_turns=max_fix_turns,
            question_penalty=question_penalty,
            pipeline_logger=pipeline_logger,
        )
        fix = fix_agent.run(user_agent)
        logger.info(
            "FixAgent done  questions_asked=%d  confidence=%.2f  fix_sql=%s",
            fix.questions_asked, fix.confidence, fix.fix_sql,
        )
        sample_log.fix_result = fix.model_dump()

        # ── Step 3: Evaluator ─────────────────────────────────────────────────
        evaluation = evaluate(
            record=record,
            fix_result=fix,
            explanation_result=explanation,
            judge_llm=judge_llm,
            question_penalty=question_penalty,
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
        "Record id=%d done in %.1fs — fix_score=%.1f  explanation_score=%.2f  final_score=%.4f",
        record.id, elapsed,
        evaluation.fix_score, evaluation.explanation_score, evaluation.final_score,
    )

    run_result = RunResult(
        record_id=record.id,
        db_id=record.db_id,
        explanation=explanation,
        fix=fix,
        evaluation=evaluation,
    )
    return run_result, sample_log
