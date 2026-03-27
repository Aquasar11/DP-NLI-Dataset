"""
Runner — orchestrates the three-agent pipeline for a single dataset record.

Flow for each record:
  1. Initialize UserAgent (creates altered sandbox database).
  2. ExplanationAgent investigates via direct DB queries and Q&A with UserAgent.
  3. FixAgent receives the explanation and generates SQL to restore the database.
  4. Evaluator judges the explanation (LLM-as-judge) and the fix (DB comparison).
  5. UserAgent sandbox is cleaned up.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Union

from database_utils import get_db_path
from evaluator import evaluate
from explanation_agent import ExplanationAgent
from fix_agent import FixAgent
from llm_client import GeminiClient, LLMClient
from models import DatasetRecord, RunResult
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
) -> RunResult:
    """
    Run the full three-agent pipeline for one dataset record.

    Args:
        record: The dataset record to process.
        user_llm: LLM client for the UserAgent.
        explanation_llm: LLM client for the ExplanationAgent.
        fix_llm: LLM client for the FixAgent.
        judge_llm: LLM client for the explanation judge.
        db_base_dir: Override for directory containing database files.
        sandbox_dir: Override for sandbox directory.
        max_explanation_turns: Override for ExplanationAgent turn limit.
        max_fix_turns: Override for FixAgent turn limit.
        question_penalty: Override for per-question score penalty.

    Returns:
        :class:`RunResult` with explanation, fix, and evaluation.
    """
    t_start = time.perf_counter()
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

    user_agent = UserAgent(
        record=record,
        llm=user_llm,
        db_base_dir=db_base_dir,
        sandbox_dir=sandbox_dir,
    )

    try:
        # ── Step 1: ExplanationAgent ─────────────────────────────────────────
        explanation_agent = ExplanationAgent(
            record=record,
            llm=explanation_llm,
            altered_db_path=user_agent.sandbox_path,
            max_turns=max_explanation_turns,
        )
        explanation = explanation_agent.run(user_agent)
        logger.info(
            "ExplanationAgent done  turns=%d  root_cause=%s",
            explanation.turns_used, explanation.root_cause,
        )

        # ── Step 2: FixAgent ──────────────────────────────────────────────────
        fix_agent = FixAgent(
            record=record,
            explanation=explanation,
            llm=fix_llm,
            max_turns=max_fix_turns,
            question_penalty=question_penalty,
        )
        fix = fix_agent.run(user_agent)
        logger.info(
            "FixAgent done  questions_asked=%d  confidence=%.2f  fix_sql=%s",
            fix.questions_asked, fix.confidence, fix.fix_sql,
        )

        # ── Step 3: Evaluator ─────────────────────────────────────────────────
        evaluation = evaluate(
            record=record,
            fix_result=fix,
            explanation_result=explanation,
            judge_llm=judge_llm,
            question_penalty=question_penalty,
            sandbox_dir=sandbox_dir,
            db_base_dir=db_base_dir,
        )

    finally:
        user_agent.cleanup()

    elapsed = round(time.perf_counter() - t_start, 2)
    logger.info(
        "Record id=%d done in %.1fs — db_match=%s  explanation_score=%.2f  final_score=%.4f",
        record.id, elapsed,
        evaluation.db_match, evaluation.explanation_score, evaluation.final_score,
    )

    return RunResult(
        record_id=record.id,
        db_id=record.db_id,
        explanation=explanation,
        fix=fix,
        evaluation=evaluation,
    )
