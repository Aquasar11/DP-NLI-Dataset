"""
Runner — orchestrates the three-agent pipeline for a single dataset record.

Flow for each record:
  1. Initialize UserAgent (creates altered sandbox database).
  2. ExplanationAgent investigates via Q&A with UserAgent.
  3. AnswerAgent receives the explanation and predicts the DML SQL.
  4. Evaluator scores the AnswerAgent's prediction.
  5. UserAgent sandbox is cleaned up.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Union

from answer_agent import AnswerAgent
from database_utils import get_db_path
from evaluator import evaluate
from explanation_agent import ExplanationAgent
from llm_client import GeminiClient, LLMClient
from models import DatasetRecord, RunResult
from user_agent import UserAgent

import config

logger = logging.getLogger(__name__)


def run_record(
    record: DatasetRecord,
    llm: Union[LLMClient, GeminiClient],
    db_base_dir: Path | None = None,
    sandbox_dir: Path | None = None,
    max_explanation_turns: int | None = None,
    max_answer_turns: int | None = None,
    question_penalty: float | None = None,
) -> RunResult:
    """
    Run the full three-agent pipeline for one dataset record.

    Args:
        record: The dataset record to process.
        llm: Shared LLM client instance (thread-safe; each agent call is independent).
        db_base_dir: Override for directory containing database files.
        sandbox_dir: Override for sandbox directory.
        max_explanation_turns: Override for ExplanationAgent turn limit.
        max_answer_turns: Override for AnswerAgent turn limit.
        question_penalty: Override for per-question score penalty.

    Returns:
        :class:`RunResult` with explanation, answer, and evaluation.
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
        llm=llm,
        db_base_dir=db_base_dir,
        sandbox_dir=sandbox_dir,
    )

    try:
        # ── Step 1: ExplanationAgent ─────────────────────────────────────────
        explanation_agent = ExplanationAgent(
            record=record,
            llm=llm,
            max_turns=max_explanation_turns,
        )
        explanation = explanation_agent.run(user_agent)
        logger.info(
            "ExplanationAgent done  turns=%d  root_cause=%s",
            explanation.turns_used, explanation.root_cause,
        )

        # ── Step 2: AnswerAgent ───────────────────────────────────────────────
        answer_agent = AnswerAgent(
            record=record,
            explanation=explanation,
            llm=llm,
            max_turns=max_answer_turns,
            question_penalty=question_penalty,
        )
        answer = answer_agent.run(user_agent)
        logger.info(
            "AnswerAgent done  questions_asked=%d  confidence=%.2f  "
            "predicted_sql=%s",
            answer.questions_asked, answer.confidence, answer.predicted_altering_sql,
        )

        # ── Step 3: Evaluator ─────────────────────────────────────────────────
        evaluation = evaluate(
            record=record,
            answer=answer,
            question_penalty=question_penalty,
            sandbox_dir=sandbox_dir,
            db_base_dir=db_base_dir,
        )

    finally:
        user_agent.cleanup()

    elapsed = round(time.perf_counter() - t_start, 2)
    logger.info(
        "Record id=%d done in %.1fs — exact=%s  semantic=%s  final_score=%.4f",
        record.id, elapsed,
        evaluation.exact_match, evaluation.semantic_match, evaluation.final_score,
    )

    return RunResult(
        record_id=record.id,
        db_id=record.db_id,
        explanation=explanation,
        answer=answer,
        evaluation=evaluation,
    )
