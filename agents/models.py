"""
Pydantic data models for the multi-agent data debugging framework.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Input: dataset record ────────────────────────────────────────────────────

class DatasetRecord(BaseModel):
    """A single record from the generated debugging dataset."""

    id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    gold_result: list[dict[str, Any]]
    alteration_type: str
    targeted_records: list[dict[str, Any]]
    target_columns: list[str]
    altering_sql: str
    altered_result: list[dict[str, Any]]
    alteration_explanation: str
    follow_up_question: str


# ── Structured LLM action schemas ────────────────────────────────────────────

class ExplanationAgentStep(BaseModel):
    """One reasoning step produced by the ExplanationAgent LLM."""

    action: str = Field(
        description=(
            "'run_query' to inspect the database directly, or "
            "'done' to submit the final explanation"
        )
    )
    sql: str | None = Field(
        None,
        description="SELECT query to run directly on the database (required when action='run_query')",
    )
    explanation: str | None = Field(
        None,
        description="Full explanation of what changed in the data (required when action='done')",
    )
    root_cause: str | None = Field(
        None,
        description="Concise one-sentence root cause (required when action='done')",
    )


class FixAgentStep(BaseModel):
    """One reasoning step produced by the FixAgent LLM."""

    action: str = Field(
        description="'ask_question' to gather more info (incurs penalty), or 'done' to submit"
    )
    question: str | None = Field(
        None,
        description="Question for the database owner (required when action='ask_question')",
    )
    fix_sql: str | None = Field(
        None,
        description=(
            "SQL statement(s) that will RESTORE the database to its original state "
            "(required when action='done')"
        ),
    )
    reasoning: str | None = Field(
        None,
        description="Step-by-step reasoning behind the fix (required when action='done')",
    )


# ── Agent run results ────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    """A single turn in an agent–user conversation."""

    role: str    # "ExplanationAgent" | "FixAgent" | "UserAgent"
    content: str


class ExplanationResult(BaseModel):
    """Final output of the ExplanationAgent for one dataset record."""

    record_id: int
    explanation: str
    root_cause: str
    turns_used: int
    conversation: list[ConversationTurn]
    is_fallback: bool = False


class FixResult(BaseModel):
    """Final output of the FixAgent for one dataset record."""

    record_id: int
    fix_sql: str
    reasoning: str
    questions_asked: int
    conversation: list[ConversationTurn]
    is_fallback: bool = False


# ── Evaluation ───────────────────────────────────────────────────────────────

class EvaluationResult(BaseModel):
    """Evaluation of the FixAgent's output and the ExplanationAgent's explanation."""

    record_id: int
    db_id: str
    # Explanation quality (LLM-as-judge)
    explanation_score: float        # 0.0–1.0 from judge LLM
    explanation_reasoning: str      # judge's reasoning
    # Fix correctness (relaxed evaluation)
    fix_score: float                # 0.0 = fail, 1.0 = gold_result OK + no corruption, 1.5 = full restore
    fix_description: str            # human-readable description of fix evaluation result
    # Scoring
    questions_asked: int
    question_penalty: float         # total deduction (penalty × questions_asked)
    base_score: float               # same as fix_score
    final_score: float              # base_score − question_penalty, clamped to [0, 1]
    error: str | None = None


class RunResult(BaseModel):
    """Complete output for one dataset record across all three agents."""

    record_id: int
    db_id: str
    explanation: ExplanationResult
    fix: FixResult
    evaluation: EvaluationResult
