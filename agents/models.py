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
    is_aggregation: bool = False


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
        description="What physically changed in the data: which table, which rows/columns, and what the new values are (required when action='done')",
    )
    sql_impact: str | None = Field(
        None,
        description="Why the alteration causes the SQL query to return different results — which specific condition (WHERE, JOIN, HAVING, DISTINCT, etc.) the altered data no longer satisfies (required when action='done')",
    )
    alteration_type: str | None = Field(
        None,
        description="Either 'deletion' or 'modification' (required when action='done')",
    )


class FixAgentStep(BaseModel):
    """One reasoning step produced by the FixAgent LLM."""

    action: str = Field(
        description=(
            "'run_query' to inspect the altered database directly (incurs penalty), "
            "'ask_question' to gather more info from the database owner (incurs penalty), "
            "or 'done' to submit the fix SQL"
        )
    )
    sql: str | None = Field(
        None,
        description="SELECT query to run on the altered database (required when action='run_query')",
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
    sql_impact: str = ""    # why the alteration causes the SQL query to return different results
    alteration_type: str
    turns_used: int
    query_turns: int = 0    # number of run_query calls made
    conversation: list[ConversationTurn]
    is_fallback: bool = False


class FixResult(BaseModel):
    """Final output of the FixAgent for one dataset record."""

    record_id: int
    fix_sql: str
    reasoning: str
    questions_asked: int
    query_turns: int = 0    # number of run_query calls made
    retry_count: int = 0    # how many retries were used (0 = first attempt succeeded/failed)
    conversation: list[ConversationTurn]
    is_fallback: bool = False


# ── Evaluation ───────────────────────────────────────────────────────────────

class EvaluationResult(BaseModel):
    """Evaluation of the FixAgent's output and the ExplanationAgent's explanation."""

    record_id: int
    db_id: str
    # Explanation quality
    alteration_type_score: float    # 0 or 1, systematic comparison of predicted vs actual type
    explanation_score: float        # 0.0, 0.5, or 1.0 from LLM-as-judge
    explanation_reasoning: str      # judge's reasoning
    # Fix correctness
    gold_result_score: float        # 0 or 1: does gold_sql return gold_result after fix?
    full_restore_score: float       # 0 or 1: is DB fully restored to original?
    fix_description: str            # human-readable description of fix evaluation result
    # Tool usage counts
    questions_asked: int
    explanation_query_turns: int = 0        # run_query calls by ExplanationAgent
    fix_query_turns: int = 0                # run_query calls by FixAgent
    # Penalty breakdown
    tool_penalty_breakdown: dict = Field(default_factory=dict)
    tool_penalty: float             # total deduction across all tool uses
    question_penalty: float = 0.0  # backward-compat alias (equal to tool_penalty)
    retry_multiplier: float = 1.0  # 0.5 if fix used a retry attempt, else 1.0
    base_score: float               # same as gold_result_score
    final_score: float              # max(0, gold_result_score - tool_penalty) * retry_multiplier
    error: str | None = None


class RunResult(BaseModel):
    """Complete output for one dataset record across all three agents."""

    record_id: int
    db_id: str
    # ── Sample snapshot (makes results.json self-contained) ──────────────
    question: str = ""
    evidence: str = ""
    gold_sql: str = ""
    gold_result: list[dict[str, Any]] = Field(default_factory=list)
    alteration_type: str = ""
    altering_sql: str = ""
    altered_result: list[dict[str, Any]] = Field(default_factory=list)
    alteration_explanation: str = ""
    follow_up_question: str = ""
    is_aggregation: bool = False
    # ── Agent outputs ─────────────────────────────────────────────────────
    explanation: ExplanationResult
    fix: FixResult
    evaluation: EvaluationResult
