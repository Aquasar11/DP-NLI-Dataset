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
        description="'ask_question' to gather more info, or 'done' to conclude"
    )
    question: str | None = Field(
        None,
        description="Question to ask the UserAgent (required when action='ask_question')",
    )
    explanation: str | None = Field(
        None,
        description="Full explanation of what happened to the data (required when action='done')",
    )
    root_cause: str | None = Field(
        None,
        description="Concise one-sentence root cause (required when action='done')",
    )


class AnswerAgentStep(BaseModel):
    """One reasoning step produced by the AnswerAgent LLM."""

    action: str = Field(
        description="'ask_question' to gather more info (incurs penalty), or 'done' to submit"
    )
    question: str | None = Field(
        None,
        description="Question for the UserAgent (required when action='ask_question')",
    )
    predicted_altering_sql: str | None = Field(
        None,
        description="The exact SQL DML that caused the alteration (required when action='done')",
    )
    confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1 (required when action='done')",
    )
    reasoning: str | None = Field(
        None,
        description="Step-by-step reasoning behind the prediction (required when action='done')",
    )


class UserAgentStep(BaseModel):
    """One inner-loop decision step produced by the UserAgent LLM."""

    action: str = Field(
        description="'run_query' to execute a SELECT, or 'respond' to give a final answer"
    )
    sql: str | None = Field(
        None,
        description="SELECT query to run (required when action='run_query')",
    )
    answer: str | None = Field(
        None,
        description="Final text answer to the question (required when action='respond')",
    )


# ── Agent run results ────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    """A single turn in an agent–user conversation."""

    role: str    # "investigator" | "user"
    content: str


class ExplanationResult(BaseModel):
    """Final output of the ExplanationAgent for one dataset record."""

    record_id: int
    explanation: str
    root_cause: str
    turns_used: int
    conversation: list[ConversationTurn]


class AnswerResult(BaseModel):
    """Final output of the AnswerAgent for one dataset record."""

    record_id: int
    predicted_altering_sql: str
    confidence: float
    reasoning: str
    questions_asked: int
    conversation: list[ConversationTurn]


# ── Evaluation ───────────────────────────────────────────────────────────────

class EvaluationResult(BaseModel):
    """Evaluation of the AnswerAgent's prediction against ground truth."""

    record_id: int
    db_id: str
    ground_truth_sql: str
    predicted_sql: str
    exact_match: bool
    semantic_match: bool
    questions_asked: int
    question_penalty: float
    base_score: float   # 1.0 if correct, 0.0 otherwise
    final_score: float  # base_score − question_penalty × questions_asked
    error: str | None = None


class RunResult(BaseModel):
    """Complete output for one dataset record across all three agents."""

    record_id: int
    db_id: str
    explanation: ExplanationResult
    answer: AnswerResult
    evaluation: EvaluationResult
