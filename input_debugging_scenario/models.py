"""
Pydantic data models for the Data Debugging Dataset Generator.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class AlterationType(str, Enum):
    DELETE = "delete"   # Remove entire row(s)
    MODIFY = "modify"   # Change specific column values so the row no longer matches


# ── Input Models (BIRD dataset) ───────────────────────────────────────────────

class BirdSample(BaseModel):
    """A single entry from the BIRD train.json."""
    db_id: str
    question: str
    evidence: str
    SQL: str


class TableSchema(BaseModel):
    """Schema information for a single database."""
    db_id: str
    table_names_original: list[str]
    table_names: list[str]
    column_names_original: list[list[Any]]    # [[table_idx, col_name], ...]
    column_names: list[list[Any]]
    column_types: list[str]
    primary_keys: list[int]
    foreign_keys: list[list[int]]


# ── Intermediate Models ───────────────────────────────────────────────────────

class AlterationDecision(BaseModel):
    """The randomly-chosen alteration strategy for a sample."""
    alteration_type: AlterationType
    target_record_indices: list[int] = Field(
        description="Indices into the gold_result list identifying which rows to target"
    )
    target_columns: list[str] | None = Field(
        default=None,
        description="For MODIFY type: which columns to change. None means all columns.",
    )


class LLMAlterationResponse(BaseModel):
    """Structured output from the LLM for Step 1: generating the altering SQL."""
    altering_sql: str = Field(
        description="SQL statement(s) to alter the database (DELETE or UPDATE)"
    )
    explanation: str = Field(
        description="Explanation of why this alteration causes the target records to disappear from the query result"
    )


class LLMFollowUpResponse(BaseModel):
    """Structured output from the LLM for Step 2: generating follow-up Q&A."""
    follow_up_question: str = Field(
        description="A natural follow-up question a user would ask when seeing the altered (wrong) output"
    )
    gold_explanation: str = Field(
        description="The gold explanation answering the follow-up question, referencing the specific data change"
    )
    gold_fix: str = Field(
        description="SQL statement(s) to reverse the alteration and restore the original database state"
    )


class LLMFixResponse(BaseModel):
    """Structured output from the LLM for Step 3 retry: corrected fix SQL."""
    gold_fix: str = Field(
        description="Corrected SQL statement(s) to reverse the alteration and restore the original database state"
    )
    explanation: str = Field(
        description="Why this corrected fix SQL works"
    )


# ── Validation Models ─────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    """Result of validating an alteration."""
    is_valid: bool
    error_message: str | None = None
    missing_targeted: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Targeted records that were successfully removed",
    )
    still_present_targeted: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Targeted records that are still present (should be empty for valid)",
    )
    unintended_missing: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Non-targeted records that were unintentionally removed",
    )


# ── Output Model ──────────────────────────────────────────────────────────────

class DatasetRecord(BaseModel):
    """A single record in the generated dataset."""
    id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    gold_result: list[dict[str, Any]]
    alteration_type: AlterationType
    targeted_records: list[dict[str, Any]]
    altering_sql: str
    altered_result: list[dict[str, Any]]
    alteration_explanation: str
    follow_up_question: str
    gold_explanation: str
    gold_fix: str
