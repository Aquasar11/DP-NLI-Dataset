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
    """A single entry from the BIRD train.json or dev.json."""
    db_id: str
    question: str
    evidence: str
    SQL: str


class SpiderSample(BaseModel):
    """A single entry from a Spider train_spider.json or test.json."""
    db_id: str
    question: str
    query: str  # Spider uses 'query'; BIRD uses 'SQL'

    def to_bird_sample(self) -> "BirdSample":
        """Normalize to BirdSample for uniform pipeline processing."""
        return BirdSample(
            db_id=self.db_id,
            question=self.question,
            evidence="",   # Spider has no evidence field
            SQL=self.query,
        )


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


class LLMAlterationResponse(BaseModel):
    """Structured output from the LLM for Step 1: generating the altering SQL."""
    altering_sql: str = Field(
        description="SQL statement(s) to alter the database (DELETE or UPDATE)"
    )
    target_columns: list[str] = Field(
        description=(
            "Which column(s) were modified. For DELETE: ['all']. "
            "For MODIFY: the specific column name(s) that were changed."
        )
    )
    explanation: str = Field(
        description="Explanation of why this alteration causes the target records to disappear from the query result"
    )


class LLMFollowUpResponse(BaseModel):
    """Structured output from the LLM for Step 2: generating follow-up Q&A."""
    follow_up_question: str = Field(
        description="A natural follow-up question a user would ask when seeing the altered (wrong) output"
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
    target_columns: list[str]
    altering_sql: str
    altered_result: list[dict[str, Any]]
    alteration_explanation: str
    follow_up_question: str
    is_aggregation: bool = False