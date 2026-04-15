"""
Per-sample structured logging for the Data Debugging pipeline.

Every sample's full lifecycle — LLM inputs/outputs, alteration decisions,
validation results, and final outcome — is captured in a SampleLog and
written incrementally to JSONL files.  Successful samples go to
``sample_logs.jsonl``; skipped / failed samples go to
``failed_samples.jsonl``.  At pipeline end, ``consolidate()`` writes
final JSON arrays for convenient look-up by ``id`` or ``sample_idx``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Atomic log records ─────────────────────────────────────────────────────


@dataclass
class LLMCallLog:
    """Full record of a single LLM API call."""

    step: str  # e.g. "alteration_step1", "followup_step2"
    attempt: int
    system_prompt: str
    user_prompt: str
    raw_response: str | None
    parsed_response: dict[str, Any] | None
    success: bool
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class AttemptLog:
    """One Step-1 attempt: LLM call + sandbox execution + validation."""

    attempt: int
    llm_call: LLMCallLog
    altering_sql: str
    sandbox_execute_success: bool
    sandbox_execute_error: str | None
    gold_sql_on_sandbox_success: bool
    gold_sql_on_sandbox_error: str | None
    altered_result: list[dict[str, Any]]
    validation_is_valid: bool
    validation_error: str | None
    validation_still_present: list[dict[str, Any]]
    validation_unintended_missing: list[dict[str, Any]]


@dataclass
class AlterationDecisionLog:
    """Captures every factor that went into the alteration decision."""

    alteration_type: str  # "delete" | "modify" | "insert"
    num_result_rows: int
    max_targets_config: int
    num_targets_chosen: int
    target_record_indices: list[int]
    targeted_records: list[dict[str, Any]]
    delete_probability_config: float
    insert_probability_config: float = 0.0
    force_non_insert: bool = False
    random_draw: float = 0.0


# ── Top-level per-sample record ────────────────────────────────────────────


@dataclass
class SampleLog:
    """Complete log for one pipeline sample — success or failure."""

    # Identity
    sample_idx: int
    record_id: int | None  # assigned only on success
    db_id: str
    question: str
    evidence: str

    # Gold
    gold_sql: str
    gold_result: list[dict[str, Any]]
    gold_result_row_count: int

    # Decision
    alteration_decision: AlterationDecisionLog | None

    # Step 1
    step1_attempts: list[AttemptLog]
    step1_final_altering_sql: str | None
    step1_final_explanation: str | None
    step1_target_columns: list[str] | None  # reported by LLM
    step1_total_attempts: int
    step1_passed: bool

    # Step 2
    step2_llm_call: LLMCallLog | None
    step2_follow_up_question: str | None
    step2_passed: bool

    # Outcome
    status: str  # success | skipped_empty | skipped_error | skipped_aggregate
    #               failed_validation | failed_llm
    skip_reason: str | None = None
    total_duration_seconds: float = 0.0
    timestamp_start: str = ""
    # Set when this run is a fallback after INSERT failed all its retries
    insert_fallback_warning: str | None = None


# ── Helper factories ───────────────────────────────────────────────────────


def make_llm_call_log(
    *,
    step: str,
    attempt: int,
    system_prompt: str,
    user_prompt: str,
    raw_response: str | None,
    parsed_response: dict[str, Any] | None,
    success: bool,
    error: str | None = None,
    duration_seconds: float = 0.0,
) -> LLMCallLog:
    return LLMCallLog(
        step=step,
        attempt=attempt,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_response=raw_response,
        parsed_response=parsed_response,
        success=success,
        error=error,
        duration_seconds=duration_seconds,
    )


def make_attempt_log(
    *,
    attempt: int,
    llm_call: LLMCallLog,
    altering_sql: str,
    sandbox_execute_success: bool,
    sandbox_execute_error: str | None,
    gold_sql_on_sandbox_success: bool,
    gold_sql_on_sandbox_error: str | None,
    altered_result: list[dict[str, Any]],
    validation_is_valid: bool,
    validation_error: str | None,
    validation_still_present: list[dict[str, Any]],
    validation_unintended_missing: list[dict[str, Any]],
) -> AttemptLog:
    return AttemptLog(
        attempt=attempt,
        llm_call=llm_call,
        altering_sql=altering_sql,
        sandbox_execute_success=sandbox_execute_success,
        sandbox_execute_error=sandbox_execute_error,
        gold_sql_on_sandbox_success=gold_sql_on_sandbox_success,
        gold_sql_on_sandbox_error=gold_sql_on_sandbox_error,
        altered_result=altered_result,
        validation_is_valid=validation_is_valid,
        validation_error=validation_error,
        validation_still_present=validation_still_present,
        validation_unintended_missing=validation_unintended_missing,
    )


# ── Writer ─────────────────────────────────────────────────────────────────


class SampleLogger:
    """
    Incrementally writes SampleLog records to JSONL files.

    * Successful samples  → ``sample_logs.jsonl``
    * Failed / skipped    → ``failed_samples.jsonl``

    Call ``consolidate()`` at the end to produce final JSON arrays.
    """

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._success_path = output_dir / "sample_logs.jsonl"
        self._failure_path = output_dir / "failed_samples.jsonl"
        # Open in append mode so partial runs are preserved
        self._success_fh = open(self._success_path, "a", encoding="utf-8")
        self._failure_fh = open(self._failure_path, "a", encoding="utf-8")

    # ── public API ─────────────────────────────────────────────────────────

    def write(self, log: SampleLog) -> None:
        """Serialise *log* and write to the appropriate JSONL file."""
        line = json.dumps(asdict(log), ensure_ascii=False, default=str) + "\n"
        if log.status == "success":
            self._success_fh.write(line)
            self._success_fh.flush()
        else:
            self._failure_fh.write(line)
            self._failure_fh.flush()

    def consolidate(self) -> None:
        """
        Read back the JSONL files and produce consolidated JSON arrays.

        Output:
            sample_logs.json      — array of successful sample logs
            failed_samples.json   — array of failed / skipped sample logs
        """
        self._success_fh.close()
        self._failure_fh.close()

        for jsonl_path in (self._success_path, self._failure_path):
            json_path = jsonl_path.with_suffix(".json")
            records: list[dict[str, Any]] = []
            if jsonl_path.exists():
                with open(jsonl_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2, ensure_ascii=False, default=str)
            logger.info(
                "Consolidated %d records → %s",
                len(records),
                json_path,
            )
