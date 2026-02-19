"""
Main pipeline orchestrating the Data Debugging Dataset generation.

Sequentially processes BIRD-bench samples: executes gold SQL, decides on
alteration strategy, calls LLM to generate altering SQL, validates in
a sandbox, and generates follow-up Q&A.

All inputs/outputs and decisions are captured via SampleLogger:
  output/sample_logs.jsonl    — one JSON record per successful sample
  output/sample_logs.json     — consolidated JSON array
  output/failed_samples.jsonl — skipped / failed samples
  output/failed_samples.json  — consolidated JSON array
"""

from __future__ import annotations

import datetime
import json
import logging
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

import config
from db_manager import DatabaseManager
from llm_client import LLMClient
from models import (
    AlterationDecision,
    AlterationType,
    BirdSample,
    DatasetRecord,
    ValidationResult,
)
from prompts import (
    build_alteration_prompt,
    build_followup_prompt,
    build_retry_prompt,
)
from sample_logger import (
    AlterationDecisionLog,
    AttemptLog,
    SampleLog,
    SampleLogger,
    make_attempt_log,
    make_llm_call_log,
)
from validator import (
    is_aggregate_query,
    validate_alteration,
)

logger = logging.getLogger(__name__)


class PipelineStats:
    """Aggregate counts across the full pipeline run."""

    def __init__(self) -> None:
        self.total = 0
        self.processed = 0
        self.success = 0
        self.skipped_empty = 0
        self.skipped_error = 0
        self.skipped_aggregate = 0
        self.failed_validation = 0
        self.failed_llm = 0

    def summary(self) -> dict[str, Any]:
        return {
            "total_samples": self.total,
            "processed": self.processed,
            "success": self.success,
            "skipped_empty_result": self.skipped_empty,
            "skipped_query_error": self.skipped_error,
            "skipped_aggregate": self.skipped_aggregate,
            "failed_validation_after_retries": self.failed_validation,
            "failed_llm_error": self.failed_llm,
            "success_rate": f"{self.success / max(self.processed, 1) * 100:.1f}%",
        }


class Pipeline:
    """Orchestrates the data debugging dataset generation."""

    def __init__(
        self,
        *,
        db_manager: DatabaseManager | None = None,
        llm_client: LLMClient | None = None,
        sample_count: int | None = None,
        delete_probability: float | None = None,
        max_target_records: int | None = None,
        max_retries: int | None = None,
        seed: int | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.db = db_manager or DatabaseManager()
        self.llm = llm_client
        self.sample_count = sample_count if sample_count is not None else config.SAMPLE_COUNT
        self.delete_prob = (
            delete_probability if delete_probability is not None else config.DELETE_PROBABILITY
        )
        self.max_targets = (
            max_target_records if max_target_records is not None else config.MAX_TARGET_RECORDS
        )
        self.max_retries = max_retries if max_retries is not None else config.MAX_RETRIES
        self.seed = seed if seed is not None else config.RANDOM_SEED
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.rng = random.Random(self.seed)
        self.stats = PipelineStats()
        self.sample_logger: SampleLogger | None = None

    # ── Data Loading ───────────────────────────────────────────────────────

    def load_samples(self, path: Path | None = None) -> list[BirdSample]:
        """Load BIRD training samples from train.json."""
        path = path or config.BIRD_TRAIN_JSON
        logger.info("Loading samples from %s", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = [BirdSample(**entry) for entry in data]
        logger.info("Loaded %d samples total", len(samples))
        return samples

    # ── Alteration Decision ────────────────────────────────────────────────

    def make_alteration_decision(
        self,
        gold_result: list[dict[str, Any]],
        sample_idx: int,
    ) -> tuple[AlterationDecision, AlterationDecisionLog]:
        """
        Randomly decide the alteration strategy and log every factor.

        Returns both the AlterationDecision and a detailed AlterationDecisionLog.
        """
        # Draw random value and decide delete vs. modify
        rand_draw = self.rng.random()
        if rand_draw < self.delete_prob:
            alt_type = AlterationType.DELETE
        else:
            alt_type = AlterationType.MODIFY

        logger.info(
            "[%d] Decision — alteration type: rand_draw=%.4f, delete_prob=%.2f → %s",
            sample_idx, rand_draw, self.delete_prob, alt_type.value.upper(),
        )

        # Choose target record count and indices
        num_result_rows = len(gold_result)
        num_targets = min(self.max_targets, num_result_rows)
        target_indices = sorted(self.rng.sample(range(num_result_rows), num_targets))
        targeted_records = [gold_result[i] for i in target_indices]

        logger.info(
            "[%d] Decision — targets: result_rows=%d, max_targets_config=%d, "
            "num_chosen=%d, indices=%s",
            sample_idx, num_result_rows, self.max_targets, num_targets, target_indices,
        )
        for i, rec in enumerate(targeted_records):
            logger.info("[%d]   target[%d]: %s", sample_idx, i, rec)

        # For MODIFY: choose which result columns the LLM should alter
        target_columns: list[str] | None = None
        if alt_type == AlterationType.MODIFY and gold_result:
            all_cols = list(gold_result[0].keys())
            if len(all_cols) > 1:
                num_cols = self.rng.randint(1, len(all_cols))
                target_columns = sorted(self.rng.sample(all_cols, num_cols))
                logger.info(
                    "[%d] Decision — MODIFY columns: available=%s, chosen=%s",
                    sample_idx, all_cols, target_columns,
                )
            else:
                target_columns = all_cols
                logger.info(
                    "[%d] Decision — MODIFY columns: single column available → %s",
                    sample_idx, target_columns,
                )

        decision = AlterationDecision(
            alteration_type=alt_type,
            target_record_indices=target_indices,
            target_columns=target_columns,
        )
        decision_log = AlterationDecisionLog(
            alteration_type=alt_type.value,
            num_result_rows=num_result_rows,
            max_targets_config=self.max_targets,
            num_targets_chosen=num_targets,
            target_record_indices=target_indices,
            targeted_records=targeted_records,
            target_columns=target_columns,
            delete_probability_config=self.delete_prob,
            random_draw=round(rand_draw, 6),
        )
        return decision, decision_log

    # ── Single Sample Processing ───────────────────────────────────────────

    def process_sample(
        self,
        sample: BirdSample,
        sample_idx: int,
        record_id: int,
    ) -> DatasetRecord | None:
        """
        Process a single BIRD sample through the full pipeline.

        Every decision, LLM call, and validation result is captured in a
        SampleLog and written incrementally via SampleLogger.
        Returns a DatasetRecord on success, None on skip/failure.
        """
        self.stats.processed += 1
        t_start = time.perf_counter()
        ts_start = datetime.datetime.utcnow().isoformat() + "Z"

        logger.info(
            "━━━ [sample %d / total %d]  db=%s",
            sample_idx, self.stats.total, sample.db_id,
        )
        logger.info("[%d] question : %s", sample_idx, sample.question)
        logger.info("[%d] gold_sql : %s", sample_idx, sample.SQL)

        # Mutable dict progressively filled and handed to SampleLog
        base: dict[str, Any] = dict(
            sample_idx=sample_idx,
            record_id=None,
            db_id=sample.db_id,
            question=sample.question,
            evidence=sample.evidence,
            gold_sql=sample.SQL,
            gold_result=[],
            gold_result_row_count=0,
            alteration_decision=None,
            step1_attempts=[],
            step1_final_altering_sql=None,
            step1_final_explanation=None,
            step1_total_attempts=0,
            step1_passed=False,
            step2_llm_call=None,
            step2_follow_up_question=None,
            step2_gold_explanation=None,
            step2_gold_fix=None,
            step2_passed=False,
            total_duration_seconds=0.0,
            timestamp_start=ts_start,
        )

        def _emit(status: str, skip_reason: str | None = None) -> None:
            base["total_duration_seconds"] = round(time.perf_counter() - t_start, 3)
            if self.sample_logger:
                self.sample_logger.write(SampleLog(status=status, skip_reason=skip_reason, **base))

        # ── 1. Resolve DB path ─────────────────────────────────────────────
        db_path = self.db.get_db_path(sample.db_id)
        if not db_path.exists():
            msg = f"Database not found: {db_path}"
            logger.warning("[%d] SKIP — %s", sample_idx, msg)
            self.stats.skipped_error += 1
            _emit("skipped_error", msg)
            return None

        # ── 2. Execute gold SQL ────────────────────────────────────────────
        logger.info("[%d] Executing gold SQL on original DB…", sample_idx)
        try:
            gold_result_obj = self.db.execute_query(db_path, sample.SQL)
            gold_result = gold_result_obj.rows
        except sqlite3.Error as e:
            msg = f"Gold SQL execution error: {e}"
            logger.warning("[%d] SKIP — %s", sample_idx, msg)
            self.stats.skipped_error += 1
            _emit("skipped_error", msg)
            return None

        base["gold_result"] = gold_result
        base["gold_result_row_count"] = len(gold_result)
        logger.info(
            "[%d] Gold SQL → %d row(s) | preview: %s",
            sample_idx, len(gold_result), gold_result[:3],
        )

        # ── 3. Skip empty results ─────────────────────────────────────────
        if not gold_result:
            logger.info("[%d] SKIP — empty result set", sample_idx)
            self.stats.skipped_empty += 1
            _emit("skipped_empty", "Gold SQL returned 0 rows")
            return None

        # ── 4. Skip scalar aggregates ─────────────────────────────────────
        if is_aggregate_query(sample.SQL) and len(gold_result) == 1:
            logger.info("[%d] SKIP — scalar aggregate query (no targetable rows)", sample_idx)
            self.stats.skipped_aggregate += 1
            _emit("skipped_aggregate", "Scalar aggregate — no individual row to target")
            return None

        # ── 5. Alteration decision ─────────────────────────────────────────
        decision, decision_log = self.make_alteration_decision(gold_result, sample_idx)
        targeted_records = [gold_result[i] for i in decision.target_record_indices]
        base["alteration_decision"] = decision_log

        # ── 6. Retrieve DDL ───────────────────────────────────────────────
        try:
            db_ddl = self.db.get_ddl(sample.db_id)
            logger.debug("[%d] DDL retrieved (%d chars)", sample_idx, len(db_ddl))
        except Exception as e:
            logger.warning("[%d] Could not retrieve DDL: %s — using placeholder", sample_idx, e)
            db_ddl = "(DDL unavailable)"

        # ── 7. Step 1 loop: generate + validate altering SQL ───────────────
        step1_attempts: list[AttemptLog] = []
        final_alteration_result = None
        final_altered_result: list[dict[str, Any]] = []
        final_validation: ValidationResult | None = None
        prev_altering_sql = ""
        prev_explanation = ""
        prev_altered_result: list[dict[str, Any]] = []

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                "[%d] ── Step 1 attempt %d/%d ──────────────────",
                sample_idx, attempt, self.max_retries,
            )

            # Build prompt
            if attempt == 1:
                prompt = build_alteration_prompt(
                    gold_sql=sample.SQL,
                    db_ddl=db_ddl,
                    gold_result=gold_result,
                    targeted_records=targeted_records,
                    alteration_type=decision.alteration_type,
                    target_columns=decision.target_columns,
                )
            else:
                prompt = build_retry_prompt(
                    previous_altering_sql=prev_altering_sql,
                    previous_explanation=prev_explanation,
                    error_message=(
                        final_validation.error_message if final_validation else "Unknown"
                    ),
                    gold_sql=sample.SQL,
                    db_ddl=db_ddl,
                    gold_result=gold_result,
                    altered_result=prev_altered_result,
                    targeted_records=targeted_records,
                    alteration_type=decision.alteration_type,
                    target_columns=decision.target_columns,
                )
                logger.info(
                    "[%d] Retry — previous validation error: %s",
                    sample_idx,
                    final_validation.error_message if final_validation else "N/A",
                )

            logger.debug("[%d] Sending %d-char prompt to LLM", sample_idx, len(prompt))

            # LLM call
            llm_result = self.llm.generate_alteration(prompt)

            if not llm_result.success:
                logger.error(
                    "[%d] LLM call FAILED (attempt %d): %s",
                    sample_idx, attempt, llm_result.error,
                )
                llm_log = make_llm_call_log(
                    step="alteration_step1", attempt=attempt,
                    system_prompt=llm_result.system_prompt,
                    user_prompt=llm_result.user_prompt,
                    raw_response=llm_result.raw_response,
                    parsed_response=llm_result.parsed_dict,
                    success=False, error=llm_result.error,
                    duration_seconds=llm_result.duration_seconds,
                )
                step1_attempts.append(make_attempt_log(
                    attempt=attempt, llm_call=llm_log,
                    altering_sql="",
                    sandbox_execute_success=False,
                    sandbox_execute_error="LLM call failed",
                    gold_sql_on_sandbox_success=False,
                    gold_sql_on_sandbox_error=None,
                    altered_result=[],
                    validation_is_valid=False,
                    validation_error="LLM call failed: " + (llm_result.error or ""),
                    validation_still_present=[],
                    validation_unintended_missing=[],
                ))
                if attempt == self.max_retries:
                    base["step1_attempts"] = step1_attempts
                    base["step1_total_attempts"] = attempt
                    self.stats.failed_llm += 1
                    _emit("failed_llm", f"LLM error after {attempt} attempts: {llm_result.error}")
                    return None
                continue

            alteration_response = llm_result.parsed
            prev_altering_sql = alteration_response.altering_sql
            prev_explanation = alteration_response.explanation

            logger.info(
                "[%d] LLM response (%.2fs)\n"
                "  altering_sql : %s\n"
                "  explanation  : %s",
                sample_idx, llm_result.duration_seconds,
                alteration_response.altering_sql,
                alteration_response.explanation,
            )

            # Sandbox run
            logger.info("[%d] Creating sandbox…", sample_idx)
            sandbox_path = self.db.create_sandbox(sample.db_id)
            sandbox_ok = False
            sandbox_err: str | None = None
            gold_on_sandbox_ok = False
            gold_on_sandbox_err: str | None = None
            altered_result: list[dict[str, Any]] = []
            this_validation: ValidationResult | None = None

            try:
                try:
                    self.db.execute_alter(sandbox_path, alteration_response.altering_sql)
                    sandbox_ok = True
                    logger.info("[%d] Altering SQL executed on sandbox ✓", sample_idx)
                except sqlite3.Error as e:
                    sandbox_err = str(e)
                    logger.warning("[%d] Altering SQL FAILED on sandbox: %s", sample_idx, e)

                if sandbox_ok:
                    try:
                        altered_obj = self.db.execute_query(sandbox_path, sample.SQL)
                        altered_result = altered_obj.rows
                        gold_on_sandbox_ok = True
                        logger.info(
                            "[%d] Gold SQL on sandbox → %d row(s) | preview: %s",
                            sample_idx, len(altered_result), altered_result[:3],
                        )
                    except sqlite3.Error as e:
                        gold_on_sandbox_err = str(e)
                        logger.warning(
                            "[%d] Gold SQL on sandbox FAILED: %s", sample_idx, e,
                        )

                if sandbox_ok and gold_on_sandbox_ok:
                    this_validation = validate_alteration(
                        gold_result=gold_result,
                        altered_result=altered_result,
                        targeted_records=targeted_records,
                        alteration_type=decision.alteration_type,
                    )
                    if this_validation.is_valid:
                        logger.info(
                            "[%d] ✓ Validation PASSED — %d targeted record(s) removed",
                            sample_idx, len(this_validation.missing_targeted),
                        )
                    else:
                        logger.info(
                            "[%d] ✗ Validation FAILED: %s",
                            sample_idx, this_validation.error_message,
                        )
                        if this_validation.still_present_targeted:
                            logger.info(
                                "[%d]   Still present (should be gone): %s",
                                sample_idx, this_validation.still_present_targeted,
                            )
                        if this_validation.unintended_missing:
                            logger.info(
                                "[%d]   Unintentionally removed: %s",
                                sample_idx, this_validation.unintended_missing,
                            )
                else:
                    this_validation = ValidationResult(
                        is_valid=False,
                        error_message=sandbox_err or gold_on_sandbox_err or "Unknown error",
                    )
            finally:
                self.db.destroy_sandbox(sandbox_path)
                logger.debug("[%d] Sandbox destroyed", sample_idx)

            llm_log = make_llm_call_log(
                step="alteration_step1", attempt=attempt,
                system_prompt=llm_result.system_prompt,
                user_prompt=llm_result.user_prompt,
                raw_response=llm_result.raw_response,
                parsed_response=llm_result.parsed_dict,
                success=True,
                duration_seconds=llm_result.duration_seconds,
            )
            step1_attempts.append(make_attempt_log(
                attempt=attempt, llm_call=llm_log,
                altering_sql=alteration_response.altering_sql,
                sandbox_execute_success=sandbox_ok,
                sandbox_execute_error=sandbox_err,
                gold_sql_on_sandbox_success=gold_on_sandbox_ok,
                gold_sql_on_sandbox_error=gold_on_sandbox_err,
                altered_result=altered_result,
                validation_is_valid=this_validation.is_valid,
                validation_error=this_validation.error_message,
                validation_still_present=this_validation.still_present_targeted,
                validation_unintended_missing=this_validation.unintended_missing,
            ))
            final_validation = this_validation
            prev_altered_result = altered_result

            if this_validation.is_valid:
                final_alteration_result = alteration_response
                final_altered_result = altered_result
                break

        base["step1_attempts"] = step1_attempts
        base["step1_total_attempts"] = len(step1_attempts)

        # Step 1 ultimately failed?
        if final_alteration_result is None or (
            final_validation and not final_validation.is_valid
        ):
            reason = (
                final_validation.error_message
                if final_validation
                else "All LLM attempts failed"
            )
            logger.info(
                "[%d] Step 1 FAILED after %d attempt(s): %s",
                sample_idx, len(step1_attempts), reason,
            )
            self.stats.failed_validation += 1
            base["step1_passed"] = False
            _emit(
                "failed_validation",
                f"Validation failed after {self.max_retries} retries: {reason}",
            )
            return None

        base["step1_passed"] = True
        base["step1_final_altering_sql"] = final_alteration_result.altering_sql
        base["step1_final_explanation"] = final_alteration_result.explanation
        logger.info("[%d] Step 1 complete ✓  (%d attempt(s))", sample_idx, len(step1_attempts))

        # ── 8. Step 2: follow-up Q&A ───────────────────────────────────────
        logger.info("[%d] ── Step 2: follow-up Q&A ──────────────────────", sample_idx)
        followup_prompt = build_followup_prompt(
            question=sample.question,
            evidence=sample.evidence,
            gold_sql=sample.SQL,
            gold_result=gold_result,
            altered_result=final_altered_result,
            alteration_type=decision.alteration_type,
            targeted_records=targeted_records,
            altering_sql=final_alteration_result.altering_sql,
            alteration_explanation=final_alteration_result.explanation,
        )
        logger.debug("[%d] Follow-up prompt (%d chars)", sample_idx, len(followup_prompt))

        followup_result = self.llm.generate_followup(followup_prompt)

        step2_llm_log = make_llm_call_log(
            step="followup_step2", attempt=1,
            system_prompt=followup_result.system_prompt,
            user_prompt=followup_result.user_prompt,
            raw_response=followup_result.raw_response,
            parsed_response=followup_result.parsed_dict,
            success=followup_result.success,
            error=followup_result.error,
            duration_seconds=followup_result.duration_seconds,
        )
        base["step2_llm_call"] = step2_llm_log

        if not followup_result.success:
            logger.error("[%d] Step 2 LLM FAILED: %s", sample_idx, followup_result.error)
            self.stats.failed_llm += 1
            base["step2_passed"] = False
            _emit("failed_llm", f"Follow-up LLM error: {followup_result.error}")
            return None

        followup_response = followup_result.parsed
        logger.info(
            "[%d] Step 2 response (%.2fs)\n"
            "  follow_up_question : %s\n"
            "  gold_explanation   : %s\n"
            "  gold_fix           : %s",
            sample_idx, followup_result.duration_seconds,
            followup_response.follow_up_question,
            followup_response.gold_explanation,
            followup_response.gold_fix,
        )

        base["step2_passed"] = True
        base["step2_follow_up_question"] = followup_response.follow_up_question
        base["step2_gold_explanation"] = followup_response.gold_explanation
        base["step2_gold_fix"] = followup_response.gold_fix

        # ── 9. Assemble and return DatasetRecord ───────────────────────────
        record = DatasetRecord(
            id=record_id,
            db_id=sample.db_id,
            question=sample.question,
            evidence=sample.evidence,
            gold_sql=sample.SQL,
            gold_result=gold_result,
            alteration_type=decision.alteration_type,
            targeted_records=targeted_records,
            altering_sql=final_alteration_result.altering_sql,
            altered_result=final_altered_result,
            alteration_explanation=final_alteration_result.explanation,
            follow_up_question=followup_response.follow_up_question,
            gold_explanation=followup_response.gold_explanation,
            gold_fix=followup_response.gold_fix,
        )

        base["record_id"] = record_id
        self.stats.success += 1
        _emit("success")
        logger.info("[%d] ✓ DatasetRecord id=%d complete\n", sample_idx, record_id)
        return record

    # ── Full Pipeline Run ──────────────────────────────────────────────────

    def run(self) -> list[DatasetRecord]:
        """
        Run the complete pipeline over the configured number of samples.

        Returns:
            List of successfully generated DatasetRecords.
        """
        start_time = time.time()

        samples = self.load_samples()
        self.rng.shuffle(samples)
        if self.sample_count > 0:
            samples = samples[: self.sample_count]
        self.stats.total = len(samples)

        logger.info(
            "Pipeline config: samples=%d, delete_prob=%.2f, max_targets=%d, "
            "max_retries=%d, seed=%d, model=%s",
            len(samples), self.delete_prob, self.max_targets,
            self.max_retries, self.seed,
            self.llm.model if self.llm else "N/A",
        )

        self.db.cleanup_all_sandboxes()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_logger = SampleLogger(self.output_dir)

        results: list[DatasetRecord] = []
        record_id = 1

        try:
            for idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
                record = self.process_sample(sample, idx, record_id)
                if record is not None:
                    results.append(record)
                    record_id += 1
        finally:
            self.db.cleanup_all_sandboxes()
            self.sample_logger.consolidate()

        elapsed = time.time() - start_time
        self._save_results(results)

        summary = self.stats.summary()
        summary["elapsed_seconds"] = round(elapsed, 1)
        logger.info(
            "Pipeline done in %.1fs — %d success / %d processed "
            "(skipped: %d empty, %d error, %d aggregate | "
            "failed: %d validation, %d LLM)",
            elapsed, self.stats.success, self.stats.processed,
            self.stats.skipped_empty, self.stats.skipped_error,
            self.stats.skipped_aggregate,
            self.stats.failed_validation, self.stats.failed_llm,
        )
        self._save_stats(summary)
        return results

    # ── Output ─────────────────────────────────────────────────────────────

    def _save_results(self, results: list[DatasetRecord]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.output_dir / "dataset.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                [r.model_dump() for r in results],
                f, indent=2, ensure_ascii=False, default=str,
            )
        logger.info("Saved %d records → %s", len(results), json_path)

        jsonl_path = self.output_dir / "dataset.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.model_dump(), ensure_ascii=False, default=str) + "\n")
        logger.info("Saved %d records → %s", len(results), jsonl_path)

    def _save_stats(self, summary: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stats_path = self.output_dir / "pipeline_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved stats → %s", stats_path)
