"""
Main pipeline orchestrating the Data Debugging Dataset generation.

Sequentially processes BIRD-bench samples: executes gold SQL, decides on
alteration strategy, calls LLM to generate altering SQL, validates in
a sandbox, and generates follow-up Q&A.
"""

from __future__ import annotations

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
)
from prompts import (
    build_alteration_prompt,
    build_followup_prompt,
    build_retry_prompt,
)
from validator import (
    is_aggregate_query,
    validate_alteration,
    validate_alteration_aggregate,
)

logger = logging.getLogger(__name__)


class PipelineStats:
    """Track pipeline statistics."""

    def __init__(self):
        self.total = 0
        self.processed = 0
        self.success = 0
        self.skipped_empty = 0
        self.skipped_error = 0
        self.skipped_aggregate = 0
        self.failed_validation = 0
        self.failed_llm = 0
        self.errors: list[dict[str, Any]] = []

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
            "error_log": self.errors[:50],  # Keep first 50 errors
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
    ):
        self.db = db_manager or DatabaseManager()
        self.llm = llm_client  # Initialized lazily or passed in
        self.sample_count = sample_count if sample_count is not None else config.SAMPLE_COUNT
        self.delete_prob = delete_probability if delete_probability is not None else config.DELETE_PROBABILITY
        self.max_targets = max_target_records if max_target_records is not None else config.MAX_TARGET_RECORDS
        self.max_retries = max_retries if max_retries is not None else config.MAX_RETRIES
        self.seed = seed if seed is not None else config.RANDOM_SEED
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.rng = random.Random(self.seed)
        self.stats = PipelineStats()

    # ── Data Loading ───────────────────────────────────────────────────────

    def load_samples(self, path: Path | None = None) -> list[BirdSample]:
        """Load BIRD training samples from train.json."""
        path = path or config.BIRD_TRAIN_JSON
        logger.info("Loading samples from %s", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = [BirdSample(**entry) for entry in data]
        logger.info("Loaded %d samples", len(samples))
        return samples

    # ── Alteration Decision ────────────────────────────────────────────────

    def make_alteration_decision(
        self,
        gold_result: list[dict[str, Any]],
    ) -> AlterationDecision:
        """
        Randomly decide the alteration strategy:
        - DELETE or MODIFY (based on delete_probability)
        - Which records to target (random subset of result rows)
        """
        # Choose alteration type
        if self.rng.random() < self.delete_prob:
            alt_type = AlterationType.DELETE
        else:
            alt_type = AlterationType.MODIFY

        # Choose how many records to target (1 to max_targets, capped by result size)
        num_targets = min(self.max_targets, len(gold_result))
        target_indices = sorted(self.rng.sample(range(len(gold_result)), num_targets))

        # For MODIFY, optionally select which columns to change
        target_columns = None
        if alt_type == AlterationType.MODIFY and gold_result:
            all_cols = list(gold_result[0].keys())
            if len(all_cols) > 1:
                # Randomly select 1 to all columns
                num_cols = self.rng.randint(1, len(all_cols))
                target_columns = sorted(self.rng.sample(all_cols, num_cols))

        return AlterationDecision(
            alteration_type=alt_type,
            target_record_indices=target_indices,
            target_columns=target_columns,
        )

    # ── Single Sample Processing ───────────────────────────────────────────

    def process_sample(
        self,
        sample: BirdSample,
        sample_idx: int,
        record_id: int,
    ) -> DatasetRecord | None:
        """
        Process a single BIRD sample through the full pipeline.

        Returns a DatasetRecord on success, or None if the sample was
        skipped or failed after all retries.
        """
        self.stats.processed += 1

        # ── 1. Execute gold SQL on original database ───────────────────────
        db_path = self.db.get_db_path(sample.db_id)
        if not db_path.exists():
            logger.warning("[%d] Database not found: %s", sample_idx, db_path)
            self.stats.skipped_error += 1
            self.stats.errors.append({
                "sample_idx": sample_idx, "db_id": sample.db_id,
                "error": f"Database not found: {db_path}",
            })
            return None

        try:
            gold_result_obj = self.db.execute_query(db_path, sample.SQL)
            gold_result = gold_result_obj.rows
        except sqlite3.Error as e:
            logger.warning("[%d] Gold SQL error on %s: %s", sample_idx, sample.db_id, e)
            self.stats.skipped_error += 1
            self.stats.errors.append({
                "sample_idx": sample_idx, "db_id": sample.db_id,
                "sql": sample.SQL, "error": str(e),
            })
            return None

        # ── 2. Skip empty results ─────────────────────────────────────────
        if not gold_result:
            logger.debug("[%d] Skipping: empty result for %s", sample_idx, sample.db_id)
            self.stats.skipped_empty += 1
            return None

        # ── 3. Skip pure scalar aggregates (no meaningful row to remove) ──
        if is_aggregate_query(sample.SQL) and len(gold_result) == 1:
            logger.debug("[%d] Skipping: scalar aggregate query", sample_idx)
            self.stats.skipped_aggregate += 1
            return None

        # ── 4. Make alteration decision ───────────────────────────────────
        decision = self.make_alteration_decision(gold_result)
        targeted_records = [gold_result[i] for i in decision.target_record_indices]

        # ── 5. Get database schema for the prompt ─────────────────────────
        try:
            db_ddl = self.db.get_ddl(sample.db_id)
        except Exception as e:
            logger.warning("[%d] Could not get DDL for %s: %s", sample_idx, sample.db_id, e)
            db_ddl = "(DDL unavailable)"

        # ── 6. LLM Step 1: Generate altering SQL (with retries) ───────────
        alteration_response = None
        altered_result = None
        validation = None

        for attempt in range(1, self.max_retries + 1):
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
                    previous_altering_sql=alteration_response.altering_sql,
                    previous_explanation=alteration_response.explanation,
                    error_message=validation.error_message or "Unknown error",
                    gold_sql=sample.SQL,
                    db_ddl=db_ddl,
                    gold_result=gold_result,
                    altered_result=altered_result or [],
                    targeted_records=targeted_records,
                    alteration_type=decision.alteration_type,
                    target_columns=decision.target_columns,
                )

            # Call LLM
            try:
                alteration_response = self.llm.generate_alteration(prompt)
            except Exception as e:
                logger.error(
                    "[%d] LLM error (attempt %d/%d): %s",
                    sample_idx, attempt, self.max_retries, e,
                )
                if attempt == self.max_retries:
                    self.stats.failed_llm += 1
                    self.stats.errors.append({
                        "sample_idx": sample_idx, "db_id": sample.db_id,
                        "error": f"LLM error: {e}",
                    })
                    return None
                continue

            # ── 7. Create sandbox and apply alteration ────────────────────
            sandbox_path = self.db.create_sandbox(sample.db_id)
            try:
                # Execute altering SQL
                try:
                    self.db.execute_alter(sandbox_path, alteration_response.altering_sql)
                except sqlite3.Error as e:
                    logger.warning(
                        "[%d] Alter SQL error (attempt %d/%d): %s",
                        sample_idx, attempt, self.max_retries, e,
                    )
                    validation = type("V", (), {
                        "is_valid": False,
                        "error_message": f"SQL execution error: {e}",
                    })()
                    self.db.destroy_sandbox(sandbox_path)
                    if attempt == self.max_retries:
                        self.stats.failed_validation += 1
                        self.stats.errors.append({
                            "sample_idx": sample_idx, "db_id": sample.db_id,
                            "error": f"Alter SQL error after {self.max_retries} retries: {e}",
                        })
                        return None
                    continue

                # ── 8. Re-execute gold SQL on sandbox ─────────────────────
                try:
                    altered_result_obj = self.db.execute_query(sandbox_path, sample.SQL)
                    altered_result = altered_result_obj.rows
                except sqlite3.Error as e:
                    logger.warning(
                        "[%d] Gold SQL failed on sandbox (attempt %d/%d): %s",
                        sample_idx, attempt, self.max_retries, e,
                    )
                    validation = type("V", (), {
                        "is_valid": False,
                        "error_message": f"Gold SQL failed on altered DB: {e}",
                    })()
                    self.db.destroy_sandbox(sandbox_path)
                    if attempt == self.max_retries:
                        self.stats.failed_validation += 1
                        return None
                    continue

                # ── 9. Validate the alteration ────────────────────────────
                validation = validate_alteration(
                    gold_result=gold_result,
                    altered_result=altered_result,
                    targeted_records=targeted_records,
                    alteration_type=decision.alteration_type,
                )

                if validation.is_valid:
                    logger.info(
                        "[%d] ✓ Validation passed on attempt %d — %s %s %d record(s)",
                        sample_idx, attempt, sample.db_id,
                        decision.alteration_type.value, len(targeted_records),
                    )
                    break
                else:
                    logger.info(
                        "[%d] ✗ Validation failed (attempt %d/%d): %s",
                        sample_idx, attempt, self.max_retries,
                        validation.error_message,
                    )
            finally:
                self.db.destroy_sandbox(sandbox_path)

        # Check if validation ultimately failed
        if not validation or not validation.is_valid:
            self.stats.failed_validation += 1
            self.stats.errors.append({
                "sample_idx": sample_idx, "db_id": sample.db_id,
                "error": f"Validation failed after {self.max_retries} retries: "
                         f"{validation.error_message if validation else 'No validation'}",
            })
            return None

        # ── 10. LLM Step 2: Generate follow-up Q&A ───────────────────────
        try:
            followup_prompt = build_followup_prompt(
                question=sample.question,
                evidence=sample.evidence,
                gold_sql=sample.SQL,
                gold_result=gold_result,
                altered_result=altered_result,
                alteration_type=decision.alteration_type,
                targeted_records=targeted_records,
                altering_sql=alteration_response.altering_sql,
                alteration_explanation=alteration_response.explanation,
            )
            followup_response = self.llm.generate_followup(followup_prompt)
        except Exception as e:
            logger.error("[%d] Follow-up LLM error: %s", sample_idx, e)
            self.stats.failed_llm += 1
            self.stats.errors.append({
                "sample_idx": sample_idx, "db_id": sample.db_id,
                "error": f"Follow-up LLM error: {e}",
            })
            return None

        # ── 11. Assemble the dataset record ───────────────────────────────
        record = DatasetRecord(
            id=record_id,
            db_id=sample.db_id,
            question=sample.question,
            evidence=sample.evidence,
            gold_sql=sample.SQL,
            gold_result=gold_result,
            alteration_type=decision.alteration_type,
            targeted_records=targeted_records,
            altering_sql=alteration_response.altering_sql,
            altered_result=altered_result,
            alteration_explanation=alteration_response.explanation,
            follow_up_question=followup_response.follow_up_question,
            gold_explanation=followup_response.gold_explanation,
            gold_fix=followup_response.gold_fix,
        )

        self.stats.success += 1
        return record

    # ── Full Pipeline Run ──────────────────────────────────────────────────

    def run(self) -> list[DatasetRecord]:
        """
        Run the complete pipeline over the configured number of samples.

        Returns:
            List of successfully generated DatasetRecords.
        """
        start_time = time.time()

        # Load samples
        samples = self.load_samples()

        # Shuffle and take subset
        self.rng.shuffle(samples)
        if self.sample_count > 0:
            samples = samples[: self.sample_count]
        self.stats.total = len(samples)

        logger.info(
            "Starting pipeline: %d samples, delete_prob=%.2f, max_targets=%d, "
            "max_retries=%d, seed=%d",
            len(samples), self.delete_prob, self.max_targets,
            self.max_retries, self.seed,
        )

        # Clean up any leftover sandboxes
        self.db.cleanup_all_sandboxes()

        # Process samples
        results: list[DatasetRecord] = []
        record_id = 1

        for idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
            record = self.process_sample(sample, idx, record_id)
            if record is not None:
                results.append(record)
                record_id += 1

        elapsed = time.time() - start_time

        # Final cleanup
        self.db.cleanup_all_sandboxes()

        # Save results
        self._save_results(results)

        # Log summary
        summary = self.stats.summary()
        summary["elapsed_seconds"] = round(elapsed, 1)
        logger.info("Pipeline completed in %.1fs", elapsed)
        logger.info(
            "Results: %d success / %d processed (skipped: %d empty, %d error, "
            "%d aggregate | failed: %d validation, %d LLM)",
            self.stats.success, self.stats.processed,
            self.stats.skipped_empty, self.stats.skipped_error,
            self.stats.skipped_aggregate,
            self.stats.failed_validation, self.stats.failed_llm,
        )

        # Save stats
        self._save_stats(summary)

        return results

    # ── Output ─────────────────────────────────────────────────────────────

    def _save_results(self, results: list[DatasetRecord]) -> None:
        """Save generated dataset records to JSON and JSONL files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = self.output_dir / "dataset.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                [r.model_dump() for r in results],
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        logger.info("Saved %d records to %s", len(results), json_path)

        # JSONL
        jsonl_path = self.output_dir / "dataset.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.model_dump(), ensure_ascii=False, default=str) + "\n")
        logger.info("Saved %d records to %s", len(results), jsonl_path)

    def _save_stats(self, summary: dict[str, Any]) -> None:
        """Save pipeline statistics to a JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stats_path = self.output_dir / "pipeline_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved stats to %s", stats_path)
