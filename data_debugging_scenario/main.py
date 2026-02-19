#!/usr/bin/env python3
"""
Data Debugging Dataset Generator — CLI Entry Point

Generates a data debugging dataset from the BIRD-bench text-to-SQL seed dataset.
For each sample, the pipeline:
  1. Executes the gold SQL query on the original database
  2. Randomly selects records to alter (delete rows or modify columns)
  3. Uses an LLM to generate an altering SQL statement
  4. Validates the alteration in an isolated sandbox database
  5. Generates a follow-up question, explanation, and fix via LLM

Usage:
    python main.py
    python main.py --samples 50 --model gpt-4o-mini --delete-prob 0.7
    python main.py --samples 0   # process all ~9,428 samples
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config
from db_manager import DatabaseManager
from llm_client import LLMClient
from pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a data debugging dataset from BIRD-bench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── LLM settings ──────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model to use (overrides OPENAI_MODEL env var)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI API base URL (overrides OPENAI_BASE_URL env var)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature",
    )

    # ── Pipeline settings ─────────────────────────────────────────────────
    parser.add_argument(
        "--samples",
        type=int,
        default=config.SAMPLE_COUNT,
        help="Number of samples to process (0 = all)",
    )
    parser.add_argument(
        "--delete-prob",
        type=float,
        default=config.DELETE_PROBABILITY,
        help="Probability of full-row deletion vs column modification (0.0 - 1.0)",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=config.MAX_TARGET_RECORDS,
        help="Maximum number of result records to alter per sample",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=config.MAX_RETRIES,
        help="Max LLM retries on validation failure",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed for reproducibility",
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Output directory for generated dataset",
    )
    parser.add_argument(
        "--train-json",
        type=str,
        default=str(config.BIRD_TRAIN_JSON),
        help="Path to BIRD train.json",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=str(config.BIRD_DB_DIR),
        help="Path to BIRD train_databases directory",
    )

    # ── Logging ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--log-level",
        type=str,
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure logging format and level."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Data Debugging Dataset Generator")
    logger.info("=" * 60)

    # ── Override config paths if provided ──────────────────────────────────
    if args.train_json != str(config.BIRD_TRAIN_JSON):
        config.BIRD_TRAIN_JSON = Path(args.train_json)
    if args.db_dir != str(config.BIRD_DB_DIR):
        config.BIRD_DB_DIR = Path(args.db_dir)

    # ── Initialize components ─────────────────────────────────────────────
    db_manager = DatabaseManager(
        db_dir=Path(args.db_dir),
        sandbox_dir=Path(args.output_dir).parent / "sandbox",
    )

    llm_client = LLMClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )

    pipeline = Pipeline(
        db_manager=db_manager,
        llm_client=llm_client,
        sample_count=args.samples,
        delete_probability=args.delete_prob,
        max_target_records=args.max_targets,
        max_retries=args.max_retries,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )

    # ── Log configuration ─────────────────────────────────────────────────
    logger.info("Model: %s", llm_client.model)
    logger.info("Samples: %s", args.samples if args.samples > 0 else "ALL")
    logger.info("Delete probability: %.2f", args.delete_prob)
    logger.info("Max target records: %d", args.max_targets)
    logger.info("Max retries: %d", args.max_retries)
    logger.info("Random seed: %d", args.seed)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("-" * 60)

    # ── Run pipeline ──────────────────────────────────────────────────────
    results = pipeline.run()

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DONE — Generated %d dataset records", len(results))
    logger.info("Output: %s/dataset.json, %s/dataset.jsonl", args.output_dir, args.output_dir)
    logger.info("Stats:  %s/pipeline_stats.json", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
