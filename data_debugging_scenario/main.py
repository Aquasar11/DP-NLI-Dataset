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
from llm_client import GeminiClient, LLMClient
from pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a data debugging dataset from BIRD-bench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── LLM settings ──────────────────────────────────────────────────────
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides OPENAI_MODEL / GEMINI_MODEL env var)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (overrides OPENAI_API_KEY / GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI API base URL (overrides OPENAI_BASE_URL env var; ignored for Gemini)",
    )
    parser.add_argument(
        "--use-vertexai",
        action="store_true",
        default=False,
        help="Route Gemini requests through Vertex AI (ignored for OpenAI)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature",
    )

    # ── Dataset selection ─────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        type=str,
        default="bird_train",
        choices=["bird_train", "bird_dev", "spider_train", "spider_dev", "spider_test"],
        help=(
            "Dataset to process. Automatically sets --train-json and --db-dir defaults. "
            "Explicit --train-json or --db-dir flags override these defaults."
        ),
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
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel worker threads",
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

    # ── Resolve dataset-specific path defaults ─────────────────────────────
    from pipeline import _dataset_paths, _dataset_json_path  # noqa: E402

    _DATASET_DEFAULTS = {
        "bird_train":   (str(config.BIRD_TRAIN_JSON),   str(config.BIRD_DB_DIR),        config.BIRD_TABLES_JSON),
        "bird_dev":     (str(config.BIRD_DEV_JSON),     str(config.BIRD_DEV_DB_DIR),     config.BIRD_DEV_TABLES_JSON),
        "spider_train": (str(config.SPIDER_TRAIN_JSON), str(config.SPIDER_DB_DIR),       config.SPIDER_TABLES_JSON),
        "spider_dev":   (str(config.SPIDER_DEV_JSON),   str(config.SPIDER_DEV_DB_DIR),   config.SPIDER_DEV_TABLES_JSON),
        "spider_test":  (str(config.SPIDER_TEST_JSON),  str(config.SPIDER_TEST_DB_DIR),  config.SPIDER_TEST_TABLES_JSON),
    }
    dataset = args.dataset
    default_train_json, default_db_dir, tables_json_path = _DATASET_DEFAULTS[dataset]

    # Use explicitly provided CLI values; otherwise fall back to dataset defaults
    train_json = args.train_json if args.train_json != str(config.BIRD_TRAIN_JSON) else default_train_json
    db_dir = args.db_dir if args.db_dir != str(config.BIRD_DB_DIR) else default_db_dir

    # Validate database directory exists
    db_dir_path = Path(db_dir)
    if not db_dir_path.exists():
        logger.error(
            "Database directory not found: %s\n"
            "For 'bird_dev', extract dev_databases.zip to data/bird_dev/dev_databases/ first.",
            db_dir_path,
        )
        sys.exit(1)

    logger.info("Dataset:    %s", dataset)
    logger.info("Train JSON: %s", train_json)
    logger.info("DB dir:     %s", db_dir)

    # ── Initialize components ─────────────────────────────────────────────
    db_manager = DatabaseManager(
        db_dir=db_dir_path,
        sandbox_dir=Path(args.output_dir).parent / "sandbox",
        tables_json_path=tables_json_path,
    )

    if args.provider == "gemini":
        llm_client = GeminiClient(
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            use_vertexai=args.use_vertexai or None,
        )
    else:
        llm_client = LLMClient(
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
        )

    pipeline = Pipeline(
        db_manager=db_manager,
        llm_client=llm_client,
        dataset=dataset,
        sample_count=args.samples,
        delete_probability=args.delete_prob,
        max_target_records=args.max_targets,
        max_retries=args.max_retries,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        max_workers=args.workers,
    )

    # ── Log configuration ─────────────────────────────────────────────────
    logger.info("Model: %s", llm_client.model)
    logger.info("Samples: %s", args.samples if args.samples > 0 else "ALL")
    logger.info("Delete probability: %.2f", args.delete_prob)
    logger.info("Max target records: %d", args.max_targets)
    logger.info("Max retries: %d", args.max_retries)
    logger.info("Random seed: %d", args.seed)
    logger.info("Workers: %d", args.workers)
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
