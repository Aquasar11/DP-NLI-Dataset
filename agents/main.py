"""
CLI entry point for the multi-agent data debugging framework.

Loads the generated dataset, runs all three agents for each record, evaluates
predictions, and writes results plus aggregate statistics to the output directory.

Usage:
    python main.py
    python main.py --provider gemini --use-vertexai --samples 10
    python main.py --dataset /path/to/dataset.json --db-dir /path/to/databases
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import config
from llm_client import GeminiClient, LLMClient
from models import DatasetRecord, RunResult
from runner import run_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent data debugging framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── LLM settings ──────────────────────────────────────────────────────
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
        help="LLM provider",
    )
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--api-key", default=None, help="API key override")
    parser.add_argument(
        "--base-url", default=None, help="OpenAI-compatible base URL (OpenAI only)"
    )
    parser.add_argument(
        "--use-vertexai",
        action="store_true",
        default=False,
        help="Route Gemini through Vertex AI",
    )
    parser.add_argument("--temperature", type=float, default=None, help="LLM temperature")

    # ── Pipeline settings ─────────────────────────────────────────────────
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of records to process (0 = all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--max-explanation-turns",
        type=int,
        default=config.MAX_EXPLANATION_TURNS,
        help="Max Q&A turns for the ExplanationAgent",
    )
    parser.add_argument(
        "--max-answer-turns",
        type=int,
        default=config.MAX_ANSWER_TURNS,
        help="Max Q&A turns for the AnswerAgent",
    )
    parser.add_argument(
        "--question-penalty",
        type=float,
        default=config.QUESTION_PENALTY,
        help="Score penalty per question asked by the AnswerAgent",
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        default=str(config.DATASET_PATH),
        help="Path to dataset.json produced by the generation pipeline",
    )
    parser.add_argument(
        "--db-dir",
        default=str(config.DB_BASE_DIR),
        help="Root directory containing per-database SQLite folders",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.OUTPUT_DIR),
        help="Directory for results and statistics",
    )

    # ── Logging ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _build_llm(args: argparse.Namespace) -> LLMClient | GeminiClient:
    if args.provider == "gemini":
        return GeminiClient(
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            use_vertexai=args.use_vertexai or None,
        )
    return LLMClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )


def _save_results(results: list[RunResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in results],
            f, indent=2, ensure_ascii=False, default=str,
        )

    # Aggregate statistics
    evals = [r.evaluation for r in results]
    n = len(evals)
    if n:
        exact = sum(e.exact_match for e in evals)
        semantic = sum(e.semantic_match for e in evals)
        correct = sum(e.exact_match or e.semantic_match for e in evals)
        avg_score = sum(e.final_score for e in evals) / n
        avg_questions = sum(e.questions_asked for e in evals) / n

        stats = {
            "total_records": n,
            "exact_match": exact,
            "semantic_match": semantic,
            "correct_total": correct,
            "exact_match_rate": f"{exact / n * 100:.1f}%",
            "semantic_match_rate": f"{semantic / n * 100:.1f}%",
            "accuracy": f"{correct / n * 100:.1f}%",
            "avg_final_score": round(avg_score, 4),
            "avg_questions_asked": round(avg_questions, 2),
        }
    else:
        stats = {"total_records": 0}

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats, results_path, stats_path


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Multi-Agent Data Debugging Framework")
    logger.info("=" * 60)

    # ── Load dataset ──────────────────────────────────────────────────────
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        raw_records = json.load(f)

    records = [DatasetRecord(**r) for r in raw_records]
    if args.samples > 0:
        records = records[: args.samples]
    logger.info("Loaded %d record(s) from %s", len(records), dataset_path)

    # ── Build LLM client ──────────────────────────────────────────────────
    llm = _build_llm(args)
    logger.info("Provider: %s  model: %s", args.provider, llm.model)
    logger.info("Workers: %d  explanation_turns: %d  answer_turns: %d",
                args.workers, args.max_explanation_turns, args.max_answer_turns)
    logger.info("-" * 60)

    db_base_dir = Path(args.db_dir)
    sandbox_dir = Path(args.output_dir) / "sandbox"
    output_dir = Path(args.output_dir)

    # ── Run pipeline ──────────────────────────────────────────────────────
    t_start = time.time()
    results: list[RunResult] = []
    errors: list[str] = []

    def _process(record: DatasetRecord) -> RunResult | None:
        try:
            return run_record(
                record=record,
                llm=llm,
                db_base_dir=db_base_dir,
                sandbox_dir=sandbox_dir,
                max_explanation_turns=args.max_explanation_turns,
                max_answer_turns=args.max_answer_turns,
                question_penalty=args.question_penalty,
            )
        except Exception as exc:
            logger.error("record=%d failed: %s", record.id, exc, exc_info=True)
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process, rec): rec for rec in records}
        for future in tqdm(as_completed(futures), total=len(records), desc="Processing"):
            result = future.result()
            if result is not None:
                results.append(result)
            else:
                errors.append(f"record={futures[future].id}")

    results.sort(key=lambda r: r.record_id)

    # ── Save and report ───────────────────────────────────────────────────
    stats, results_path, stats_path = _save_results(results, output_dir)
    elapsed = round(time.time() - t_start, 1)

    logger.info("=" * 60)
    logger.info("DONE in %.1fs — %d/%d records processed", elapsed, len(results), len(records))
    logger.info("Accuracy:          %s", stats.get("accuracy", "N/A"))
    logger.info("Exact match:       %s", stats.get("exact_match_rate", "N/A"))
    logger.info("Semantic match:    %s", stats.get("semantic_match_rate", "N/A"))
    logger.info("Avg final score:   %s", stats.get("avg_final_score", "N/A"))
    logger.info("Avg questions:     %s", stats.get("avg_questions_asked", "N/A"))
    if errors:
        logger.warning("Failed records: %s", ", ".join(errors))
    logger.info("Results: %s", results_path)
    logger.info("Stats:   %s", stats_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
