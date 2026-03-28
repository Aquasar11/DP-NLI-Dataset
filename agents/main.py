"""
CLI entry point for the multi-agent data debugging framework.

Loads the generated dataset, runs all three agents for each record, evaluates
results, and writes outputs plus aggregate statistics to the output directory.

Usage:
    python main.py
    python main.py --provider gemini --use-vertexai --samples 10
    python main.py --dataset /path/to/dataset.json --db-dir /path/to/databases

Per-agent overrides:
    python main.py --user-agent-model gpt-4o --explanation-agent-model gpt-4o-mini \\
                   --fix-agent-model gpt-4o --judge-model gpt-4o-mini
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
from config import AgentLLMConfig
from llm_client import GeminiClient, LLMClient
from models import DatasetRecord, RunResult
from runner import run_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent data debugging framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Global LLM settings (fallback for all agents) ─────────────────────
    global_group = parser.add_argument_group(
        "Global LLM settings",
        "Default settings for all agents. Per-agent flags override these.",
    )
    global_group.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
        help="LLM provider (default for all agents)",
    )
    global_group.add_argument("--model", default=None, help="Model name (default for all agents)")
    global_group.add_argument("--api-key", default=None, help="API key (default for all agents)")
    global_group.add_argument(
        "--base-url", default=None, help="OpenAI-compatible base URL (default for all agents)"
    )
    global_group.add_argument(
        "--use-vertexai",
        action="store_true",
        default=False,
        help="Route Gemini through Vertex AI (default for all agents)",
    )
    global_group.add_argument(
        "--temperature", type=float, default=None, help="LLM temperature (default for all agents)"
    )

    # ── Per-agent LLM overrides ───────────────────────────────────────────
    def _add_agent_args(group_name: str, prefix: str, description: str) -> None:
        group = parser.add_argument_group(group_name, description)
        group.add_argument(f"--{prefix}-provider", choices=["openai", "gemini"], default=None)
        group.add_argument(f"--{prefix}-model", default=None)
        group.add_argument(f"--{prefix}-api-key", default=None)
        group.add_argument(f"--{prefix}-base-url", default=None)
        group.add_argument(f"--{prefix}-temperature", type=float, default=None)
        group.add_argument(
            f"--{prefix}-use-vertexai", action="store_true", default=None,
            help=argparse.SUPPRESS,
        )

    _add_agent_args(
        "UserAgent LLM", "user-agent",
        "LLM settings for the oracle / database-owner agent",
    )
    _add_agent_args(
        "ExplanationAgent LLM", "explanation-agent",
        "LLM settings for the database investigation agent",
    )
    _add_agent_args(
        "FixAgent LLM", "fix-agent",
        "LLM settings for the database repair agent",
    )
    _add_agent_args(
        "Judge LLM", "judge",
        "LLM settings for the explanation quality judge (defaults to UserAgent settings)",
    )

    # ── Pipeline settings ─────────────────────────────────────────────────
    pipeline_group = parser.add_argument_group("Pipeline settings")
    pipeline_group.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of records to process (0 = all)",
    )
    pipeline_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    pipeline_group.add_argument(
        "--max-explanation-turns",
        type=int,
        default=config.MAX_EXPLANATION_TURNS,
        help="Max turns for the ExplanationAgent",
    )
    pipeline_group.add_argument(
        "--max-fix-turns",
        type=int,
        default=config.MAX_FIX_TURNS,
        help="Max turns for the FixAgent",
    )
    pipeline_group.add_argument(
        "--question-penalty",
        type=float,
        default=config.QUESTION_PENALTY,
        help="Score penalty per question asked by the FixAgent",
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument(
        "--dataset",
        default=str(config.DATASET_PATH),
        help="Path to dataset.json produced by the generation pipeline",
    )
    path_group.add_argument(
        "--db-dir",
        default=str(config.DB_BASE_DIR),
        help="Root directory containing per-database SQLite folders",
    )
    path_group.add_argument(
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


def _resolve_agent_config(
    args: argparse.Namespace,
    prefix: str,
    global_fallback: AgentLLMConfig,
    env_config: AgentLLMConfig,
) -> AgentLLMConfig:
    """
    Build an AgentLLMConfig with 3-level priority:
      1. Per-agent CLI flag  (highest)
      2. Global CLI flag
      3. Env-var agent config  (lowest)
    """
    def _get(attr: str):
        return getattr(args, f"{prefix}_{attr}", None)

    provider = _get("provider") or global_fallback.provider or env_config.provider
    api_key = _get("api_key") or global_fallback.api_key or env_config.api_key
    base_url = _get("base_url") or global_fallback.base_url or env_config.base_url

    # Only inherit env_config.model when it belongs to the *same* provider;
    # otherwise fall back to the provider-appropriate default.
    explicit_model = _get("model") or global_fallback.model
    if explicit_model:
        model = explicit_model
    elif env_config.provider == provider:
        model = env_config.model or (
            config.GEMINI_MODEL if provider == "gemini" else config.OPENAI_MODEL
        )
    else:
        model = config.GEMINI_MODEL if provider == "gemini" else config.OPENAI_MODEL
    temperature = (
        _get("temperature")
        if _get("temperature") is not None
        else global_fallback.temperature
    )
    use_vertexai = (
        _get("use_vertexai")
        if _get("use_vertexai") is not None
        else (global_fallback.use_vertexai or env_config.use_vertexai)
    )

    return AgentLLMConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        use_vertexai=use_vertexai,
    )


def _build_llm(agent_config: AgentLLMConfig) -> LLMClient | GeminiClient:
    """Construct an LLM client from an AgentLLMConfig."""
    if agent_config.provider == "gemini":
        return GeminiClient(
            api_key=agent_config.api_key or None,
            model=agent_config.model or None,
            temperature=agent_config.temperature,
            use_vertexai=agent_config.use_vertexai or None,
        )
    return LLMClient(
        api_key=agent_config.api_key or None,
        model=agent_config.model or None,
        base_url=agent_config.base_url or None,
        temperature=agent_config.temperature,
    )


def _save_results(results: list[RunResult], output_dir: Path) -> tuple[dict, Path, Path]:
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
        avg_fix_score = sum(e.fix_score for e in evals) / n
        avg_explanation_score = sum(e.explanation_score for e in evals) / n
        avg_final_score = sum(e.final_score for e in evals) / n
        avg_questions = sum(e.questions_asked for e in evals) / n

        stats = {
            "total_records": n,
            "avg_fix_score": round(avg_fix_score, 4),
            "avg_explanation_score": round(avg_explanation_score, 4),
            "avg_final_score": round(avg_final_score, 4),
            "avg_questions_asked": round(avg_questions, 2),
            "explanation_score_distribution": {
                "score_0.0": sum(1 for e in evals if e.explanation_score == 0.0),
                "score_0.5": sum(1 for e in evals if e.explanation_score == 0.5),
                "score_1.0": sum(1 for e in evals if e.explanation_score == 1.0),
            },
            "fix_score_distribution": {
                "score_0.0": sum(1 for e in evals if e.fix_score == 0.0),
                "score_1.0": sum(1 for e in evals if e.fix_score == 1.0),
                "score_1.5": sum(1 for e in evals if e.fix_score == 1.5),
            },
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

    # ── Build per-agent LLM clients ───────────────────────────────────────
    # Global fallback config from CLI global flags + env vars
    global_fallback = AgentLLMConfig(
        provider=args.provider,
        api_key=args.api_key or "",
        model=args.model or "",
        base_url=args.base_url,
        temperature=args.temperature if args.temperature is not None else 0.3,
        use_vertexai=args.use_vertexai,
    )
    # Merge: per-agent CLI → global CLI → env-var per-agent config
    user_cfg = _resolve_agent_config(args, "user_agent", global_fallback, config.USER_AGENT_CONFIG)
    explanation_cfg = _resolve_agent_config(args, "explanation_agent", global_fallback, config.EXPLANATION_AGENT_CONFIG)
    fix_cfg = _resolve_agent_config(args, "fix_agent", global_fallback, config.FIX_AGENT_CONFIG)
    # Judge falls back to user_agent env config (not global)
    judge_cfg = _resolve_agent_config(args, "judge", global_fallback, config.JUDGE_CONFIG)

    user_llm = _build_llm(user_cfg)
    explanation_llm = _build_llm(explanation_cfg)
    fix_llm = _build_llm(fix_cfg)
    judge_llm = _build_llm(judge_cfg)

    logger.info(
        "UserAgent:       provider=%s  model=%s", user_cfg.provider, user_llm.model
    )
    logger.info(
        "ExplanationAgent: provider=%s  model=%s", explanation_cfg.provider, explanation_llm.model
    )
    logger.info(
        "FixAgent:        provider=%s  model=%s", fix_cfg.provider, fix_llm.model
    )
    logger.info(
        "Judge:           provider=%s  model=%s", judge_cfg.provider, judge_llm.model
    )
    logger.info(
        "Workers: %d  explanation_turns: %d  fix_turns: %d",
        args.workers, args.max_explanation_turns, args.max_fix_turns,
    )
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
                user_llm=user_llm,
                explanation_llm=explanation_llm,
                fix_llm=fix_llm,
                judge_llm=judge_llm,
                db_base_dir=db_base_dir,
                sandbox_dir=sandbox_dir,
                max_explanation_turns=args.max_explanation_turns,
                max_fix_turns=args.max_fix_turns,
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
    logger.info("Avg fix score:          %s", stats.get("avg_fix_score", "N/A"))
    logger.info("Avg explanation score:  %s", stats.get("avg_explanation_score", "N/A"))
    logger.info("Avg final score:        %s", stats.get("avg_final_score", "N/A"))
    logger.info("Avg questions asked:    %s", stats.get("avg_questions_asked", "N/A"))
    if errors:
        logger.warning("Failed records: %s", ", ".join(errors))
    logger.info("Results: %s", results_path)
    logger.info("Stats:   %s", stats_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
