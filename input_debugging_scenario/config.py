"""
Configuration for the Data Debugging Dataset Generator.

All settings can be overridden via CLI arguments or environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
BIRD_TRAIN_JSON = PROJECT_ROOT / "data" / "bird" / "train" / "train.json"
BIRD_TABLES_JSON = PROJECT_ROOT / "data" / "bird" / "train" / "train_tables.json"
BIRD_DB_DIR = PROJECT_ROOT / "data" / "bird" / "train" / "train_databases"
OUTPUT_DIR = PROJECT_ROOT / "output"
SANDBOX_DIR = PROJECT_ROOT / "sandbox"

# ── OpenAI ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL", None)
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

# ── Pipeline ───────────────────────────────────────────────────────────────────
SAMPLE_COUNT: int = 100          # 0 = process all samples
RANDOM_SEED: int = 42
DELETE_PROBABILITY: float = 0.5  # probability of full-row deletion vs column modification
MAX_TARGET_RECORDS: int = 1      # max number of result records to alter per sample
MAX_RETRIES: int = 3             # retry LLM on validation failure

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
