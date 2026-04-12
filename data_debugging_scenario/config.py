"""
Configuration for the Data Debugging Dataset Generator.

All settings can be overridden via CLI arguments or environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
_DOTENV_PATH = PROJECT_ROOT.parent / ".env"
load_dotenv(_DOTENV_PATH)

# ── Paths ──────────────────────────────────────────────────────────────────────
BIRD_TRAIN_JSON = PROJECT_ROOT / "data" / "bird" / "train" / "train.json"
BIRD_TABLES_JSON = PROJECT_ROOT / "data" / "bird" / "train" / "train_tables.json"
BIRD_DB_DIR = PROJECT_ROOT / "data" / "bird" / "train" / "train_databases"

_DATA_ROOT = PROJECT_ROOT.parent / "data"

# ── BIRD dev ───────────────────────────────────────────────────────────────────
# NOTE: dev_databases.zip must be extracted to data/bird_dev/dev_databases/ first.
BIRD_DEV_JSON = _DATA_ROOT / "bird_dev" / "dev.json"
BIRD_DEV_TABLES_JSON = _DATA_ROOT / "bird_dev" / "dev_tables.json"
BIRD_DEV_DB_DIR = _DATA_ROOT / "bird_dev" / "dev_databases"

# ── Spider train ───────────────────────────────────────────────────────────────
SPIDER_TRAIN_JSON = _DATA_ROOT / "spider_data" / "train_spider.json"
SPIDER_TABLES_JSON = _DATA_ROOT / "spider_data" / "tables.json"
SPIDER_DB_DIR = _DATA_ROOT / "spider_data" / "database"

# ── Spider dev ─────────────────────────────────────────────────────────────────
SPIDER_DEV_JSON = _DATA_ROOT / "spider_data" / "dev.json"
SPIDER_DEV_TABLES_JSON = _DATA_ROOT / "spider_data" / "tables.json"
SPIDER_DEV_DB_DIR = _DATA_ROOT / "spider_data" / "database"

# ── Spider test ────────────────────────────────────────────────────────────────
SPIDER_TEST_JSON = _DATA_ROOT / "spider_data" / "test.json"
SPIDER_TEST_TABLES_JSON = _DATA_ROOT / "spider_data" / "test_tables.json"
SPIDER_TEST_DB_DIR = _DATA_ROOT / "spider_data" / "test_database"

OUTPUT_DIR = PROJECT_ROOT / "output"
SANDBOX_DIR = PROJECT_ROOT / "sandbox"

# ── OpenAI ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL", None)
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

# ── GCP ───────────────────────────────────────────────────────────────────────
GCP_PROJECT: str = os.getenv("GCP_PROJECT", "")
GCP_REGION: str = os.getenv("GCP_REGION", "global")
GCP_CREDENTIALS: str = os.getenv("GCP_CREDENTIALS", "")

# ── Gemini (Google Gen AI SDK) ─────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
# Set GOOGLE_GENAI_USE_VERTEXAI=True to route through Vertex AI instead of the
# Gemini Developer API. When using Vertex AI, GOOGLE_CLOUD_PROJECT and
# GOOGLE_CLOUD_LOCATION must also be set (or exported as env vars).
GEMINI_USE_VERTEXAI: bool = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"
GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

# ── Pipeline ───────────────────────────────────────────────────────────────────
SAMPLE_COUNT: int = 10          # 0 = process all samples
RANDOM_SEED: int = 42
DELETE_PROBABILITY: float = 0.5  # probability of full-row deletion vs column modification
MAX_TARGET_RECORDS: int = 5      # max number of result records to alter per sample
MAX_RETRIES: int = 3             # retry LLM on validation failure

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
