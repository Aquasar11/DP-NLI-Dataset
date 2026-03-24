"""
Configuration for the multi-agent data debugging framework.

All settings can be overridden via environment variables in the shared .env
file located at the repository root.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
_DOTENV_PATH = PROJECT_ROOT.parent / ".env"
load_dotenv(_DOTENV_PATH)

# ── OpenAI ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL") or None
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

# ── GCP / Gemini ─────────────────────────────────────────────────────────
GCP_PROJECT: str = os.getenv("GCP_PROJECT", "")
GCP_REGION: str = os.getenv("GCP_REGION", "global")
GCP_CREDENTIALS: str = os.getenv("GCP_CREDENTIALS", "")

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
GEMINI_USE_VERTEXAI: bool = (
    os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"
)
GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

# ── Agent settings ──────────────────────────────────────────────────────────
# Maximum Q&A turns the ExplanationAgent may use
MAX_EXPLANATION_TURNS: int = int(os.getenv("MAX_EXPLANATION_TURNS", "6"))
# Maximum Q&A turns the AnswerAgent may use (each incurs a score penalty)
MAX_ANSWER_TURNS: int = int(os.getenv("MAX_ANSWER_TURNS", "4"))
# Score penalty subtracted for every question asked by the AnswerAgent
QUESTION_PENALTY: float = float(os.getenv("QUESTION_PENALTY", "0.05"))

# ── Paths ───────────────────────────────────────────────────────────────────
DB_BASE_DIR: Path = Path(
    os.getenv(
        "DB_BASE_DIR",
        str(
            PROJECT_ROOT.parent
            / "data_debugging_scenario"
            / "data"
            / "train"
            / "train_databases"
        ),
    )
)

DATASET_PATH: Path = Path(
    os.getenv(
        "AGENTS_DATASET_PATH",
        str(
            PROJECT_ROOT.parent
            / "data_debugging_scenario"
            / "output"
            / "dataset.json"
        ),
    )
)

OUTPUT_DIR: Path = PROJECT_ROOT / "output"
SANDBOX_DIR: Path = PROJECT_ROOT / "sandbox"

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
