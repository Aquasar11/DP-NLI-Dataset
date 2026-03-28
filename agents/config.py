"""
Configuration for the multi-agent data debugging framework.

All settings can be overridden via environment variables in the shared .env
file located at the repository root.

Per-agent LLM settings use the prefix pattern:
  {AGENT_PREFIX}_{SETTING}   e.g. USER_AGENT_MODEL, EXPLANATION_AGENT_API_KEY
Each agent falls back to the global settings if its own env var is not set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
GEMINI_USE_VERTEXAI: bool = (
    os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"
)
GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

# ── Agent settings ──────────────────────────────────────────────────────────
# Maximum Q&A turns the ExplanationAgent may use
MAX_EXPLANATION_TURNS: int = int(os.getenv("MAX_EXPLANATION_TURNS", "6"))
# Maximum Q&A turns the FixAgent may use (each incurs a score penalty)
MAX_FIX_TURNS: int = int(os.getenv("MAX_FIX_TURNS", "4"))
# Score penalty subtracted for every question asked by the FixAgent
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


# ── Per-agent LLM configuration ─────────────────────────────────────────────

@dataclass
class AgentLLMConfig:
    """LLM settings for a single agent."""

    provider: str          # "openai" or "gemini"
    api_key: str
    model: str
    base_url: str | None
    temperature: float
    use_vertexai: bool = False


def _global_config() -> AgentLLMConfig:
    """Build the global (fallback) LLM config from top-level env vars."""
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "gemini":
        return AgentLLMConfig(
            provider="gemini",
            api_key=GEMINI_API_KEY,
            model=GEMINI_MODEL,
            base_url=None,
            temperature=GEMINI_TEMPERATURE,
            use_vertexai=GEMINI_USE_VERTEXAI,
        )
    return AgentLLMConfig(
        provider="openai",
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        temperature=OPENAI_TEMPERATURE,
    )


def _agent_config(prefix: str, fallback: AgentLLMConfig) -> AgentLLMConfig:
    """
    Read per-agent env vars, falling back to *fallback* for unset fields.

    Expected env var pattern: ``{PREFIX}_PROVIDER``, ``{PREFIX}_API_KEY``, etc.
    """
    provider = os.getenv(f"{prefix}_PROVIDER") or fallback.provider
    api_key = os.getenv(f"{prefix}_API_KEY") or fallback.api_key
    model = os.getenv(f"{prefix}_MODEL") or fallback.model
    base_url = os.getenv(f"{prefix}_BASE_URL") or fallback.base_url
    temperature_str = os.getenv(f"{prefix}_TEMPERATURE")
    temperature = float(temperature_str) if temperature_str else fallback.temperature
    use_vertexai_str = os.getenv(f"{prefix}_USE_VERTEXAI")
    use_vertexai = (
        use_vertexai_str.lower() == "true"
        if use_vertexai_str is not None
        else fallback.use_vertexai
    )
    return AgentLLMConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        use_vertexai=use_vertexai,
    )


_GLOBAL_CONFIG: AgentLLMConfig = _global_config()

USER_AGENT_CONFIG: AgentLLMConfig = _agent_config("USER_AGENT", _GLOBAL_CONFIG)
EXPLANATION_AGENT_CONFIG: AgentLLMConfig = _agent_config("EXPLANATION_AGENT", _GLOBAL_CONFIG)
FIX_AGENT_CONFIG: AgentLLMConfig = _agent_config("FIX_AGENT", _GLOBAL_CONFIG)
# Judge defaults to user agent settings so it shares the same trust boundary
JUDGE_CONFIG: AgentLLMConfig = _agent_config("JUDGE", USER_AGENT_CONFIG)
