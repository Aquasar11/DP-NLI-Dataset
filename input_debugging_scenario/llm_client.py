"""
LLM client wrapper for OpenAI API calls.

Handles structured JSON output parsing, retries for API errors,
and configurable model selection.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

import config
from models import LLMAlterationResponse, LLMFollowUpResponse
from prompts import (
    ALTERATION_SYSTEM_PROMPT,
    FOLLOWUP_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Retry config for API-level errors (rate limits, timeouts)
API_MAX_RETRIES = 5
API_RETRY_BASE_DELAY = 2.0  # seconds


class LLMClient:
    """Wrapper around the OpenAI client for structured LLM interactions."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
    ):
        self.model = model or config.OPENAI_MODEL
        self.temperature = temperature if temperature is not None else config.OPENAI_TEMPERATURE

        client_kwargs: dict[str, Any] = {
            "api_key": api_key or config.OPENAI_API_KEY,
        }
        resolved_base_url = base_url or config.OPENAI_BASE_URL
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        if not client_kwargs["api_key"]:
            raise ValueError(
                "OPENAI_API_KEY is required. Set it in .env or via --api-key."
            )

        self.client = OpenAI(**client_kwargs)
        logger.info("LLM client initialized: model=%s", self.model)

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Make an API call with retry logic for transient errors.

        Returns the raw content string from the LLM response.
        """
        for attempt in range(1, API_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned empty content")
                return content.strip()

            except (RateLimitError, APITimeoutError) as e:
                delay = API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "API error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, API_MAX_RETRIES, e, delay,
                )
                time.sleep(delay)
            except APIError as e:
                if e.status_code and e.status_code >= 500:
                    delay = API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Server error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, API_MAX_RETRIES, e, delay,
                    )
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(f"API call failed after {API_MAX_RETRIES} attempts")

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code fences."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    def generate_alteration(self, user_prompt: str) -> LLMAlterationResponse:
        """
        Step 1: Generate altering SQL and explanation.

        Args:
            user_prompt: The fully-constructed alteration prompt.

        Returns:
            Parsed LLMAlterationResponse with altering_sql and explanation.
        """
        raw = self._call_api(ALTERATION_SYSTEM_PROMPT, user_prompt)
        data = self._parse_json(raw)
        return LLMAlterationResponse(**data)

    def generate_followup(self, user_prompt: str) -> LLMFollowUpResponse:
        """
        Step 2: Generate follow-up question, explanation, and fix.

        Args:
            user_prompt: The fully-constructed follow-up prompt.

        Returns:
            Parsed LLMFollowUpResponse.
        """
        raw = self._call_api(FOLLOWUP_SYSTEM_PROMPT, user_prompt)
        data = self._parse_json(raw)
        return LLMFollowUpResponse(**data)
