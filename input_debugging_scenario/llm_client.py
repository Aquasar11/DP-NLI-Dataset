"""
LLM client wrapper for OpenAI API calls.

Handles structured JSON output parsing, retries for API errors,
and configurable model selection.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

import config
from models import LLMAlterationResponse, LLMFixResponse, LLMFollowUpResponse
from prompts import (
    ALTERATION_SYSTEM_PROMPT,
    FIX_RETRY_SYSTEM_PROMPT,
    FOLLOWUP_SYSTEM_PROMPT,
)

T = TypeVar("T")


@dataclass
class LLMCallResult(Generic[T]):
    """Wraps an LLM response together with the full raw I/O for logging."""

    parsed: T | None
    system_prompt: str
    user_prompt: str
    raw_response: str | None
    parsed_dict: dict[str, Any] | None
    duration_seconds: float
    success: bool
    error: str | None = None

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
        _t0 = time.perf_counter()
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
                return content.strip(), time.perf_counter() - _t0

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

        raise RuntimeError(f"API call failed after {API_MAX_RETRIES} attempts")  # pragma: no cover

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

    def generate_alteration(self, user_prompt: str) -> LLMCallResult[LLMAlterationResponse]:
        """
        Step 1: Generate altering SQL and explanation.

        Returns an LLMCallResult wrapping the parsed model **and** full I/O.
        On any error ``result.success`` is False and ``result.error`` is set.
        """
        try:
            raw, duration = self._call_api(ALTERATION_SYSTEM_PROMPT, user_prompt)
            data = self._parse_json(raw)
            parsed = LLMAlterationResponse(**data)
            return LLMCallResult(
                parsed=parsed,
                system_prompt=ALTERATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                raw_response=raw,
                parsed_dict=data,
                duration_seconds=round(duration, 3),
                success=True,
            )
        except Exception as e:
            return LLMCallResult(
                parsed=None,
                system_prompt=ALTERATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                raw_response=None,
                parsed_dict=None,
                duration_seconds=0.0,
                success=False,
                error=str(e),
            )

    def generate_followup(self, user_prompt: str) -> LLMCallResult[LLMFollowUpResponse]:
        """
        Step 2: Generate follow-up question, explanation, and fix.

        Returns an LLMCallResult wrapping the parsed model **and** full I/O.
        On any error ``result.success`` is False and ``result.error`` is set.
        """
        try:
            raw, duration = self._call_api(FOLLOWUP_SYSTEM_PROMPT, user_prompt)
            data = self._parse_json(raw)
            parsed = LLMFollowUpResponse(**data)
            return LLMCallResult(
                parsed=parsed,
                system_prompt=FOLLOWUP_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                raw_response=raw,
                parsed_dict=data,
                duration_seconds=round(duration, 3),
                success=True,
            )
        except Exception as e:
            return LLMCallResult(
                parsed=None,
                system_prompt=FOLLOWUP_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                raw_response=None,
                parsed_dict=None,
                duration_seconds=0.0,
                success=False,
                error=str(e),
            )

    def generate_fix_retry(self, user_prompt: str) -> LLMCallResult[LLMFixResponse]:
        """
        Step 3 retry: Generate a corrected fix SQL.

        Uses FIX_RETRY_SYSTEM_PROMPT to ask the LLM to fix a previously
        failed gold_fix SQL.  Returns an LLMCallResult wrapping the parsed
        LLMFixResponse.
        """
        try:
            raw, duration = self._call_api(FIX_RETRY_SYSTEM_PROMPT, user_prompt)
            data = self._parse_json(raw)
            parsed = LLMFixResponse(**data)
            return LLMCallResult(
                parsed=parsed,
                system_prompt=FIX_RETRY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                raw_response=raw,
                parsed_dict=data,
                duration_seconds=round(duration, 3),
                success=True,
            )
        except Exception as e:
            return LLMCallResult(
                parsed=None,
                system_prompt=FIX_RETRY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                raw_response=None,
                parsed_dict=None,
                duration_seconds=0.0,
                success=False,
                error=str(e),
            )
