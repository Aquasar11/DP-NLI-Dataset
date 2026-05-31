"""
LLM client wrappers for OpenAI, Google Gen AI (Gemini), and Anthropic Claude
(via Vertex AI) API calls.

Handles structured JSON output parsing, retries for API errors,
and configurable model selection.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from anthropic import AnthropicVertex
from google import genai
from google.genai.types import GenerateContentConfig
from google.oauth2 import service_account
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from retry import retry

import config
from models import LLMAlterationResponse, LLMFollowUpResponse
from prompts import (
    ALTERATION_SYSTEM_PROMPT,
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


class RetryableLLMError(RuntimeError):
    """Raised to trigger retry-package backoff for transient LLM errors."""


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse JSON from an LLM response, stripping markdown code fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


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

        @retry(
            RetryableLLMError,
            tries=API_MAX_RETRIES,
            delay=API_RETRY_BASE_DELAY,
            backoff=2,
            logger=logger,
        )
        def _call_with_retry() -> str:
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
            except Exception as e:
                raise RetryableLLMError(str(e)) from e

        return _call_with_retry(), time.perf_counter() - _t0

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        return _parse_json(raw)

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


# Retryable HTTP status codes for the Gemini API
_GEMINI_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class GeminiClient:
    """
    Wrapper around the Google Gen AI SDK for structured LLM interactions.

    Supports both the Gemini Developer API and Vertex AI backends.  Set
    ``use_vertexai=True`` (or ``GOOGLE_GENAI_USE_VERTEXAI=True`` in the
    environment) to route requests through Vertex AI; otherwise the
    Gemini Developer API is used and an API key must be supplied.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        use_vertexai: bool | None = None,
    ):
        self.model = model or config.GEMINI_MODEL
        self.temperature = temperature if temperature is not None else config.GEMINI_TEMPERATURE

        _use_vertexai = use_vertexai if use_vertexai is not None else config.GEMINI_USE_VERTEXAI

        if _use_vertexai:
            _creds_path = config.GCP_CREDENTIALS
            # _credentials = (
            #     service_account.Credentials.from_service_account_file(
            #         _creds_path,
            #         scopes=["https://www.googleapis.com/auth/cloud-platform"],
            #     )
            #     if _creds_path
            #     else None
            # )
            self.client = genai.Client(
                vertexai=True,
                project=config.GCP_PROJECT or config.GOOGLE_CLOUD_PROJECT or None,
                location=config.GCP_REGION or config.GOOGLE_CLOUD_LOCATION,
                # credentials=_credentials,
            )
        else:
            _api_key = api_key or config.GEMINI_API_KEY
            if not _api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required when not using Vertex AI. "
                    "Set it in .env or pass api_key to GeminiClient."
                )
            self.client = genai.Client(api_key=_api_key, location=config.GOOGLE_CLOUD_LOCATION)

        logger.info("Gemini client initialized: model=%s, vertexai=%s", self.model, _use_vertexai)

    def _call_api(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        """
        Make a Gemini API call with retry logic for transient errors.

        Returns the raw content string and the elapsed time in seconds.
        """
        _t0 = time.perf_counter()

        @retry(
            RetryableLLMError,
            tries=API_MAX_RETRIES,
            delay=API_RETRY_BASE_DELAY,
            backoff=2,
            logger=logger,
        )
        def _call_with_retry() -> str:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.temperature,
                        response_mime_type="application/json",
                    ),
                )
                content = response.text
                if content is None:
                    raise ValueError("Gemini returned empty content")
                return content.strip()
            except Exception as e:
                raise RetryableLLMError(str(e)) from e

        return _call_with_retry(), time.perf_counter() - _t0

    # Re-use the module-level JSON parser.
    _parse_json = staticmethod(_parse_json)

    def generate_alteration(self, user_prompt: str) -> LLMCallResult[LLMAlterationResponse]:
        """Step 1: Generate altering SQL and explanation."""
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
        """Step 2: Generate follow-up question, explanation, and fix."""
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


# ── Anthropic Claude on Vertex AI ────────────────────────────────────────────

_CLAUDE_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class ClaudeVertexClient:
    """
    Wrapper around the Anthropic Vertex AI SDK for structured LLM interactions.

    JSON responses are elicited via an instruction appended to the user prompt
    (Vertex Claude does not support assistant prefill or a native JSON mode).
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        project_id: str | None = None,
        region: str | None = None,
    ):
        self.model = model or config.CLAUDE_MODEL
        self.temperature = temperature if temperature is not None else config.CLAUDE_TEMPERATURE
        _project = project_id or config.GCP_PROJECT or config.GOOGLE_CLOUD_PROJECT
        _region = region or config.GCP_REGION or config.GOOGLE_CLOUD_LOCATION or "global"
        # The SDK builds the base URL as f"https://{region}-aiplatform.googleapis.com/v1".
        # For the special "global" endpoint the correct URL has no region prefix.
        _base_url = (
            "https://aiplatform.googleapis.com/v1"
            if _region == "global"
            else None
        )
        self.client = AnthropicVertex(region=_region, project_id=_project, base_url=_base_url)
        logger.info(
            "Claude Vertex client initialized: model=%s, project=%s, region=%s",
            self.model, _project, _region,
        )

    def _call_api(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        """Make a Claude Vertex API call with retry logic for transient errors."""
        _t0 = time.perf_counter()
        # Append explicit JSON instruction — Vertex Claude does not support prefill.
        full_user_prompt = (
            user_prompt + "\n\nRespond with a valid JSON object only. "
            "Do not include any text outside the JSON."
        )
        messages = [{"role": "user", "content": full_user_prompt}]

        @retry(
            RetryableLLMError,
            tries=API_MAX_RETRIES,
            delay=API_RETRY_BASE_DELAY,
            backoff=2,
            logger=logger,
        )
        def _call_with_retry() -> str:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                    temperature=self.temperature,
                )
                content = response.content[0].text if response.content else ""
                if not content:
                    raise ValueError("Claude returned empty content")
                return content.strip()
            except Exception as e:
                raise RetryableLLMError(str(e)) from e

        return _call_with_retry(), time.perf_counter() - _t0

    _parse_json = staticmethod(_parse_json)

    def generate_alteration(self, user_prompt: str) -> LLMCallResult[LLMAlterationResponse]:
        """Step 1: Generate altering SQL and explanation."""
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
        """Step 2: Generate follow-up question, explanation, and fix."""
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
