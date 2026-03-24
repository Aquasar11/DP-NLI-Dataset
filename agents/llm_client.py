"""
LLM client wrappers for OpenAI and Google Gen AI (Gemini) API calls.

Provides a general-purpose multi-turn chat interface with automatic retry
logic for transient API errors, supporting both plain-text and JSON-structured
responses from either the OpenAI-compatible API or Google Vertex AI / Gemini
Developer API.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types as genai_types
from google.genai.types import GenerateContentConfig
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

import config

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Result of a single LLM API call."""

    content: str
    duration_seconds: float
    success: bool
    error: str | None = None

# ── Retry configuration ──────────────────────────────────────────────────────

API_MAX_RETRIES = 5
API_RETRY_BASE_DELAY = 2.0  # seconds


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown ```json ... ``` fences, if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)
    return text.strip()


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a JSON response from an LLM, tolerating markdown code fences."""
    return json.loads(_strip_markdown_fences(raw))


# ── OpenAI-compatible client ─────────────────────────────────────────────────

class LLMClient:
    """
    Wrapper around the OpenAI-compatible Chat Completions API.

    Supports multi-turn conversations via the ``messages`` list and can
    return either plain-text or JSON-object responses.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.model = model or config.OPENAI_MODEL
        self.temperature = (
            temperature if temperature is not None else config.OPENAI_TEMPERATURE
        )

        client_kwargs: dict[str, Any] = {
            "api_key": api_key or config.OPENAI_API_KEY,
        }
        resolved_base_url = base_url or config.OPENAI_BASE_URL
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        if not client_kwargs["api_key"]:
            raise ValueError(
                "OPENAI_API_KEY is required. Set it in .env or pass api_key."
            )

        self.client = OpenAI(**client_kwargs)
        logger.info("LLM client initialized: model=%s", self.model)

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        response_format: str = "text",
    ) -> ChatResult:
        """
        Send a multi-turn conversation and return the model's response.

        Args:
            system_prompt: Global instruction context placed before all messages.
            messages: Ordered turns as ``[{"role": "user"|"assistant", "content": "..."}]``.
            response_format: ``"text"`` for plain text or ``"json"`` for a JSON object.

        Returns:
            :class:`ChatResult` — successful or not.
        """
        _t0 = time.perf_counter()
        _fmt = (
            {"type": "json_object"} if response_format == "json" else {"type": "text"}
        )
        api_messages = [{"role": "system", "content": system_prompt}] + messages

        for attempt in range(1, API_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=api_messages,
                    response_format=_fmt,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM returned empty content")
                return ChatResult(
                    content=content.strip(),
                    duration_seconds=round(time.perf_counter() - _t0, 3),
                    success=True,
                )

            except (RateLimitError, APITimeoutError) as exc:
                delay = API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "API transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, API_MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)

            except APIError as exc:
                if exc.status_code and exc.status_code >= 500:
                    delay = API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Server error %d (attempt %d/%d): %s — retrying in %.1fs",
                        exc.status_code, attempt, API_MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    return ChatResult(
                        content="",
                        duration_seconds=round(time.perf_counter() - _t0, 3),
                        success=False,
                        error=str(exc),
                    )

        return ChatResult(
            content="",
            duration_seconds=round(time.perf_counter() - _t0, 3),
            success=False,
            error=f"API call failed after {API_MAX_RETRIES} attempts",
        )

    def chat_json(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> tuple[ChatResult, dict[str, Any] | None]:
        """
        Convenience wrapper: call :meth:`chat` with ``response_format="json"``
        and parse the returned JSON.

        Returns:
            ``(ChatResult, parsed_dict)`` — ``parsed_dict`` is ``None`` on error.
        """
        result = self.chat(system_prompt, messages, response_format="json")
        if not result.success:
            return result, None
        try:
            return result, parse_json_response(result.content)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "JSON parse error: %s — raw snippet: %.200s", exc, result.content
            )
            return (
                ChatResult(
                    content=result.content,
                    duration_seconds=result.duration_seconds,
                    success=False,
                    error=f"JSON parse error: {exc}",
                ),
                None,
            )


# ── Retryable HTTP statuses for Gemini ───────────────────────────────────────

_GEMINI_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class GeminiClient:
    """
    Wrapper around the Google Gen AI SDK for chat interactions.

    Supports multi-turn conversations by converting the OpenAI-style
    ``messages`` list to a :class:`google.genai.types.Content` sequence.
    Routes requests through Vertex AI when ``use_vertexai=True``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        use_vertexai: bool | None = None,
    ) -> None:
        self.model = model or config.GEMINI_MODEL
        self.temperature = (
            temperature if temperature is not None else config.GEMINI_TEMPERATURE
        )

        _use_vertexai = (
            use_vertexai if use_vertexai is not None else config.GEMINI_USE_VERTEXAI
        )

        if _use_vertexai:
            self.client = genai.Client(
                vertexai=True,
                project=config.GCP_PROJECT or config.GOOGLE_CLOUD_PROJECT or None,
                location=config.GCP_REGION or config.GOOGLE_CLOUD_LOCATION,
            )
        else:
            _api_key = api_key or config.GEMINI_API_KEY
            if not _api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required when not using Vertex AI. "
                    "Set it in .env or pass api_key."
                )
            self.client = genai.Client(
                api_key=_api_key,
                location=config.GOOGLE_CLOUD_LOCATION,
            )

        logger.info(
            "Gemini client initialized: model=%s, vertexai=%s",
            self.model,
            _use_vertexai,
        )

    @staticmethod
    def _to_gemini_contents(
        messages: list[dict[str, str]],
    ) -> list[genai_types.Content]:
        """Convert OpenAI-style message list to Gemini Content objects."""
        contents: list[genai_types.Content] = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part(text=msg["content"])],
                )
            )
        return contents

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        response_format: str = "text",
    ) -> ChatResult:
        """
        Send a multi-turn conversation and return the model's response.

        Args:
            system_prompt: Global instruction context.
            messages: Ordered turns as ``[{"role": "user"|"assistant", "content": "..."}]``.
            response_format: ``"text"`` for plain text or ``"json"`` for a JSON object.

        Returns:
            :class:`ChatResult` — successful or not.
        """
        _t0 = time.perf_counter()
        mime_type = (
            "application/json" if response_format == "json" else "text/plain"
        )
        contents = self._to_gemini_contents(messages)

        # Gemini requires at least one Content item; seed with a prompt when
        # the conversation is fresh (all context lives in system_instruction).
        if not contents:
            contents = [
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text="Begin.")],
                )
            ]

        for attempt in range(1, API_MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.temperature,
                        response_mime_type=mime_type,
                    ),
                )
                content = response.text
                if not content:
                    raise ValueError("Gemini returned empty content")
                return ChatResult(
                    content=content.strip(),
                    duration_seconds=round(time.perf_counter() - _t0, 3),
                    success=True,
                )

            except Exception as exc:
                status_code: int | None = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                retryable = (
                    status_code in _GEMINI_RETRYABLE_STATUS_CODES
                    if status_code is not None
                    else any(
                        kw in str(exc).lower()
                        for kw in ("rate", "quota", "timeout", "unavailable", "500", "503")
                    )
                )
                if retryable and attempt < API_MAX_RETRIES:
                    delay = API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Gemini transient error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, API_MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    return ChatResult(
                        content="",
                        duration_seconds=round(time.perf_counter() - _t0, 3),
                        success=False,
                        error=str(exc),
                    )

        return ChatResult(
            content="",
            duration_seconds=round(time.perf_counter() - _t0, 3),
            success=False,
            error=f"Gemini API call failed after {API_MAX_RETRIES} attempts",
        )

    def chat_json(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> tuple[ChatResult, dict[str, Any] | None]:
        """
        Convenience wrapper: call :meth:`chat` with ``response_format="json"``
        and parse the returned JSON.

        Returns:
            ``(ChatResult, parsed_dict)`` — ``parsed_dict`` is ``None`` on error.
        """
        result = self.chat(system_prompt, messages, response_format="json")
        if not result.success:
            return result, None
        try:
            return result, parse_json_response(result.content)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "JSON parse error: %s — raw snippet: %.200s", exc, result.content
            )
            return (
                ChatResult(
                    content=result.content,
                    duration_seconds=result.duration_seconds,
                    success=False,
                    error=f"JSON parse error: {exc}",
                ),
                None,
            )
