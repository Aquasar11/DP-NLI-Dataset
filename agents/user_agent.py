"""
UserAgent — the oracle / database-owner agent.

Acts as the human database owner who:
  - Knows what happened to the data (but does not reveal the exact DML SQL).
  - Receives a pre-computed text diff of the database changes as input.
  - Answers questions solely based on the provided context and diff — has NO
    direct SQL or database access.
  - Maintains a shared conversation history across all investigators so answers
    remain consistent throughout a single record's evaluation.
"""

from __future__ import annotations

import json
import logging
from typing import Union

from llm_client import GeminiClient, LLMClient
from models import DatasetRecord
from prompts import USER_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class UserAgent:
    """
    Oracle agent that simulates the database owner.

    The agent receives a pre-computed text diff describing the differences
    between the original and altered databases. It answers questions from
    investigating agents based solely on this context — it has NO SQL tools
    or direct database access.

    Conversation history is shared: both the ExplanationAgent and the
    FixAgent see the same conversation thread, ensuring consistency.
    """

    def __init__(
        self,
        record: DatasetRecord,
        llm: Union[LLMClient, GeminiClient],
        diff_text: str,
    ) -> None:
        self._record = record
        self._llm = llm

        self._system_prompt = USER_AGENT_SYSTEM_PROMPT.format_map(
            {
                "db_id": record.db_id,
                "question": record.question,
                "evidence": record.evidence,
                "gold_sql": record.gold_sql,
                "gold_result": json.dumps(record.gold_result, default=str),
                "altered_result": json.dumps(record.altered_result, default=str),
                "follow_up_question": record.follow_up_question,
                "alteration_type": record.alteration_type,
                "targeted_records": json.dumps(record.targeted_records, default=str),
                "alteration_explanation": record.alteration_explanation,
                "diff_text": diff_text,
            }
        )

        # Shared conversation history: alternating user/assistant messages
        # from the perspective of the LLM (questions are "user", answers are "assistant")
        self._conversation: list[dict[str, str]] = []

        logger.info(
            "[UserAgent] record=%d  db=%s  initialized (no SQL access)",
            record.id, record.db_id,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def respond(self, question: str) -> str:
        """
        Answer *question* from an investigating agent.

        Makes a single LLM call using the system prompt (which contains the
        diff text) and conversation history. No SQL tools are available.

        Returns:
            The agent's plain-text answer.
        """
        logger.debug("[UserAgent] received question: %s", question)

        messages = list(self._conversation)
        messages.append({"role": "user", "content": question})

        logger.info("[UserAgent] calling LLM (message_count=%d)", len(messages))
        result, data = self._llm.chat_json(self._system_prompt, messages)

        if not result.success or data is None:
            logger.warning("[UserAgent] FALLBACK — LLM call failed: %s", result.error)
            return f"I'm having trouble processing that right now. ({result.error})"

        # Gemini occasionally wraps the JSON object in an array — unwrap it.
        if isinstance(data, list):
            data = data[0] if data else {}

        answer = str(data.get("answer", "")).strip()
        if not answer:
            logger.warning("[UserAgent] FALLBACK — empty answer from LLM, using default")
            answer = "I don't have additional information to share on that."

        logger.debug("[UserAgent] LLM response: %.200s", answer)

        # Persist the Q&A to shared history
        self._conversation.append({"role": "user", "content": question})
        self._conversation.append({"role": "assistant", "content": answer})
        return answer
