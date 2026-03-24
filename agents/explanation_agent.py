"""
ExplanationAgent — investigates and explains the database alteration.

Receives the follow-up question, gold SQL, and current (altered) result, then
conducts a structured Q&A with the UserAgent to identify what changed in the
database and why it caused the unexpected query result.
"""

from __future__ import annotations

import json
import logging
from typing import Union

from llm_client import GeminiClient, LLMClient
from models import (
    ConversationTurn,
    DatasetRecord,
    ExplanationAgentStep,
    ExplanationResult,
)
from prompts import EXPLANATION_AGENT_SYSTEM_PROMPT
from user_agent import UserAgent

import config

logger = logging.getLogger(__name__)


class ExplanationAgent:
    """
    Investigates a data anomaly by questioning the UserAgent and produces
    a human-readable explanation of the root cause.

    The agent uses a structured action loop:
      - ``ask_question``: submit a targeted question to the UserAgent.
      - ``done``: conclude with a full explanation and root-cause statement.
    """

    def __init__(
        self,
        record: DatasetRecord,
        llm: Union[LLMClient, GeminiClient],
        max_turns: int | None = None,
    ) -> None:
        self._record = record
        self._llm = llm
        self._max_turns = max_turns if max_turns is not None else config.MAX_EXPLANATION_TURNS

        self._system_prompt = EXPLANATION_AGENT_SYSTEM_PROMPT.format_map(
            {
                "db_id": record.db_id,
                "question": record.question,
                "evidence": record.evidence,
                "gold_sql": record.gold_sql,
                "gold_result": json.dumps(record.gold_result, default=str),
                "altered_result": json.dumps(record.altered_result, default=str),
                "follow_up_question": record.follow_up_question,
            }
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, user_agent: UserAgent) -> ExplanationResult:
        """
        Run the explanation investigation loop.

        Args:
            user_agent: The oracle agent to direct questions to.

        Returns:
            :class:`ExplanationResult` with the full explanation and conversation log.
        """
        logger.info(
            "[ExplanationAgent] starting investigation for record=%d  db=%s",
            self._record.id, self._record.db_id,
        )

        conversation: list[ConversationTurn] = []
        agent_messages: list[dict[str, str]] = []
        turns_used = 0

        for turn in range(1, self._max_turns + 1):
            logger.debug("[ExplanationAgent] turn %d/%d", turn, self._max_turns)

            result, data = self._llm.chat_json(self._system_prompt, agent_messages)

            if not result.success or data is None:
                logger.error("[ExplanationAgent] LLM call failed: %s", result.error)
                break

            # Gemini occasionally wraps the JSON object in an array — unwrap it.
            if isinstance(data, list):
                data = data[0] if data else {}

            try:
                step = ExplanationAgentStep(**data)
            except Exception as exc:
                logger.warning("[ExplanationAgent] could not parse step: %s — raw: %s", exc, data)
                break

            if step.action == "ask_question" and step.question:
                turns_used += 1
                question = step.question.strip()
                logger.info("[ExplanationAgent] asking (turn %d): %s", turn, question)

                answer = user_agent.respond(question)
                logger.info("[ExplanationAgent] user_agent answered: %s", answer)

                conversation.append(ConversationTurn(role="investigator", content=question))
                conversation.append(ConversationTurn(role="user", content=answer))

                # Feed the Q&A back to the agent's own context
                agent_messages.append({"role": "assistant", "content": json.dumps(data)})
                agent_messages.append({
                    "role": "user",
                    "content": f"Database owner's answer: {answer}",
                })

            elif step.action == "done":
                explanation = (step.explanation or "").strip()
                root_cause = (step.root_cause or "").strip()

                if not explanation:
                    logger.warning("[ExplanationAgent] 'done' with empty explanation; retrying")
                    agent_messages.append({
                        "role": "user",
                        "content": (
                            "Please provide a complete explanation and root_cause before finishing."
                        ),
                    })
                    continue

                logger.info(
                    "[ExplanationAgent] concluded after %d turn(s). Root cause: %s",
                    turns_used, root_cause,
                )
                return ExplanationResult(
                    record_id=self._record.id,
                    explanation=explanation,
                    root_cause=root_cause,
                    turns_used=turns_used,
                    conversation=conversation,
                )

            else:
                logger.warning("[ExplanationAgent] unexpected action=%s; stopping", step.action)
                break

        # Fallback if the loop ended without a "done" action
        logger.warning(
            "[ExplanationAgent] max turns (%d) reached without conclusion for record=%d",
            self._max_turns, self._record.id,
        )
        fallback_explanation = (
            "Based on the investigation, the query results changed because some "
            "records that previously satisfied the query conditions were modified "
            "or removed from the database."
        )
        return ExplanationResult(
            record_id=self._record.id,
            explanation=fallback_explanation,
            root_cause="Data modification caused relevant records to become unavailable.",
            turns_used=turns_used,
            conversation=conversation,
        )
