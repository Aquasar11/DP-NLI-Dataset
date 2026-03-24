"""
AnswerAgent — determines the exact DML SQL that caused the data alteration.

Receives the explanation from the ExplanationAgent plus the original context,
then reasons toward the specific SQL statement.  The agent may optionally ask
clarifying questions from the UserAgent, but each question incurs a configurable
score penalty — so reasoning from available context is preferred.
"""

from __future__ import annotations

import json
import logging
from typing import Union

from llm_client import GeminiClient, LLMClient
from models import (
    AnswerAgentStep,
    AnswerResult,
    ConversationTurn,
    DatasetRecord,
    ExplanationResult,
)
from prompts import ANSWER_AGENT_SYSTEM_PROMPT
from user_agent import UserAgent

import config

logger = logging.getLogger(__name__)

_FALLBACK_SQL = "-- could not determine the altering SQL"


class AnswerAgent:
    """
    SQL detective that predicts the exact DML used to alter the database.

    The agent uses a structured action loop:
      - ``ask_question``: put a targeted question to the UserAgent (incurs penalty).
      - ``done``: submit the predicted SQL, confidence score, and reasoning.
    """

    def __init__(
        self,
        record: DatasetRecord,
        explanation: ExplanationResult,
        llm: Union[LLMClient, GeminiClient],
        max_turns: int | None = None,
        question_penalty: float | None = None,
    ) -> None:
        self._record = record
        self._explanation = explanation
        self._llm = llm
        self._max_turns = max_turns if max_turns is not None else config.MAX_ANSWER_TURNS
        self._question_penalty = (
            question_penalty
            if question_penalty is not None
            else config.QUESTION_PENALTY
        )

        self._system_prompt = ANSWER_AGENT_SYSTEM_PROMPT.format_map(
            {
                "db_id": record.db_id,
                "question": record.question,
                "evidence": record.evidence,
                "gold_sql": record.gold_sql,
                "gold_result": json.dumps(record.gold_result, default=str),
                "altered_result": json.dumps(record.altered_result, default=str),
                "follow_up_question": record.follow_up_question,
                "explanation": explanation.explanation,
                "root_cause": explanation.root_cause,
                "question_penalty": self._question_penalty,
            }
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, user_agent: UserAgent) -> AnswerResult:
        """
        Run the answer-detection loop.

        Args:
            user_agent: The oracle agent that can answer clarifying questions.

        Returns:
            :class:`AnswerResult` with the predicted SQL, confidence, and
            conversation log.
        """
        logger.info(
            "[AnswerAgent] starting for record=%d  db=%s",
            self._record.id, self._record.db_id,
        )

        conversation: list[ConversationTurn] = []
        agent_messages: list[dict[str, str]] = []
        questions_asked = 0

        for turn in range(1, self._max_turns + 1):
            logger.debug("[AnswerAgent] turn %d/%d", turn, self._max_turns)

            result, data = self._llm.chat_json(self._system_prompt, agent_messages)

            if not result.success or data is None:
                logger.error("[AnswerAgent] LLM call failed: %s", result.error)
                break

            # Gemini occasionally wraps the JSON object in an array — unwrap it.
            if isinstance(data, list):
                data = data[0] if data else {}

            try:
                step = AnswerAgentStep(**data)
            except Exception as exc:
                logger.warning("[AnswerAgent] could not parse step: %s — raw: %s", exc, data)
                break

            if step.action == "ask_question" and step.question:
                questions_asked += 1
                question = step.question.strip()
                logger.info(
                    "[AnswerAgent] asking question #%d (penalty ×%d): %s",
                    questions_asked, questions_asked, question,
                )

                answer = user_agent.respond(question)
                logger.info("[AnswerAgent] user_agent answered: %s", answer)

                conversation.append(ConversationTurn(role="investigator", content=question))
                conversation.append(ConversationTurn(role="user", content=answer))

                agent_messages.append({"role": "assistant", "content": json.dumps(data)})
                agent_messages.append({
                    "role": "user",
                    "content": f"Database owner's answer: {answer}",
                })

            elif step.action == "done":
                predicted_sql = (step.predicted_altering_sql or "").strip() or _FALLBACK_SQL
                confidence = step.confidence if step.confidence is not None else 0.0
                reasoning = (step.reasoning or "").strip()

                logger.info(
                    "[AnswerAgent] concluded after %d question(s). "
                    "confidence=%.2f  predicted_sql=%s",
                    questions_asked, confidence, predicted_sql,
                )

                return AnswerResult(
                    record_id=self._record.id,
                    predicted_altering_sql=predicted_sql,
                    confidence=confidence,
                    reasoning=reasoning,
                    questions_asked=questions_asked,
                    conversation=conversation,
                )

            else:
                logger.warning("[AnswerAgent] unexpected action=%s; stopping", step.action)
                break

        logger.warning(
            "[AnswerAgent] max turns (%d) reached without conclusion for record=%d",
            self._max_turns, self._record.id,
        )
        return AnswerResult(
            record_id=self._record.id,
            predicted_altering_sql=_FALLBACK_SQL,
            confidence=0.0,
            reasoning="Agent reached maximum turns without producing a prediction.",
            questions_asked=questions_asked,
            conversation=conversation,
        )
