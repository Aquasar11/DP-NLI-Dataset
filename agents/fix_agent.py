"""
FixAgent — repairs the database by generating SQL to restore its original state.

Receives the explanation from the ExplanationAgent plus the original context,
then reasons toward a SQL statement that undoes the alteration.  The agent may
optionally ask clarifying questions from the UserAgent, but each question incurs
a configurable score penalty — so reasoning from available context is preferred.
"""

from __future__ import annotations

import json
import logging
from typing import Union

from database_utils import get_ddl
from llm_client import GeminiClient, LLMClient
from models import (
    ConversationTurn,
    DatasetRecord,
    ExplanationResult,
    FixAgentStep,
    FixResult,
)
from prompts import FIX_AGENT_SYSTEM_PROMPT
from user_agent import UserAgent

import config

logger = logging.getLogger(__name__)

_FALLBACK_SQL = "-- could not determine fix SQL"


class FixAgent:
    """
    Database repair agent that produces SQL to restore the altered database.

    The agent uses a structured action loop:
      - ``ask_question``: put a targeted question to the UserAgent (incurs penalty).
      - ``done``: submit the fix SQL, confidence score, and reasoning.
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
        self._max_turns = max_turns if max_turns is not None else config.MAX_FIX_TURNS
        self._question_penalty = (
            question_penalty if question_penalty is not None else config.QUESTION_PENALTY
        )

        ddl = self._get_ddl(record)
        self._system_prompt = FIX_AGENT_SYSTEM_PROMPT.format_map(
            {
                "db_id": record.db_id,
                "gold_sql": record.gold_sql,
                "gold_result": json.dumps(record.gold_result, default=str),
                "altered_result": json.dumps(record.altered_result, default=str),
                "explanation": explanation.explanation,
                "root_cause": explanation.root_cause,
                "question_penalty": self._question_penalty,
                "ddl": ddl,
            }
        )

    def _get_ddl(self, record: DatasetRecord) -> str:
        try:
            from database_utils import get_db_path
            db_path = get_db_path(record.db_id)
            return get_ddl(db_path)
        except Exception as exc:
            logger.warning("[FixAgent] could not read DDL: %s", exc)
            return "(schema unavailable)"

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, user_agent: UserAgent) -> FixResult:
        """
        Run the fix generation loop.

        Args:
            user_agent: The oracle agent that can answer clarifying questions.

        Returns:
            :class:`FixResult` with the fix SQL, confidence, and conversation log.
        """
        logger.info(
            "[FixAgent] starting for record=%d  db=%s",
            self._record.id, self._record.db_id,
        )

        conversation: list[ConversationTurn] = []
        agent_messages: list[dict[str, str]] = []
        questions_asked = 0

        for turn in range(1, self._max_turns + 1):
            logger.debug(
                "[FixAgent] turn %d/%d  message_count=%d",
                turn, self._max_turns, len(agent_messages),
            )

            result, data = self._llm.chat_json(self._system_prompt, agent_messages)

            if not result.success or data is None:
                logger.error("[FixAgent] LLM call failed: %s", result.error)
                break

            # Gemini occasionally wraps the JSON object in an array — unwrap it.
            if isinstance(data, list):
                data = data[0] if data else {}

            try:
                step = FixAgentStep(**data)
            except Exception as exc:
                logger.warning("[FixAgent] could not parse step: %s — raw: %s", exc, data)
                break

            if step.action == "ask_question" and step.question:
                questions_asked += 1
                question = step.question.strip()
                logger.info(
                    "[FixAgent] asking question #%d (penalty ×%d): %s",
                    questions_asked, questions_asked, question,
                )

                answer = user_agent.respond(question)
                logger.info("[FixAgent] user_agent answered: %s", answer)

                conversation.append(ConversationTurn(role="FixAgent", content=question))
                conversation.append(ConversationTurn(role="UserAgent", content=answer))

                agent_messages.append({"role": "assistant", "content": json.dumps(data)})
                agent_messages.append({
                    "role": "user",
                    "content": f"Database owner's answer: {answer}",
                })

            elif step.action == "done":
                fix_sql = (step.fix_sql or "").strip() or _FALLBACK_SQL
                is_fallback = (fix_sql == _FALLBACK_SQL)
                confidence = step.confidence if step.confidence is not None else 0.0
                reasoning = (step.reasoning or "").strip()

                if is_fallback:
                    logger.warning(
                        "[FixAgent] FALLBACK triggered for record=%d — empty fix_sql, assigning score 0",
                        self._record.id,
                    )
                else:
                    logger.info(
                        "[FixAgent] concluded after %d question(s). confidence=%.2f  fix_sql=%s",
                        questions_asked, confidence, fix_sql,
                    )

                return FixResult(
                    record_id=self._record.id,
                    fix_sql=fix_sql,
                    confidence=confidence,
                    reasoning=reasoning,
                    questions_asked=questions_asked,
                    conversation=conversation,
                    is_fallback=is_fallback,
                )

            else:
                logger.warning("[FixAgent] unexpected action=%s; stopping", step.action)
                break

        logger.warning(
            "[FixAgent] FALLBACK triggered for record=%d — max turns (%d) reached, assigning score 0",
            self._record.id, self._max_turns,
        )
        return FixResult(
            record_id=self._record.id,
            fix_sql=_FALLBACK_SQL,
            confidence=0.0,
            reasoning="Agent reached maximum turns without producing a fix.",
            questions_asked=questions_asked,
            conversation=conversation,
            is_fallback=True,
        )
