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
from pathlib import Path
from typing import Union

from database_utils import get_ddl, run_select_query
from llm_client import GeminiClient, LLMClient
from models import (
    ConversationTurn,
    DatasetRecord,
    ExplanationResult,
    FixAgentStep,
    FixResult,
)
from prompts import FIX_AGENT_SYSTEM_PROMPT
from sample_logger import PipelineLogger
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
        pipeline_logger: PipelineLogger | None = None,
        altered_db_path: Path | None = None,
    ) -> None:
        self._record = record
        self._explanation = explanation
        self._llm = llm
        self._max_turns = max_turns if max_turns is not None else config.MAX_FIX_TURNS
        self._question_penalty = (
            question_penalty if question_penalty is not None else config.ASK_QUESTION_PENALTY
        )
        self._pipeline_logger = pipeline_logger
        self._altered_db_path = altered_db_path

        ddl = self._get_ddl(record)
        self._system_prompt = FIX_AGENT_SYSTEM_PROMPT.format_map(
            {
                "db_id": record.db_id,
                "question": record.question,
                "follow_up_question": record.follow_up_question,
                "gold_sql": record.gold_sql,
                "altered_result": json.dumps(record.altered_result, default=str),
                "explanation": explanation.explanation,
                "sql_impact": explanation.sql_impact or "(not provided)",
                "alteration_type": explanation.alteration_type,
                "question_penalty": self._question_penalty,
                "fix_query_penalty": config.FIX_QUERY_PENALTY,
                "ddl": ddl,
                "max_turns": self._max_turns,
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

    def run(
        self,
        user_agent: UserAgent,
        retry_context: str | None = None,
    ) -> FixResult:
        """
        Run the fix generation loop.

        Args:
            user_agent: The oracle agent that can answer clarifying questions.
            retry_context: Optional feedback from a previous failed attempt.
                If provided, it is injected as the first user message so the
                agent knows what went wrong.

        Returns:
            :class:`FixResult` with the fix SQL, confidence, and conversation log.
        """
        logger.info(
            "[FixAgent] starting for record=%d  db=%s  retry=%s",
            self._record.id, self._record.db_id, retry_context is not None,
        )

        conversation: list[ConversationTurn] = []
        agent_messages: list[dict[str, str]] = []
        questions_asked = 0
        query_turns = 0

        if retry_context:
            agent_messages.append({"role": "user", "content": retry_context})

        for turn in range(1, self._max_turns + 1):
            logger.debug(
                "[FixAgent] turn %d/%d  message_count=%d",
                turn, self._max_turns, len(agent_messages),
            )

            messages_for_llm = list(agent_messages)
            if turn == self._max_turns:
                messages_for_llm.append({
                    "role": "user",
                    "content": (
                        f"[FINAL TURN {turn}/{self._max_turns}] "
                        "This is your last allowed turn. "
                        "You MUST respond with action='done' right now — "
                        "submit your fix SQL immediately, no more tool calls are possible."
                    ),
                })
            elif turn == self._max_turns - 1:
                messages_for_llm.append({
                    "role": "user",
                    "content": (
                        f"[Turn {turn}/{self._max_turns} — 1 turn remaining] "
                        "You have only one more turn after this. "
                        "Prepare to submit 'done' with your fix SQL in the next turn."
                    ),
                })

            result, data = self._llm.chat_json(self._system_prompt, messages_for_llm)

            if self._pipeline_logger:
                self._pipeline_logger.log_llm_call(
                    agent="FixAgent",
                    step=f"turn_{turn}",
                    system_prompt=self._system_prompt,
                    messages=list(agent_messages),
                    raw_response=result.content if result.content else None,
                    parsed_response=data,
                    success=result.success,
                    error=result.error,
                    duration_seconds=result.duration_seconds,
                )

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

            if step.action == "run_query" and step.sql:
                query_turns += 1
                sql = step.sql.strip()
                first_stmt = next((s.strip() for s in sql.split(";") if s.strip()), sql)
                logger.info("[FixAgent] running query (turn %d): %s", turn, first_stmt)

                query_success = True
                query_error = None
                if self._altered_db_path:
                    try:
                        rows = run_select_query(self._altered_db_path, first_stmt, max_rows=500)
                        tool_result = (
                            f"Query succeeded ({len(rows)} row(s) shown, results capped at 500): "
                            + json.dumps(rows[:20], default=str)
                        )
                    except Exception as exc:
                        query_success = False
                        query_error = str(exc)
                        tool_result = f"Query failed: {exc}"
                        logger.warning("[FixAgent] query error: %s", exc)
                else:
                    tool_result = "Query tool unavailable: no database path provided."
                    query_success = False

                if self._pipeline_logger:
                    self._pipeline_logger.log_tool_call(
                        agent="FixAgent",
                        step=f"turn_{turn}",
                        tool="run_query",
                        input_data={"sql": first_stmt},
                        output_data=tool_result,
                        success=query_success,
                        error=query_error,
                    )

                agent_messages.append({"role": "assistant", "content": json.dumps(data)})
                agent_messages.append({
                    "role": "user",
                    "content": f"Query result: {tool_result}",
                })

            elif step.action == "ask_question" and step.question:
                questions_asked += 1
                question = step.question.strip()
                logger.info(
                    "[FixAgent] asking question #%d (penalty ×%d): %s",
                    questions_asked, questions_asked, question,
                )

                answer = user_agent.respond(question)
                logger.info("[FixAgent] user_agent answered: %s", answer)

                if self._pipeline_logger:
                    self._pipeline_logger.log_tool_call(
                        agent="FixAgent",
                        step=f"turn_{turn}",
                        tool="ask_question",
                        input_data={"question": question},
                        output_data=answer,
                        success=True,
                    )

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
                reasoning = (step.reasoning or "").strip()

                if is_fallback:
                    logger.warning(
                        "[FixAgent] FALLBACK triggered for record=%d — empty fix_sql, assigning score 0",
                        self._record.id,
                    )
                else:
                    logger.info(
                        "[FixAgent] concluded after %d question(s).  fix_sql=%s",
                        questions_asked, fix_sql,
                    )

                return FixResult(
                    record_id=self._record.id,
                    fix_sql=fix_sql,
                    reasoning=reasoning,
                    questions_asked=questions_asked,
                    query_turns=query_turns,
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
            reasoning="Agent reached maximum turns without producing a fix.",
            questions_asked=questions_asked,
            query_turns=query_turns,
            conversation=conversation,
            is_fallback=True,
        )
