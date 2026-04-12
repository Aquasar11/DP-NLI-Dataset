"""
ExplanationAgent — independently investigates and explains the database alteration.

The agent operates fully autonomously using only direct database queries
(run_query tool). It does not interact with the UserAgent or any human.

The goal is to identify WHAT changed in the database without being told —
it must discover the issue on its own through investigation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

from database_utils import run_select_query
from llm_client import GeminiClient, LLMClient
from models import (
    DatasetRecord,
    ExplanationAgentStep,
    ExplanationResult,
)
from prompts import EXPLANATION_AGENT_SYSTEM_PROMPT
from sample_logger import PipelineLogger

import config

logger = logging.getLogger(__name__)


class ExplanationAgent:
    """
    Investigates a data anomaly and produces a human-readable explanation.

    The agent operates fully autonomously using a structured action loop:
      - ``run_query``: execute a SELECT directly on the altered database.
      - ``done``: conclude with a full explanation and root-cause statement.
    """

    def __init__(
        self,
        record: DatasetRecord,
        llm: Union[LLMClient, GeminiClient],
        altered_db_path: Path,
        max_turns: int | None = None,
        pipeline_logger: PipelineLogger | None = None,
    ) -> None:
        self._record = record
        self._llm = llm
        self._altered_db_path = altered_db_path
        self._max_turns = max_turns if max_turns is not None else config.MAX_EXPLANATION_TURNS
        self._pipeline_logger = pipeline_logger

        ddl = self._get_ddl()
        self._system_prompt = EXPLANATION_AGENT_SYSTEM_PROMPT.format_map(
            {
                "db_id": record.db_id,
                "question": record.question,
                "evidence": record.evidence,
                "gold_sql": record.gold_sql,
                "altered_result": json.dumps(record.altered_result, default=str),
                "follow_up_question": record.follow_up_question,
                "ddl": ddl,
                "explanation_query_penalty": config.EXPLANATION_QUERY_PENALTY,
                "max_turns": self._max_turns,
            }
        )

    def _get_ddl(self) -> str:
        """Extract DDL from the altered database for context."""
        from database_utils import get_ddl
        try:
            return get_ddl(self._altered_db_path)
        except Exception as exc:
            logger.warning("[ExplanationAgent] could not read DDL: %s", exc)
            return "(schema unavailable)"

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, user_agent: Any = None) -> ExplanationResult:
        """
        Run the explanation investigation loop.

        The agent operates fully autonomously using only run_query tool calls.
        The user_agent parameter is accepted for backward compatibility but is
        never called.

        Returns:
            :class:`ExplanationResult` with the full explanation and conversation log.
        """
        logger.info(
            "[ExplanationAgent] starting investigation for record=%d  db=%s",
            self._record.id, self._record.db_id,
        )

        conversation: list = []
        agent_messages: list[dict[str, str]] = []
        turns_used = 0

        for turn in range(1, self._max_turns + 1):
            logger.debug(
                "[ExplanationAgent] turn %d/%d  message_count=%d",
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
                        "submit your explanation immediately, no more run_query calls are possible."
                    ),
                })
            elif turn == self._max_turns - 1:
                messages_for_llm.append({
                    "role": "user",
                    "content": (
                        f"[Turn {turn}/{self._max_turns} — 1 turn remaining] "
                        "You have only one more turn after this. "
                        "Wrap up your investigation and prepare to submit 'done' next turn."
                    ),
                })

            result, data = self._llm.chat_json(self._system_prompt, messages_for_llm)

            if self._pipeline_logger:
                self._pipeline_logger.log_llm_call(
                    agent="ExplanationAgent",
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

            if step.action == "run_query" and step.sql:
                turns_used += 1
                sql = step.sql.strip()
                # Only the first statement to avoid multi-statement injection
                first_stmt = next((s.strip() for s in sql.split(";") if s.strip()), sql)
                logger.info("[ExplanationAgent] running query (turn %d): %s", turn, first_stmt)

                query_success = True
                query_error = None
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
                    logger.warning("[ExplanationAgent] query error: %s", exc)

                if self._pipeline_logger:
                    self._pipeline_logger.log_tool_call(
                        agent="ExplanationAgent",
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

            elif step.action == "done":
                explanation = (step.explanation or "").strip()
                sql_impact = (step.sql_impact or "").strip()
                alteration_type = (step.alteration_type or "").strip()

                if not explanation:
                    logger.warning("[ExplanationAgent] 'done' with empty explanation; retrying")
                    agent_messages.append({
                        "role": "user",
                        "content": (
                            "Please provide a complete explanation, sql_impact, and alteration_type before finishing."
                        ),
                    })
                    continue

                logger.info(
                    "[ExplanationAgent] concluded after %d turn(s). Alteration type: %s",
                    turns_used, alteration_type,
                )
                return ExplanationResult(
                    record_id=self._record.id,
                    explanation=explanation,
                    sql_impact=sql_impact,
                    alteration_type=alteration_type,
                    turns_used=turns_used,
                    query_turns=turns_used,
                    conversation=conversation,
                )

            else:
                logger.warning("[ExplanationAgent] unexpected action=%s; stopping", step.action)
                break

        # Fallback if the loop ended without a "done" action
        logger.warning(
            "[ExplanationAgent] FALLBACK triggered for record=%d — assigning score 0",
            self._record.id,
        )
        return ExplanationResult(
            record_id=self._record.id,
            explanation="Agent reached maximum turns without producing an explanation.",
            sql_impact="",
            alteration_type="",
            turns_used=turns_used,
            query_turns=turns_used,
            conversation=conversation,
            is_fallback=True,
        )
