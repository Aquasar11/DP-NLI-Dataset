"""
UserAgent — the oracle / database-owner agent.

Acts as the human database owner who:
  - Knows what happened to the data (but does not reveal the exact DML SQL).
  - Has access to BOTH the original and altered databases.
  - Is aware of the diff between them but only reveals info when directly asked.
  - Maintains a shared conversation history across all investigators so answers
    remain consistent throughout a single record's evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

from database_utils import (
    compare_databases,
    create_altered_sandbox,
    destroy_sandbox,
    get_db_path,
    get_ddl,
    run_select_query,
)
from llm_client import GeminiClient, LLMClient
from models import DatasetRecord
from prompts import USER_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Maximum inner tool-use iterations before forcing a plain response
_MAX_INNER_ITERATIONS = 3


class UserAgent:
    """
    Oracle agent that simulates the database owner.

    The agent has access to both the original and altered databases.  It
    answers questions from investigating agents but must not expose the literal
    DML that caused the alteration, and must not volunteer information beyond
    what is directly asked.

    Conversation history is shared: both the ExplanationAgent and the
    FixAgent see the same conversation thread, ensuring consistency.
    """

    def __init__(
        self,
        record: DatasetRecord,
        llm: Union[LLMClient, GeminiClient],
        db_base_dir: Path | None = None,
        sandbox_dir: Path | None = None,
    ) -> None:
        self._record = record
        self._llm = llm

        # Original (unaltered) database path
        self._original_db_path = get_db_path(record.db_id, db_base_dir)

        # Create an altered sandbox for authentic post-alteration query results
        self._sandbox_path = create_altered_sandbox(
            record.db_id,
            record.altering_sql,
            sandbox_dir=sandbox_dir,
            db_base_dir=db_base_dir,
        )
        logger.info(
            "[UserAgent] record=%d  db=%s  sandbox=%s",
            record.id, record.db_id, self._sandbox_path,
        )

        # Pre-compute the diff between original and altered databases.
        # This is stored internally and only shared when directly asked.
        self._db_identical, self._db_diff_summary = compare_databases(
            self._original_db_path, self._sandbox_path
        )
        if not self._db_identical:
            logger.debug("[UserAgent] diff detected: %s", self._db_diff_summary)

        ddl = get_ddl(self._sandbox_path)
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
                "ddl": ddl,
            }
        )

        # Shared conversation history: alternating user/assistant messages
        # from the perspective of the LLM (questions are "user", answers are "assistant")
        self._conversation: list[dict[str, str]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def sandbox_path(self) -> Path:
        """Path to the altered sandbox database (for ExplanationAgent direct queries)."""
        return self._sandbox_path

    def respond(self, question: str) -> str:
        """
        Answer *question* from an investigating agent.

        Internally runs a ReAct-style loop allowing the LLM to execute SELECT
        queries on either the original or altered database before responding.

        Returns:
            The agent's plain-text answer.
        """
        logger.debug("[UserAgent] received question: %s", question)

        messages = list(self._conversation)
        messages.append({"role": "user", "content": question})

        answer = self._react_loop(messages)

        # Persist the high-level Q&A to shared history
        self._conversation.append({"role": "user", "content": question})
        self._conversation.append({"role": "assistant", "content": answer})
        return answer

    def cleanup(self) -> None:
        """Remove the altered sandbox database from disk."""
        destroy_sandbox(self._sandbox_path)
        logger.info("[UserAgent] sandbox cleaned up for record=%d", self._record.id)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _run_query_on(self, db_path: Path, sql: str) -> tuple[list[dict[str, Any]], str | None]:
        """Execute a SELECT on *db_path*. Returns (rows, error_or_None)."""
        first_stmt = next(
            (s.strip() for s in sql.split(";") if s.strip()), sql
        )
        try:
            rows = run_select_query(db_path, first_stmt)
            return rows, None
        except Exception as exc:
            return [], str(exc)

    def _react_loop(self, messages: list[dict[str, str]]) -> str:
        """
        Inner ReAct tool-use loop.

        The LLM may issue ``run_query_original`` or ``run_query_altered``
        actions to inspect either database before providing a final ``respond``
        action.  Iteration is capped at ``_MAX_INNER_ITERATIONS``.
        """
        for iteration in range(1, _MAX_INNER_ITERATIONS + 1):
            result, data = self._llm.chat_json(self._system_prompt, messages)

            if not result.success or data is None:
                logger.warning("[UserAgent] LLM call failed: %s", result.error)
                return f"I'm having trouble processing that right now. ({result.error})"

            # Gemini occasionally wraps the JSON object in an array — unwrap it.
            if isinstance(data, list):
                data = data[0] if data else {}

            action = data.get("action", "respond")
            logger.debug("[UserAgent] inner iteration=%d  action=%s", iteration, action)

            if action in ("run_query_original", "run_query_altered"):
                sql: str = data.get("sql", "").strip()
                db_path = (
                    self._original_db_path
                    if action == "run_query_original"
                    else self._sandbox_path
                )
                db_label = "original" if action == "run_query_original" else "altered"
                logger.debug("[UserAgent] querying %s db: %s", db_label, sql)

                rows, error = self._run_query_on(db_path, sql)
                if error:
                    tool_result = f"Query on {db_label} database failed: {error}"
                    logger.warning("[UserAgent] query error (%s): %s", db_label, error)
                else:
                    tool_result = (
                        f"Query on {db_label} database succeeded ({len(rows)} row(s)): "
                        + json.dumps(rows[:20], default=str)
                    )

                messages.append({
                    "role": "assistant",
                    "content": json.dumps(data, default=str),
                })
                messages.append({
                    "role": "user",
                    "content": tool_result,
                })

            elif action == "respond":
                answer = str(data.get("answer", "")).strip()
                if not answer:
                    answer = "I don't have additional information to share on that."
                return answer

            else:
                # Unknown action — treat the whole response as the answer
                return str(data)

        # Exhausted iterations — ask for a plain text response
        messages.append({
            "role": "user",
            "content": "Please provide your final answer now.",
        })
        fallback, _ = self._llm.chat_json(self._system_prompt, messages)
        return fallback.content if fallback.success else "I was unable to formulate a response."
