"""
Per-sample structured logging for the multi-agent debugging pipeline.

Every sample's full lifecycle — LLM inputs/outputs, tool calls, agent results,
and evaluation — is captured in a SampleLog and written incrementally to JSONL
files.  At pipeline end, ``consolidate()`` writes final JSON arrays.

Usage::

    sample_logger = SampleLogger(output_dir)
    ...
    sample_logger.write(sample_log)
    ...
    sample_logger.consolidate()
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Atomic log records ─────────────────────────────────────────────────────


@dataclass
class LLMCallLog:
    """Full record of a single LLM API call."""

    agent: str           # "ExplanationAgent" | "FixAgent" | "UserAgent" | "Judge"
    step: str            # "turn_1", "turn_2", "respond_1", "judge"
    system_prompt: str
    messages: list[dict[str, str]]
    raw_response: str | None
    parsed_response: dict[str, Any] | list | None
    success: bool
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class ToolCallLog:
    """Record of a tool call made by an agent."""

    agent: str           # "ExplanationAgent" | "FixAgent"
    step: str            # "turn_1", "turn_2"
    tool: str            # "run_query" | "ask_question"
    input_data: dict[str, Any]
    output_data: str
    success: bool
    error: str | None = None


# ── Pipeline event collector ───────────────────────────────────────────────


class PipelineLogger:
    """
    Collects pipeline events (LLM calls, tool calls) in order.

    Passed to agents so they can log their internal operations without
    coupling to the file-writing SampleLogger.
    """

    def __init__(self) -> None:
        self.events: list[LLMCallLog | ToolCallLog] = []

    def log_llm_call(
        self,
        *,
        agent: str,
        step: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        raw_response: str | None,
        parsed_response: dict[str, Any] | list | None,
        success: bool,
        error: str | None = None,
        duration_seconds: float = 0.0,
    ) -> None:
        self.events.append(LLMCallLog(
            agent=agent,
            step=step,
            system_prompt=system_prompt,
            messages=messages,
            raw_response=raw_response,
            parsed_response=parsed_response,
            success=success,
            error=error,
            duration_seconds=duration_seconds,
        ))

    def log_tool_call(
        self,
        *,
        agent: str,
        step: str,
        tool: str,
        input_data: dict[str, Any],
        output_data: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        self.events.append(ToolCallLog(
            agent=agent,
            step=step,
            tool=tool,
            input_data=input_data,
            output_data=output_data,
            success=success,
            error=error,
        ))


# ── Top-level per-sample record ───────────────────────────────────────────


@dataclass
class SampleLog:
    """Complete log for one pipeline sample — success or failure."""

    # Identity
    record_id: int
    db_id: str
    question: str
    evidence: str

    # Gold
    gold_sql: str
    gold_result: list[dict[str, Any]]
    alteration_type: str
    altering_sql: str
    altered_result: list[dict[str, Any]]
    alteration_explanation: str
    follow_up_question: str

    # Pipeline context
    sandbox_path: str
    diff_tables_count: int
    diff_text: str

    # Ordered pipeline events (LLM calls + tool calls)
    events: list[dict[str, Any]] = field(default_factory=list)

    # Agent results
    explanation_result: dict[str, Any] | None = None
    fix_result: dict[str, Any] | None = None
    evaluation_result: dict[str, Any] | None = None

    # Outcome
    status: str = "success"       # "success" | "error"
    error: str | None = None
    total_duration_seconds: float = 0.0
    timestamp_start: str = ""


# ── Writer ─────────────────────────────────────────────────────────────────


class SampleLogger:
    """
    Incrementally writes SampleLog records to JSONL files.

    * Successful samples  -> ``sample_logs.jsonl``
    * Failed / errored    -> ``failed_samples.jsonl``

    Call ``consolidate()`` at the end to produce final JSON arrays.
    """

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._success_path = output_dir / "sample_logs.jsonl"
        self._failure_path = output_dir / "failed_samples.jsonl"
        self._success_fh = open(self._success_path, "a", encoding="utf-8")
        self._failure_fh = open(self._failure_path, "a", encoding="utf-8")

    def write(self, log: SampleLog) -> None:
        """Serialise *log* and write to the appropriate JSONL file."""
        line = json.dumps(asdict(log), ensure_ascii=False, default=str) + "\n"
        if log.status == "success":
            self._success_fh.write(line)
            self._success_fh.flush()
        else:
            self._failure_fh.write(line)
            self._failure_fh.flush()

    def consolidate(self) -> None:
        """
        Read back the JSONL files and produce consolidated JSON arrays.

        Output:
            sample_logs.json      - array of successful sample logs
            failed_samples.json   - array of failed / skipped sample logs
        """
        self._success_fh.close()
        self._failure_fh.close()

        for jsonl_path in (self._success_path, self._failure_path):
            json_path = jsonl_path.with_suffix(".json")
            records: list[dict[str, Any]] = []
            if jsonl_path.exists():
                with open(jsonl_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2, ensure_ascii=False, default=str)
            logger.info(
                "Consolidated %d records -> %s",
                len(records),
                json_path,
            )
