# Multi-Agent Data Debugging Framework

A three-agent system that simulates a structured forensic investigation into unexplained database query result changes. Given a dataset record produced by the [data debugging pipeline](../data_debugging_scenario/README.md), three LLM-powered agents collaborate to identify a hidden data alteration and repair the database.

---

## Overview

The framework models a real-world scenario in which a database query that previously returned a known result suddenly returns something different. Three agents work together to uncover the root cause and fix it:

| Agent | Role | Input | Output |
|---|---|---|---|
| **UserAgent** | Database owner / oracle | Full alteration context; pre-computed DB diff (no SQL access) | Answers questions based on provided diff text |
| **ExplanationAgent** | Investigator | `follow_up_question`, `gold_sql`, `altered_result`, schema | Explanation + root cause (discovered autonomously via DB queries) |
| **FixAgent** | Repair engineer | Explanation from `ExplanationAgent` | SQL to restore the database to its original state |

An **Evaluator** then:
- Uses an **LLM judge** to score the `ExplanationAgent`'s explanation against the ground truth (3-level: 0.0 / 0.5 / 1.0).
- Applies the `FixAgent`'s SQL and checks whether `gold_sql` returns `gold_result` without corrupting other records. A bonus is awarded for fully restoring corrupted rows.

---

## How It Works

```
DatasetRecord
      |
      v
runner.py creates altered sandbox + computes structured DB diff
      |
      |--- diff_text (formatted table of row-level changes)
      |        |
      |        v
      |  +-----------+
      |  | UserAgent |   receives diff as text input (NO SQL access)
      |  |  (oracle) |   answers questions from FixAgent only
      |  +-----+-----+
      |        |
      |--- sandbox_path (altered DB)
      |        |
      |        v
      |  +------------------+   run_query (direct DB) / done
      |  | ExplanationAgent |------------------------------> ExplanationResult
      |  |  (autonomous)    |                                (explanation + root_cause)
      |  +------------------+
      |                                                       |
      +--- explanation passed to FixAgent                     |
               |                                              |
               v                                              |
         +----------+  ask_question (penalized) / done        |
         | FixAgent |-----------------------------------> FixResult
         | (repair) |                                     (fix_sql)
         +----------+
               |
               v
         +-----------+
         | Evaluator |---> EvaluationResult
         +-----------+     (explanation_score, fix_score, final_score)
```

### Key Design Decisions

- **ExplanationAgent is fully autonomous**: It has direct `run_query` access to the altered database and does NOT interact with the UserAgent. It must discover the root cause independently through SQL inspection.
- **UserAgent has NO SQL access**: It receives a pre-computed structured diff (original vs. altered records) as formatted text. It answers questions solely from this context.
- **FixAgent interacts with UserAgent**: It can ask clarifying questions (e.g. "what were the original column values?"), but each question incurs a score penalty.
- **Fallback detection**: When an agent fails to produce a real LLM response (timeout, max turns, empty output), the result is marked `is_fallback=True` and automatically scored 0 — the judge is skipped entirely.

### Agent Interaction Protocol

All agents communicate via structured JSON responses.

**UserAgent** (single LLM call, no tools):
```json
{"answer": "The row with id=42 no longer exists in the current database."}
```

**ExplanationAgent** (autonomous investigation loop — two actions):
```json
// Run a SELECT directly on the altered database:
{"action": "run_query", "sql": "SELECT ..."}

// Conclude with an explanation:
{"action": "done", "explanation": "...", "root_cause": "..."}
```

**FixAgent** (repair loop — three actions):
```json
// Query the altered database directly (same tool as ExplanationAgent):
{"action": "run_query", "sql": "SELECT ..."}

// Ask the UserAgent a clarifying question (incurs penalty):
{"action": "ask_question", "question": "..."}

// Submit the fix:
{"action": "done", "fix_sql": "INSERT INTO ...", "reasoning": "..."}
```

---

## Directory Structure

```
agents/
├── main.py              # CLI entry point
├── runner.py            # Per-record orchestration (sandbox, diff, agents, eval)
├── config.py            # Configuration (env vars, per-agent configs, paths)
├── models.py            # Pydantic data models
├── llm_client.py        # LLM client wrappers (OpenAI + Gemini)
├── prompts.py           # System prompt templates for all agents + judge
├── user_agent.py        # UserAgent (text-based oracle, no SQL)
├── explanation_agent.py # ExplanationAgent (autonomous, run_query only)
├── fix_agent.py         # FixAgent implementation
├── database_utils.py    # SQLite utilities (sandbox, query, DB comparison, structured diff)
├── evaluator.py         # Scoring: LLM judge + relaxed fix evaluation
├── requirements.txt     # Python dependencies
└── output/              # Generated at runtime
    ├── results.json     # Full RunResult for every processed record
    ├── stats.json       # Aggregate metrics
    └── sandbox/         # Temporary sandbox databases (auto-cleaned up)
```

---

## Input Dataset

The framework consumes `dataset.json` produced by the `data_debugging_scenario` pipeline. Each record has the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Unique record identifier |
| `db_id` | `str` | SQLite database name |
| `question` | `str` | Original natural language question |
| `evidence` | `str` | Optional hint for the query |
| `gold_sql` | `str` | SQL query whose result changed |
| `gold_result` | `list[dict]` | Expected (original) query result |
| `alteration_type` | `str` | `DELETE` or `UPDATE` |
| `targeted_records` | `list[dict]` | Rows that were affected |
| `target_columns` | `list[str]` | Columns modified (or `["all"]`) |
| `altering_sql` | `str` | **Ground-truth DML** (hidden from investigating agents) |
| `altered_result` | `list[dict]` | Query result after the alteration |
| `alteration_explanation` | `str` | Natural language explanation of the change |
| `follow_up_question` | `str` | User's confused follow-up question |

> **Default dataset path**: `../data_debugging_scenario/output/dataset.json`
> **Default database path**: `../data_debugging_scenario/data/train/train_databases/`

---

## Scoring

### Fix Score

The fix evaluation uses a **relaxed** approach — the goal is to restore the expected query result without corrupting other records, not necessarily to achieve an exact full-database match.

| `gold_result_score` | `full_restore_score` | Meaning |
|---|---|---|
| **0.0** | 0.0 | `gold_sql` does not return `gold_result` after fix |
| **1.0** | 0.0 | `gold_sql` returns `gold_result` but DB not fully restored |
| **1.0** | 1.0 | `gold_sql` returns `gold_result` AND DB is byte-identical to original |

### Per-Tool Penalty System

Every tool call incurs a configurable score deduction. All three tool types are penalized:

| Tool | Default penalty | CLI flag |
|---|---|---|
| `ExplanationAgent.run_query` | `0.01` per call | `--explanation-query-penalty` |
| `FixAgent.run_query` | `0.02` per call | `--fix-query-penalty` |
| `FixAgent.ask_question` | `0.05` per call | `--ask-question-penalty` |

```
tool_penalty = EXPLANATION_QUERY_PENALTY × explanation_query_turns
             + FIX_QUERY_PENALTY         × fix_query_turns
             + ASK_QUESTION_PENALTY      × questions_asked

final_score  = max(0.0, gold_result_score − tool_penalty)
```

The penalty breakdown is recorded in `EvaluationResult.tool_penalty_breakdown`.

### Fix Agent Retry

If the `FixAgent`'s first attempt scores 0 on the gold result check, it automatically gets one retry. The retry receives a feedback message containing:
- The previous (failed) fix SQL
- What `gold_sql` actually returned after applying it
- What the expected `gold_result` is

The retry uses the same `max_fix_turns` budget as the first attempt. The number of retries used is recorded in `FixResult.retry_count`. Configure with `--max-fix-retries` (default: `1`; set to `0` to disable).

### Explanation Score (LLM-as-Judge)

The `ExplanationAgent`'s explanation is scored by a separate **judge LLM** using a 3-level scale:

| Score | Meaning |
|---|---|
| **0.0** | Totally wrong — fails to identify any meaningful aspect of the change |
| **0.5** | Partially correct — right general area but key details wrong or incomplete |
| **1.0** | Totally correct — identifies affected table, change type, and approximate rows |

The judge reasons step-by-step before assigning a score.

### Fallback Handling

When an agent cannot produce a real LLM response (LLM call failure, max turns reached, empty output), the result is marked `is_fallback=True`. Fallback results:
- Receive an automatic **score of 0**.
- **Skip the judge entirely** (no LLM call wasted on evaluating a non-answer).
- Are clearly logged with `FALLBACK triggered` messages.

---

## Setup

### Prerequisites

- Python 3.10+

```bash
cd agents
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Set the following in the shared `.env` file at the repository root:

```bash
# OpenAI
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o                  # default model for all agents

# Gemini direct API
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3-flash-preview

# GCP / Gemini via Vertex AI
GCP_PROJECT=your-gcp-project-id
GCP_REGION=global
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=true

# Agent tuning (optional)
MAX_EXPLANATION_TURNS=6
MAX_FIX_TURNS=4
QUESTION_PENALTY=0.05

# Per-agent model overrides (optional — each falls back to global settings)
USER_AGENT_MODEL=gpt-4o
USER_AGENT_PROVIDER=openai
EXPLANATION_AGENT_MODEL=gpt-4o-mini
FIX_AGENT_MODEL=gpt-4o
JUDGE_MODEL=gpt-4o-mini

# Path overrides (optional)
DB_BASE_DIR=/path/to/train_databases
AGENTS_DATASET_PATH=/path/to/dataset.json
```

For Vertex AI, authenticate with Application Default Credentials (ADC):

```bash
gcloud auth application-default login
```

---

## Usage

```bash
cd agents
python main.py [OPTIONS]
```

### Global LLM Options

These apply to all agents unless overridden by per-agent flags.

| Flag | Default | Description |
|---|---|---|
| `--provider` | `openai` | LLM provider: `openai` or `gemini` |
| `--model` | *(from config)* | Model name |
| `--api-key` | *(from .env)* | API key |
| `--base-url` | `None` | OpenAI-compatible base URL |
| `--use-vertexai` | `False` | Route Gemini calls through Vertex AI |
| `--temperature` | *(from config)* | Sampling temperature |

### Per-Agent LLM Options

Each agent has its own set of flags that override the global settings above. Replace `{agent}` with `user-agent`, `explanation-agent`, `fix-agent`, or `judge`:

| Flag | Description |
|---|---|
| `--{agent}-provider` | `openai` or `gemini` |
| `--{agent}-model` | Model name |
| `--{agent}-api-key` | API key |
| `--{agent}-base-url` | Base URL (OpenAI-compatible) |
| `--{agent}-temperature` | Sampling temperature |

The `judge` defaults to the same settings as `user-agent` if not explicitly configured.

### Pipeline Options

| Flag | Default | Description |
|---|---|---|
| `--samples` | `0` (all) | Number of dataset records to process |
| `--workers` | `1` | Parallel workers (ThreadPoolExecutor) |
| `--max-explanation-turns` | `6` | Max turns for the `ExplanationAgent` |
| `--max-fix-turns` | `4` | Max turns for the `FixAgent` |
| `--max-fix-retries` | `1` | Retry attempts for FixAgent after a failed fix (0 = disable) |
| `--explanation-query-penalty` | `0.01` | Score penalty per `ExplanationAgent` query |
| `--fix-query-penalty` | `0.02` | Score penalty per `FixAgent` query |
| `--ask-question-penalty` | `0.05` | Score penalty per `FixAgent` question to UserAgent |
| `--question-penalty` | `0.05` | Deprecated alias for `--ask-question-penalty` |
| `--dataset` | *(from config)* | Path to `dataset.json` |
| `--db-dir` | *(from config)* | Root directory of SQLite database folders |
| `--dataset-preset` | *(none)* | Auto-set `--db-dir` from a known dataset: `bird_train`, `bird_dev`, `spider_train`, `spider_test` |
| `--output-dir` | `./output` | Directory for results and stats |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |

### Examples

**Quick test — 1 record, debug logging:**
```bash
python main.py --samples 1 --log-level DEBUG
```

**Run with different models per agent:**
```bash
python main.py \
  --user-agent-model gpt-4o \
  --explanation-agent-model gpt-4o-mini \
  --fix-agent-model gpt-4o \
  --judge-model gpt-4o-mini \
  --samples 20 \
  --workers 4
```

**Run with Gemini via Vertex AI:**
```bash
python main.py \
  --provider gemini \
  --use-vertexai \
  --samples 10 \
  --workers 5
```

**Mix providers — Gemini for investigation, OpenAI for judge:**
```bash
python main.py \
  --provider gemini \
  --use-vertexai \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --judge-api-key sk-... \
  --samples 50 \
  --workers 10
```

---

## Output

After a run, two files are written to `--output-dir`:

### `results.json`

A JSON array of `RunResult` objects, one per processed record:

```json
[
  {
    "record_id": 1,
    "db_id": "retails",
    "explanation": {
      "record_id": 1,
      "explanation": "The part 'pink powder drab lawn cyan' was deleted from the 'part' table.",
      "alteration_type": "deletion",
      "query_turns": 3,
      "turns_used": 3,
      "conversation": [],
      "is_fallback": false
    },
    "fix": {
      "record_id": 1,
      "fix_sql": "INSERT INTO part (p_partkey, p_name, ...) VALUES (42, 'pink powder drab lawn cyan', ...);",
      "reasoning": "The explanation confirms the row was deleted. I asked for the original values.",
      "questions_asked": 1,
      "query_turns": 2,
      "retry_count": 0,
      "conversation": [
        {"role": "FixAgent", "content": "What were the original column values for that part?"},
        {"role": "UserAgent", "content": "The row had p_partkey=42, p_name='pink powder drab lawn cyan', ..."}
      ],
      "is_fallback": false
    },
    "evaluation": {
      "record_id": 1,
      "db_id": "retails",
      "alteration_type_score": 1.0,
      "explanation_score": 1.0,
      "explanation_reasoning": "Correctly identified the deleted table and row.",
      "gold_result_score": 1.0,
      "full_restore_score": 1.0,
      "fix_description": "gold_result matches and DB fully restored to original",
      "questions_asked": 1,
      "explanation_query_turns": 3,
      "fix_query_turns": 2,
      "tool_penalty_breakdown": {
        "explanation_query_penalty": 0.03,
        "fix_query_penalty": 0.04,
        "ask_question_penalty": 0.05
      },
      "tool_penalty": 0.12,
      "base_score": 1.0,
      "final_score": 0.88,
      "error": null
    }
  }
]
```

### `stats.json`

Aggregate metrics across all processed records:

```json
{
  "total_records": 20,
  "avg_fix_score": 1.125,
  "avg_explanation_score": 0.775,
  "avg_final_score": 0.975,
  "avg_questions_asked": 1.15,
  "explanation_score_distribution": {
    "score_0.0": 2,
    "score_0.5": 7,
    "score_1.0": 11
  },
  "fix_score_distribution": {
    "score_0.0": 3,
    "score_1.0": 9,
    "score_1.5": 8
  }
}
```

---

## Agent Details

### UserAgent

- **No SQL access**: Receives a pre-computed structured text diff showing exactly which rows differ between the original and altered databases, formatted as readable tables.
- **Single LLM call**: Each `respond()` call makes one LLM request (no inner tool loop). The LLM answers based on the diff text, alteration context, and conversation history.
- **Information policy**: Knows the full diff but only reveals information when directly asked. Never reveals the literal DML SQL.
- **Shared history**: The same conversation history is shared across `FixAgent` sessions, ensuring consistent answers.
- **Diff computation**: `runner.py` uses `compute_structured_diff()` to identify row-level changes using primary keys (falls back to full-row comparison when PKs are unavailable), then `format_diff_as_text()` to produce readable tables.

### ExplanationAgent

- **Fully autonomous**: Does NOT interact with the UserAgent or any human. Uses only `run_query` tool calls to inspect the altered database directly.
- **Goal**: Independently discover what changed in the database — no ground-truth information is provided.
- **Turn limit**: Configurable via `--max-explanation-turns` (default: 6). Each `run_query` action consumes one turn and incurs an `explanation_query_penalty` deduction.
- **Output**: `ExplanationResult` with `explanation`, `alteration_type`, `query_turns`, `turns_used`, and `is_fallback`.

### FixAgent

- **Goal**: Produce SQL that, when applied to the altered database, restores the expected query results without corrupting other records.
- **Inputs**: `ExplanationResult` (explanation + root_cause), database schema (DDL), and query context.
- **Direct DB access**: Can run SELECT queries against the altered sandbox, just like the `ExplanationAgent`. Each query incurs a small `fix_query_penalty` deduction.
- **Oracle access**: Can ask the UserAgent clarifying questions (e.g. original column values). Each question incurs an `ask_question_penalty` deduction (larger than query penalty to discourage lazy questioning).
- **Retry**: If the first fix scores 0, the agent is automatically retried once with a feedback message describing what went wrong. Configurable via `--max-fix-retries`.
- **Turn limit**: Configurable via `--max-fix-turns` (default: 4). Shared across the initial attempt and each retry.
- **Output**: `FixResult` with `fix_sql`, `reasoning`, `questions_asked`, `query_turns`, `retry_count`, and `is_fallback`.

### Evaluator

**Explanation evaluation (LLM-as-judge)**:
- A judge LLM receives the ground-truth alteration info alongside the agent's explanation.
- Returns a score (0.0, 0.5, or 1.0) and step-by-step reasoning.
- Judge is skipped for fallback results (automatic score 0).
- Judge LLM is independently configurable and defaults to the UserAgent's LLM settings.

**Fix evaluation (relaxed — gold_result match + no corruption)**:
- A fresh sandbox is created: original DB → apply `altering_sql` → apply `fix_sql`.
- **Primary check**: Run `gold_sql` on the fixed DB. Does it return `gold_result`?
- **Restore check**: Is the fixed DB byte-identical to the original? (`full_restore_score`)

**Score formula**:
```
tool_penalty = EXPLANATION_QUERY_PENALTY × explanation_query_turns
             + FIX_QUERY_PENALTY         × fix_query_turns
             + ASK_QUESTION_PENALTY      × questions_asked

final_score  = max(0.0, gold_result_score − tool_penalty)
```

---

## Logging

All modules use Python's standard `logging` module with the same configuration as the `data_debugging_scenario` pipeline:

```
%(asctime)s [%(levelname)s] %(name)s: %(message)s
```

- **INFO**: Agent invocations, tool calls, scoring events, sandbox lifecycle.
- **DEBUG**: LLM requests/responses, query results, diff details, message counts.
- **WARNING**: Fallback triggers (with `FALLBACK` keyword), recoverable errors, retries.
- **ERROR**: Non-recoverable LLM failures.

Agent log lines include bracketed prefixes: `[UserAgent]`, `[ExplanationAgent]`, `[FixAgent]`, `[JudgeAgent]`, `[Evaluator]`.

Set log level via `--log-level DEBUG|INFO|WARNING|ERROR`.

---

## Module Reference

| Module | Key symbols |
|---|---|
| `config.py` | `AgentLLMConfig`, `USER_AGENT_CONFIG`, `EXPLANATION_AGENT_CONFIG`, `FIX_AGENT_CONFIG`, `JUDGE_CONFIG`, `MAX_EXPLANATION_TURNS`, `MAX_FIX_TURNS`, `MAX_FIX_RETRIES`, `EXPLANATION_QUERY_PENALTY`, `FIX_QUERY_PENALTY`, `ASK_QUESTION_PENALTY` |
| `models.py` | `DatasetRecord`, `ExplanationAgentStep`, `FixAgentStep`, `ExplanationResult`, `FixResult`, `EvaluationResult`, `RunResult`, `ConversationTurn` |
| `llm_client.py` | `LLMClient` (OpenAI), `GeminiClient`, `ChatResult` |
| `database_utils.py` | `get_db_path()`, `run_select_query()`, `get_ddl()`, `create_altered_sandbox()`, `destroy_sandbox()`, `compare_databases()`, `compute_structured_diff()`, `format_diff_as_text()` |
| `prompts.py` | `USER_AGENT_SYSTEM_PROMPT`, `EXPLANATION_AGENT_SYSTEM_PROMPT`, `FIX_AGENT_SYSTEM_PROMPT`, `JUDGE_SYSTEM_PROMPT` |
| `user_agent.py` | `UserAgent.respond()` |
| `explanation_agent.py` | `ExplanationAgent.run()` |
| `fix_agent.py` | `FixAgent.run(user_agent, retry_context=None)` |
| `evaluator.py` | `evaluate(...)`, `quick_fix_check(record, fix_result, ...)` |
| `runner.py` | `run_record(record, user_llm, explanation_llm, fix_llm, judge_llm, ...)` |
| `main.py` | CLI entry point |
