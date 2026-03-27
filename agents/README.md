# Multi-Agent Data Debugging Framework

A three-agent system that simulates a structured forensic investigation into unexplained database query result changes. Given a dataset record produced by the [data debugging pipeline](../data_debugging_scenario/README.md), three LLM-powered agents collaborate to identify a hidden data alteration and repair the database.

---

## Overview

The framework models a real-world scenario in which a database query that previously returned a known result suddenly returns something different. Three agents work together to uncover the root cause and fix it:

| Agent | Role | Input | Output |
|---|---|---|---|
| **UserAgent** | Database owner / oracle | Full alteration context; access to both original and altered DBs | Answers questions; runs SQL on either database |
| **ExplanationAgent** | Investigator | `follow_up_question`, `gold_sql`, `altered_result`, schema | Explanation + root cause (discovered independently) |
| **FixAgent** | Repair engineer | Explanation from `ExplanationAgent` | SQL to restore the database to its original state |

An **Evaluator** then:
- Uses an **LLM judge** to score the `ExplanationAgent`'s explanation against the ground truth.
- Applies the `FixAgent`'s SQL to a sandbox and **compares the result database to the original** to verify the fix is complete.

---

## How It Works

```
DatasetRecord
      │
      ▼
┌─────────────┐   creates altered sandbox + diffs vs original
│  UserAgent  │◄──────────────────────────────────────────────┐
│  (oracle)   │   has access to BOTH original & altered DBs   │
└──────┬──────┘                                               │
       │ responds to targeted questions only                   │
       │                                                       │
       ▼                                                       │
┌──────────────────┐  run_query (direct DB) / ask_question / done
│ ExplanationAgent │──────────────────────────► ExplanationResult
│  (investigator)  │                            (explanation + root_cause)
└──────────────────┘                                          │
                                                              │
       ┌──────────────────────────────────────────────────────┘
       │ explanation + root_cause passed in
       ▼
┌──────────┐  ask_question (penalized) / done
│ FixAgent │──────────────────────────────────► FixResult
│ (repair) │                                    (fix_sql to restore DB)
└──────────┘
       │
       ▼
┌───────────┐
│ Evaluator │──► EvaluationResult
└───────────┘    (explanation_score, db_match, final_score)
```

### Agent Interaction Protocol

All agents communicate via structured JSON responses.

**UserAgent** (inner ReAct tool-use loop):
```json
// Query the original (pre-alteration) database:
{"action": "run_query_original", "sql": "SELECT ...", "reasoning": "..."}

// Query the current (altered) database:
{"action": "run_query_altered", "sql": "SELECT ...", "reasoning": "..."}

// Give a final answer:
{"action": "respond", "answer": "..."}
```

**ExplanationAgent** (investigation loop — three possible actions):
```json
// Run a SELECT directly on the altered database:
{"action": "run_query", "sql": "SELECT ..."}

// Ask the database owner a question:
{"action": "ask_question", "question": "..."}

// Conclude with an explanation:
{"action": "done", "explanation": "...", "root_cause": "..."}
```

**FixAgent** (repair loop):
```json
// Ask a clarifying question (incurs penalty):
{"action": "ask_question", "question": "..."}

// Submit the fix:
{"action": "done", "fix_sql": "INSERT INTO ...", "confidence": 0.9, "reasoning": "..."}
```

---

## Directory Structure

```
agents/
├── main.py              # CLI entry point
├── runner.py            # Per-record orchestration of the three-agent pipeline
├── config.py            # Configuration (env vars, per-agent configs, paths)
├── models.py            # Pydantic data models
├── llm_client.py        # LLM client wrappers (OpenAI + Gemini)
├── prompts.py           # System prompt templates for all agents + judge
├── user_agent.py        # UserAgent implementation
├── explanation_agent.py # ExplanationAgent implementation
├── fix_agent.py         # FixAgent implementation
├── database_utils.py    # SQLite utilities (sandbox, query execution, DB comparison)
├── evaluator.py         # Scoring: LLM judge + database comparison
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

```
base_score  = 1.0  if db_match else 0.0
final_score = max(0.0, base_score − question_penalty × questions_asked)
```

**`db_match`** — After the `FixAgent`'s `fix_sql` is applied to a fresh copy of the altered database, every table in the result is compared row-by-row (order-insensitive) against the original database. `db_match` is `True` only if all tables are identical.

**Question penalty** — every question the `FixAgent` asks costs `question_penalty` (default `0.05`) from the final score. The agent is incentivized to reason from the `ExplanationAgent`'s output before asking questions.

### Explanation Score

The `ExplanationAgent`'s explanation is scored by a separate **judge LLM** (0.0–1.0) that compares it against the ground-truth alteration information. The judge evaluates whether the correct table, change type, and affected rows were identified.

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
| `--question-penalty` | `0.05` | Score penalty per `FixAgent` question |
| `--dataset` | *(from config)* | Path to `dataset.json` |
| `--db-dir` | *(from config)* | Root directory of SQLite database folders |
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

**Custom dataset and output directory:**
```bash
python main.py \
  --dataset /data/my_dataset.json \
  --db-dir /data/my_databases \
  --output-dir ./my_run_output \
  --samples 100
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
      "explanation": "The part 'pink powder drab lawn cyan' was deleted from the 'part' table, causing it to disappear from the query result.",
      "root_cause": "A row was deleted from the 'part' table.",
      "turns_used": 3,
      "conversation": [
        {"role": "investigator", "content": "Does the part 'pink powder...' still exist?"},
        {"role": "user", "content": "No, that record is no longer in the database."}
      ]
    },
    "fix": {
      "record_id": 1,
      "fix_sql": "INSERT INTO part (p_partkey, p_name, ...) VALUES (42, 'pink powder drab lawn cyan', ...);",
      "confidence": 0.92,
      "reasoning": "The explanation confirms the row was deleted. I asked for the original values and reconstructed the INSERT.",
      "questions_asked": 1,
      "conversation": [
        {"role": "investigator", "content": "What were the original column values for that part?"},
        {"role": "user", "content": "The row had p_partkey=42, p_name='pink powder drab lawn cyan', ..."}
      ]
    },
    "evaluation": {
      "record_id": 1,
      "db_id": "retails",
      "explanation_score": 0.95,
      "explanation_reasoning": "The investigator correctly identified the deleted table and row.",
      "db_match": true,
      "db_diff": "",
      "questions_asked": 1,
      "question_penalty": 0.05,
      "base_score": 1.0,
      "final_score": 0.95,
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
  "db_match_count": 14,
  "db_match_rate": "70.0%",
  "avg_explanation_score": 0.812,
  "avg_final_score": 0.634,
  "avg_questions_asked": 1.15
}
```

---

## Agent Details

### UserAgent

- **Initialization**: Creates a temporary SQLite sandbox (copy of original DB with `altering_sql` applied). Also retains a path to the unaltered original DB and pre-computes the diff between them.
- **Inner ReAct loop**: The LLM may issue `run_query_original` or `run_query_altered` actions (SELECT only — writes are blocked) before providing a `respond` action. Up to 3 inner iterations per question.
- **Information policy**: Knows the full diff between original and altered databases but only reveals information when directly asked. Never reveals the literal DML SQL.
- **Shared history**: The same conversation history is shared across `ExplanationAgent` and `FixAgent` sessions, ensuring consistent answers.
- **Cleanup**: `user_agent.cleanup()` destroys the sandbox. Called in a `finally` block to guarantee cleanup even on error.
- **Public property**: `sandbox_path` exposes the altered sandbox path for `ExplanationAgent` direct queries.

### ExplanationAgent

- **Goal**: Independently discover what changed in the database — no ground-truth information is provided.
- **Direct DB access**: Can execute SELECT queries directly on the altered database via the `run_query` action, without going through the UserAgent.
- **Oracle access**: Can also ask the UserAgent targeted questions via `ask_question` when DB inspection alone is insufficient (e.g. "what was the original value?").
- **Turn limit**: Configurable via `--max-explanation-turns` (default: 6). Counts both `run_query` and `ask_question` actions.
- **Output**: `ExplanationResult` with `explanation`, `root_cause`, `turns_used`, and the full `conversation` log.

### FixAgent

- **Goal**: Produce SQL that, when applied to the altered database, fully restores it to its original state.
- **Inputs**: `ExplanationResult` (explanation + root_cause), database schema (DDL), and query context.
- **Oracle access**: Can ask the UserAgent clarifying questions (e.g. original column values). Each question incurs a `question_penalty` deduction.
- **Turn limit**: Configurable via `--max-fix-turns` (default: 4).
- **Output**: `FixResult` with `fix_sql`, `confidence`, `reasoning`, and `questions_asked`.

### Evaluator

**Explanation evaluation (LLM-as-judge)**:
- A judge LLM receives the ground-truth alteration info (`altering_sql`, `alteration_explanation`, `targeted_records`) alongside the agent's `explanation` and `root_cause`.
- The judge returns a score (0.0–1.0) and a reasoning string.
- Judge LLM is independently configurable and defaults to the UserAgent's LLM settings.

**Fix evaluation (database comparison)**:
- A fresh sandbox is created: original DB → apply `altering_sql` → apply `fix_sql`.
- Every table in the result is compared against the original database (row-by-row, order-insensitive using set equality).
- `db_match = True` only if all tables are identical — no partial credit.

**Score formula**:
```
base_score  = 1.0  if db_match else 0.0
final_score = max(0.0, base_score − question_penalty × questions_asked)
```

---

## Module Reference

| Module | Key symbols |
|---|---|
| `config.py` | `AgentLLMConfig`, `USER_AGENT_CONFIG`, `EXPLANATION_AGENT_CONFIG`, `FIX_AGENT_CONFIG`, `JUDGE_CONFIG`, `MAX_EXPLANATION_TURNS`, `MAX_FIX_TURNS`, `QUESTION_PENALTY` |
| `models.py` | `DatasetRecord`, `ExplanationAgentStep`, `FixAgentStep`, `UserAgentStep`, `ExplanationResult`, `FixResult`, `EvaluationResult`, `RunResult`, `ConversationTurn` |
| `llm_client.py` | `LLMClient` (OpenAI), `GeminiClient`, `ChatResult` |
| `database_utils.py` | `get_db_path()`, `run_select_query()`, `get_ddl()`, `create_altered_sandbox()`, `destroy_sandbox()`, `compare_databases()` |
| `prompts.py` | `USER_AGENT_SYSTEM_PROMPT`, `EXPLANATION_AGENT_SYSTEM_PROMPT`, `FIX_AGENT_SYSTEM_PROMPT`, `JUDGE_SYSTEM_PROMPT` |
| `user_agent.py` | `UserAgent.respond()`, `UserAgent.cleanup()`, `UserAgent.sandbox_path` |
| `explanation_agent.py` | `ExplanationAgent.run(user_agent)` |
| `fix_agent.py` | `FixAgent.run(user_agent)` |
| `evaluator.py` | `evaluate(record, fix_result, explanation_result, judge_llm, ...)` |
| `runner.py` | `run_record(record, user_llm, explanation_llm, fix_llm, judge_llm, ...)` |
| `main.py` | CLI entry point |
