# Multi-Agent Data Debugging Framework

A three-agent system that simulates a structured forensic investigation into unexplained database query result changes. Given a dataset record produced by the [data debugging pipeline](../data_debugging_scenario/README.md), three LLM-powered agents collaborate (and compete) to identify a hidden data alteration.

---

## Overview

The framework models a real-world scenario in which a database query that previously returned a known result suddenly returns something different. Three agents work together to uncover the root cause:

| Agent | Role | Input | Output |
|---|---|---|---|
| **UserAgent** | Database owner / oracle | Full alteration context; access to the post-alteration DB | Answers questions; runs SQL queries |
| **ExplanationAgent** | Investigator | `follow_up_question`, `gold_sql`, `altered_result` | Human-readable explanation + root cause |
| **AnswerAgent** | SQL Detective | Explanation from `ExplanationAgent` | Predicted `altering_sql` DML statement |

An **Evaluator** then scores the `AnswerAgent`'s prediction using exact SQL matching and semantic equivalence.

---

## How It Works

```
DatasetRecord
      │
      ▼
┌─────────────┐  creates altered sandbox
│  UserAgent  │◄──────────────────────────────────────────────┐
│  (oracle)   │                                               │
└──────┬──────┘                                               │
       │ responds to questions                                 │
       │                                                       │
       ▼                                                       │
┌──────────────────┐  ask_question / done                     │
│ ExplanationAgent │──────────────────────────► ExplanationResult
│  (investigator)  │                            (explanation + root_cause)
└──────────────────┘                                          │
                                                              │
       ┌──────────────────────────────────────────────────────┘
       │ explanation + root_cause passed in
       ▼
┌─────────────┐  ask_question (penalized) / done
│ AnswerAgent │──────────────────────────────────► AnswerResult
│ (detective) │                                    (predicted_altering_sql)
└─────────────┘
       │
       ▼
┌───────────┐
│ Evaluator │──► EvaluationResult (exact_match, semantic_match, final_score)
└───────────┘
```

### Agent Interaction Protocol

All agents communicate via structured JSON responses:

**UserAgent** (inner ReAct tool-use loop):
```json
// To run a SELECT query before answering:
{"action": "run_query", "sql": "SELECT ...", "reasoning": "..."}

// To give a final answer:
{"action": "respond", "answer": "..."}
```

**ExplanationAgent** and **AnswerAgent** (outer Q&A loop):
```json
// To ask a question:
{"action": "ask_question", "question": "..."}

// To conclude:
{"action": "done", "explanation": "...", "root_cause": "..."}
// or for AnswerAgent:
{"action": "done", "predicted_altering_sql": "...", "confidence": 0.9, "reasoning": "..."}
```

---

## Directory Structure

```
agents/
├── main.py             # CLI entry point
├── runner.py           # Per-record orchestration of the three-agent pipeline
├── config.py           # Configuration (env vars, defaults, paths)
├── models.py           # Pydantic data models
├── llm_client.py       # LLM client wrappers (OpenAI + Gemini)
├── prompts.py          # System prompt templates for all three agents
├── user_agent.py       # UserAgent implementation
├── explanation_agent.py # ExplanationAgent implementation
├── answer_agent.py     # AnswerAgent implementation
├── database_utils.py   # SQLite utilities (sandbox management, query execution)
├── evaluator.py        # Scoring and evaluation logic
└── output/             # Generated at runtime
    ├── results.json    # Full RunResult for every processed record
    ├── stats.json      # Aggregate accuracy metrics
    └── sandbox/        # Temporary sandbox databases (auto-cleaned up)
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

Each record receives a `final_score` computed as:

```
base_score = 1.0  if exact_match OR semantic_match else 0.0
final_score = max(0.0, base_score − question_penalty × questions_asked)
```

**Exact match** — the predicted SQL is identical to `altering_sql` after lowercasing and whitespace normalization.

**Semantic match** — the predicted SQL is applied to a fresh copy of the original database; then `gold_sql` is run on the result. If the output equals `altered_result`, the prediction is semantically equivalent even if the SQL text differs.

**Question penalty** — every question the `AnswerAgent` asks costs `question_penalty` (default `0.05`) from the final score. The agent should reason from the `ExplanationAgent`'s output as much as possible before asking further questions.

---

## Setup

### Prerequisites

- Python 3.10+
- Packages from the parent project's environment (shared with `data_debugging_scenario`):

```bash
pip install pydantic python-dotenv google-genai openai tqdm
```

### Environment Variables

Copy the shared `.env` file at the repository root or set the following directly:

```bash
# GCP / Gemini (Vertex AI)
GCP_PROJECT=your-gcp-project-id
GCP_REGION=global                   # or us-central1 etc.
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=true      # set to use Vertex AI

# Gemini direct API (alternative, without Vertex AI)
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3-flash-preview

# OpenAI
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o

# Agent tuning (optional overrides)
MAX_EXPLANATION_TURNS=6
MAX_ANSWER_TURNS=4
QUESTION_PENALTY=0.05

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

### Options

| Flag | Default | Description |
|---|---|---|
| `--provider` | `openai` | LLM provider: `openai` or `gemini` |
| `--model` | *(from config)* | Override the model name |
| `--api-key` | *(from .env)* | Override the API key |
| `--base-url` | `None` | OpenAI-compatible base URL (OpenAI only) |
| `--use-vertexai` | `False` | Route Gemini calls through Vertex AI |
| `--temperature` | *(from config)* | LLM sampling temperature |
| `--samples` | `0` (all) | Number of dataset records to process |
| `--workers` | `1` | Parallel workers (ThreadPoolExecutor) |
| `--max-explanation-turns` | `6` | Max Q&A turns for the `ExplanationAgent` |
| `--max-answer-turns` | `4` | Max Q&A turns for the `AnswerAgent` |
| `--question-penalty` | `0.05` | Score deducted per `AnswerAgent` question |
| `--dataset` | *(from config)* | Path to `dataset.json` |
| `--db-dir` | *(from config)* | Root directory of SQLite database folders |
| `--output-dir` | `./output` | Directory for results and stats files |
| `--log-level` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Examples

**Run 10 records with Gemini via Vertex AI, 5 workers:**
```bash
python main.py \
  --provider gemini \
  --use-vertexai \
  --samples 10 \
  --workers 5
```

**Run on a custom dataset with OpenAI GPT-4o:**
```bash
python main.py \
  --provider openai \
  --model gpt-4o \
  --dataset /data/my_dataset.json \
  --db-dir /data/my_databases \
  --samples 50 \
  --workers 10 \
  --output-dir ./my_run_output
```

**Full debug run (1 record, verbose logging):**
```bash
python main.py \
  --provider gemini \
  --use-vertexai \
  --samples 1 \
  --log-level DEBUG
```

---

## Output

After a run, two files are written to the `--output-dir`:

### `results.json`

A JSON array of `RunResult` objects, one per processed record:

```json
[
  {
    "record_id": 1,
    "db_id": "retails",
    "explanation": {
      "record_id": 1,
      "explanation": "The part 'pink powder drab lawn cyan' was removed ...",
      "root_cause": "A row was deleted from the 'part' table.",
      "turns_used": 2,
      "conversation": [
        {"role": "investigator", "content": "Does the part 'pink powder...' still exist?"},
        {"role": "user", "content": "No, that record is no longer in the database."}
      ]
    },
    "answer": {
      "record_id": 1,
      "predicted_altering_sql": "DELETE FROM part WHERE p_name = 'pink powder drab lawn cyan';",
      "confidence": 0.95,
      "reasoning": "The explanation confirms the row was deleted ...",
      "questions_asked": 0,
      "conversation": []
    },
    "evaluation": {
      "record_id": 1,
      "db_id": "retails",
      "ground_truth_sql": "DELETE FROM part WHERE p_name = 'pink powder drab lawn cyan';",
      "predicted_sql": "DELETE FROM part WHERE p_name = 'pink powder drab lawn cyan';",
      "exact_match": true,
      "semantic_match": false,
      "questions_asked": 0,
      "question_penalty": 0.0,
      "base_score": 1.0,
      "final_score": 1.0,
      "error": null
    }
  }
]
```

### `stats.json`

Aggregate metrics across all processed records:

```json
{
  "total_records": 10,
  "exact_match": 6,
  "semantic_match": 2,
  "correct_total": 8,
  "exact_match_rate": "60.0%",
  "semantic_match_rate": "20.0%",
  "accuracy": "80.0%",
  "avg_final_score": 0.8725,
  "avg_questions_asked": 0.60
}
```

---

## Agent Details

### UserAgent

- **Initialization**: Creates a temporary SQLite sandbox by copying the original DB and applying `altering_sql` to it.
- **Inner ReAct loop**: When answering a question, the LLM may issue `run_query` actions (SELECT only — write operations are blocked) before providing a `respond` action. Up to 3 inner iterations per question.
- **Shared history**: The same conversation history is used for both `ExplanationAgent` and `AnswerAgent` sessions, ensuring consistent answers throughout a single record.
- **Constraint**: Must not reveal the literal DML SQL that caused the alteration; may only describe observable effects.
- **Cleanup**: `user_agent.cleanup()` destroys the sandbox database. This is called in a `finally` block to guarantee cleanup even on error.

### ExplanationAgent

- **Inputs at construction**: `follow_up_question`, `gold_sql`, `gold_result`, `altered_result`.
- **Loop**: Alternates between `ask_question` (fed back into its own LLM context) and receiving answers from `UserAgent`. Concludes with a `done` action.
- **Turn limit**: Configurable via `--max-explanation-turns` (default: 6). If exhausted without a `done`, a fallback explanation is returned.
- **Output**: `ExplanationResult` with `explanation`, `root_cause`, `turns_used`, and the full `conversation` log.

### AnswerAgent

- **Inputs at construction**: Everything the `ExplanationAgent` receives, plus `ExplanationResult` (explanation + root_cause).
- **Loop**: Same `ask_question` / `done` protocol. Each `ask_question` action increments an internal `questions_asked` counter.
- **Turn limit**: Configurable via `--max-answer-turns` (default: 4). Falls back to a placeholder SQL comment if exhausted.
- **Output**: `AnswerResult` with `predicted_altering_sql`, `confidence`, `reasoning`, and `questions_asked`.

### Evaluator

- **Exact match**: `predicted_altering_sql` equals `altering_sql` after lowercasing, semicolon removal, and whitespace collapsing.
- **Semantic match**: Only tested when exact match fails. Applies `predicted_altering_sql` to a fresh sandbox, executes `gold_sql` on it, and compares the result to `record.altered_result`. The sandbox is destroyed immediately after.
- **Score formula**: `final_score = max(0.0, base_score − question_penalty × questions_asked)`

---

## Module Reference

| Module | Key symbols |
|---|---|
| `config.py` | `MAX_EXPLANATION_TURNS`, `MAX_ANSWER_TURNS`, `QUESTION_PENALTY`, `DB_BASE_DIR`, `DATASET_PATH`, `OUTPUT_DIR`, `SANDBOX_DIR` |
| `models.py` | `DatasetRecord`, `ExplanationAgentStep`, `AnswerAgentStep`, `UserAgentStep`, `ExplanationResult`, `AnswerResult`, `EvaluationResult`, `RunResult`, `ConversationTurn` |
| `llm_client.py` | `LLMClient` (OpenAI), `GeminiClient`, `ChatResult` |
| `database_utils.py` | `get_db_path()`, `run_select_query()`, `get_ddl()`, `create_altered_sandbox()`, `destroy_sandbox()` |
| `prompts.py` | `USER_AGENT_SYSTEM_PROMPT`, `EXPLANATION_AGENT_SYSTEM_PROMPT`, `ANSWER_AGENT_SYSTEM_PROMPT` |
| `user_agent.py` | `UserAgent.respond()`, `UserAgent.cleanup()` |
| `explanation_agent.py` | `ExplanationAgent.run(user_agent)` |
| `answer_agent.py` | `AnswerAgent.run(user_agent)` |
| `evaluator.py` | `evaluate(record, answer, ...)` |
| `runner.py` | `run_record(record, llm, ...)` |
| `main.py` | CLI entry point |
