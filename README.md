# DP-NLI-Dataset

A research framework for studying **data debugging in text-to-SQL systems**. The project provides (1) a pipeline for generating a benchmark dataset of controlled database corruptions paired with natural language questions, and (2) a multi-agent evaluation framework where LLM agents must independently identify and repair the corruption.

This repository is the implementation accompanying the paper:

> **[Paper title placeholder]**
> Authors, Venue, Year
> [[Paper]](#) · [[Dataset]](#)

---

## Overview

The framework is organized as a two-phase pipeline:

```
Phase 1 — Dataset Generation
─────────────────────────────
BIRD-Bench / Spider samples
         │
         ▼
  Execute gold SQL → select target rows
         │
         ▼
  LLM generates DELETE / UPDATE
         │
         ▼
  Validate in sandbox (retry up to 3×)
         │
         ▼
  LLM generates follow-up question
         │
         ▼
  dataset.json  ◄── labelled, self-contained records


Phase 2 — Multi-Agent Evaluation
─────────────────────────────────
   dataset.json
         │
         ▼
  ┌─────────────┐
  │  UserAgent  │ (oracle — knows the diff, no SQL)
  └──────┬──────┘
         │ answers questions
         ▼
  ┌──────────────────┐   run_query (direct DB)
  │ ExplanationAgent │ ─────────────────────────► ExplanationResult
  │  (investigator)  │                            (what changed + why)
  └──────────────────┘
         │ explanation passed down
         ▼
  ┌──────────┐   ask_question (penalized) + run_query
  │ FixAgent │ ────────────────────────────────────► FixResult
  │ (repair) │                                       (fix SQL)
  └──────────┘
         │
         ▼
  ┌───────────┐
  │ Evaluator │ ──► EvaluationResult
  └───────────┘     (explanation score + fix score + final score)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (or Gemini / Vertex AI credentials)
- BIRD-Bench train dataset (place under `data_debugging_scenario/data/bird/train/`)

### Installation

```bash
# Clone the repository
git clone https://github.com/Aquasar11/DP-NLI-Dataset.git
cd DP-NLI-Dataset

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API credentials
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY (and optionally Gemini / GCP variables)
```

---

## Phase 1: Dataset Generation

The dataset generator corrupts BIRD-Bench or Spider databases with deliberate `DELETE`/`UPDATE` statements, validates each alteration in an isolated sandbox, then generates a natural follow-up question a confused user would ask.

### Run

```bash
cd data_debugging_scenario

# Quick test (5 samples)
python main.py --samples 5 --model gpt-4o-mini

# Full BIRD-Bench train set
python main.py --samples 0 --model gpt-4o

# Spider dataset
python main.py --dataset spider_train --samples 100

# Docker
docker compose up
```

### Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--samples N` | `100` | Records to generate (`0` = all ~9,428) |
| `--provider` | `openai` | `openai` or `gemini` |
| `--model` | `gpt-4o` | LLM model name |
| `--delete-prob P` | `0.5` | Probability of DELETE vs UPDATE |
| `--workers N` | `10` | Parallel worker threads |
| `--dataset` | `bird_train` | Source: `bird_train`, `bird_dev`, `spider_train`, `spider_test` |
| `--output-dir DIR` | `output/` | Destination directory |

### Output schema (dataset.json)

Each generated record contains:

| Field | Description |
|---|---|
| `id` | Unique integer ID |
| `db_id` | SQLite database name |
| `question` | Natural language question |
| `evidence` | Optional domain hint |
| `gold_sql` | Correct SQL query |
| `gold_result` | Query result on the original database |
| `alteration_type` | `delete` or `modify` |
| `targeted_records` | Rows targeted by the alteration |
| `target_columns` | Columns changed (`["all"]` for DELETE) |
| `altering_sql` | **Ground-truth DML** (hidden from agents during evaluation) |
| `altered_result` | Query result after corruption |
| `alteration_explanation` | Why the alteration changes the query result |
| `follow_up_question` | Question a confused user would ask |
| `is_aggregation` | `true` if the query is a scalar aggregate (e.g. `COUNT(*)`) |

See [`data_debugging_scenario/README.md`](data_debugging_scenario/README.md) for full details.

---

## Phase 2: Multi-Agent Evaluation

The evaluation framework runs three LLM agents on each dataset record:

- **ExplanationAgent** — autonomously investigates the altered database using SQL queries and produces a natural language explanation of what changed and why.
- **FixAgent** — receives the explanation and generates SQL to restore the database, optionally asking the UserAgent clarifying questions (penalized).
- **Evaluator** — scores the explanation with an LLM judge and verifies the fix by re-running the gold SQL on the repaired database.

### Run

```bash
cd agents

# Default run (all records, single worker)
python main.py

# With concurrency and Gemini
python main.py --provider gemini --use-vertexai --samples 10 --workers 5

# Debug a single record
python main.py --samples 1 --log-level DEBUG
```

### Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--samples N` | `0` (all) | Records to evaluate |
| `--workers N` | `1` | Parallel workers |
| `--provider` | `openai` | Global LLM provider |
| `--model` | *(config)* | Global model name |
| `--max-explanation-turns` | `6` | Turn budget for ExplanationAgent |
| `--max-fix-turns` | `4` | Turn budget for FixAgent |
| `--ask-question-penalty` | `0.05` | Score penalty per UserAgent question |
| `--dataset` | *(config)* | Path to `dataset.json` |
| `--db-dir` | *(config)* | Path to SQLite database folder |
| `--output-dir` | `./output` | Output directory |

### Scoring

```
tool_penalty    = explanation_query_penalty × explanation_query_turns
                + fix_query_penalty         × fix_query_turns
                + ask_question_penalty      × questions_asked

retry_multiplier = 0.5  if fix required a retry, else 1.0

final_score     = max(0.0, gold_result_score − tool_penalty) × retry_multiplier
```

Each `results.json` entry includes the full sample context alongside agent outputs and scores — no need to join against the original dataset file.

See [`agents/README.md`](agents/README.md) for full details.

---

## Repository Structure

```
DP-NLI-Dataset/
├── data_debugging_scenario/   # Phase 1: dataset generation pipeline
│   ├── main.py                # CLI entry point
│   ├── pipeline.py            # Orchestration loop
│   ├── models.py              # Pydantic data models
│   ├── validator.py           # Alteration validation + aggregate detection
│   ├── db_manager.py          # SQLite sandbox management
│   ├── llm_client.py          # OpenAI / Gemini API wrapper
│   ├── prompts.py             # LLM prompt templates
│   ├── config.py              # Configuration
│   └── output/                # Generated dataset (gitignored)
├── agents/                    # Phase 2: multi-agent evaluation framework
│   ├── main.py                # CLI entry point
│   ├── runner.py              # Per-record orchestration
│   ├── explanation_agent.py   # Autonomous investigation agent
│   ├── fix_agent.py           # Database repair agent
│   ├── user_agent.py          # Oracle agent (text-based diff)
│   ├── evaluator.py           # LLM judge + fix scoring
│   ├── models.py              # Pydantic data models
│   ├── prompts.py             # System prompt templates
│   ├── llm_client.py          # LLM client wrappers
│   ├── database_utils.py      # SQLite utilities and diff computation
│   ├── config.py              # Configuration
│   └── output/                # Agent results (gitignored)
├── query_debugging_scenario/  # Query-level (not data-level) corruption utilities
├── requirements.txt           # Shared Python dependencies
└── .env.example               # Credential template
```

---

## Citation

If you use this dataset or framework in your research, please cite:

```bibtex
@article{placeholder,
  title   = {[Paper title placeholder]},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {[Year]}
}
```

---

## License

[License placeholder]
