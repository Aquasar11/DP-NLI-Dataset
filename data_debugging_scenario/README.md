# Data Debugging Dataset Generator

A pipeline for generating a **data debugging dataset** from the [BIRD-Bench](https://bird-bench.github.io/) text-to-SQL benchmark. Each generated instance pairs a correct SQL query with a deliberately corrupted database, enabling research on **provenance explanation** and **corrective action** in the context of text-to-SQL systems.

## Motivation

To the best of our knowledge, no existing dataset supports the combination of:
- Text-to-SQL (natural language question → SQL query)
- Provenance explanation (why is the output wrong?)
- Corrective action (how to fix the data?)

This tool constructs such a dataset by extending a seed text-to-SQL dataset (BIRD-Bench) with controlled data errors and follow-up questions.

## What Each Dataset Instance Contains

| Field | Description |
|---|---|
| `id` | Unique record identifier |
| `db_id` | BIRD database identifier (e.g., `movie_platform`) |
| `question` | Natural language question over the database |
| `evidence` | Hints/evidence clarifying domain terminology |
| `gold_sql` | The correct SQL query |
| `gold_result` | Query result on the original (correct) database |
| `alteration_type` | `delete` (remove entire row) or `modify` (change column values) |
| `targeted_records` | The specific result records that were targeted for removal |
| `altering_sql` | SQL statement(s) that corrupt the database |
| `altered_result` | Query result on the corrupted database |
| `alteration_explanation` | Why the altering SQL causes the targeted records to disappear |
| `follow_up_question` | A natural question a user would ask when seeing the wrong output |

## Dataset Construction Methodology

We follow the **data debugging scenario**: the SQL query is correct, but the database is modified such that the query result becomes incorrect.

For each seed sample from BIRD-Bench:

1. **Execute** the gold SQL query on the original database → obtain the correct result
2. **Select** target record(s) from the result (randomly chosen)
3. **Decide** alteration strategy: full-row deletion or column modification (configurable probability)
4. **Generate** an altering SQL via LLM (DELETE or UPDATE statement)
5. **Validate** in an isolated sandbox database copy — verify that *only* the targeted records are removed
6. **Retry** up to 3× with error feedback if validation fails
7. **Generate** a follow-up question via a second LLM call
8. **Save** the complete record; destroy the sandbox

Each instance maintains an isolated altered database state, reconstructable via the recorded `altering_sql`.

## Project Structure

```
input_debugging_scenario/
├── main.py              # CLI entry point (argparse)
├── config.py            # All configurable parameters
├── models.py            # Pydantic data models
├── db_manager.py        # SQLite operations, sandbox management
├── prompts.py           # LLM prompt templates (2-step)
├── llm_client.py        # OpenAI API wrapper with retry/backoff
├── validator.py         # Alteration validation logic
├── pipeline.py          # Main orchestration pipeline
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container image
├── docker-compose.yml   # Service configuration
├── .env.example         # API key template
├── .gitignore
├── data/
│   └── bird/
│       └── train/
│           ├── train.json           # ~9,428 BIRD samples
│           ├── train_tables.json    # Database schemas
│           ├── train_gold.sql       # Gold SQL queries
│           └── train_databases/     # 69 SQLite databases
│               ├── movie_platform/
│               ├── restaurant/
│               └── ...
├── output/              # Generated dataset (gitignored)
│   ├── dataset.json
│   ├── dataset.jsonl
│   └── pipeline_stats.json
└── sandbox/             # Ephemeral DB copies (gitignored)
```

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key (or compatible endpoint)
- The BIRD-Bench train dataset (already included in `data/bird/`)

### Local Installation

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=your-key-here
```

### Docker

```bash
# Build and run
cp .env.example .env
# Edit .env with your API key
docker compose up

# Or with custom args
docker compose run dataset-generator --samples 50 --model gpt-4o-mini
```

## Usage

### Quick Test (5 samples)

```bash
python main.py --samples 5 --model gpt-4o-mini
```

### Default Run (100 samples)

```bash
python main.py
```

### Full Dataset (~9,428 samples)

```bash
python main.py --samples 0
```

### All CLI Options

```
python main.py --help

  --provider {openai,gemini}   LLM provider (default: openai)
  --model MODEL                Model name (overrides OPENAI_MODEL / GEMINI_MODEL env var)
  --api-key API_KEY            API key (or set OPENAI_API_KEY / GEMINI_API_KEY env var)
  --base-url BASE_URL          OpenAI-compatible API base URL
  --temperature TEMP           LLM temperature (default: 0.3)
  --dataset {bird_train,bird_dev,spider_train,spider_test}
                               Source dataset — auto-sets --train-json and --db-dir defaults
                               (default: bird_train)
  --samples N                  Number of samples to process, 0 = all (default: 100)
  --delete-prob P              Probability of row deletion vs column modification (default: 0.5)
  --max-targets N              Max result records to alter per sample (default: 1)
  --max-retries N              Max LLM retries on validation failure (default: 3)
  --seed N                     Random seed for reproducibility (default: 42)
  --workers N                  Number of parallel worker threads (default: 10)
  --output-dir DIR             Output directory (default: output/)
  --train-json PATH            Path to dataset JSON (overrides --dataset default)
  --db-dir DIR                 Path to SQLite database folder (overrides --dataset default)
  --log-level LEVEL            DEBUG, INFO, WARNING, or ERROR (default: INFO)
```

### Dataset Examples

```bash
# BIRD train (default)
python main.py --samples 100

# BIRD dev (requires dev_databases.zip extracted to data/bird_dev/database/)
python main.py --dataset bird_dev --samples 50

# Spider train
python main.py --dataset spider_train --samples 100

# Spider test
python main.py --dataset spider_test --samples 50
```

## Pipeline Details

### Two-Step LLM Prompting

The pipeline uses two separate LLM calls per sample to avoid wasting tokens on invalid alterations:

1. **Step 1 — Alteration SQL Generation**: Given the gold SQL, database DDL, query result, targeted records, and alteration type, the LLM generates a DELETE/UPDATE statement and an explanation. This is validated in a sandbox before proceeding.

2. **Step 2 — Follow-up Q&A Generation**: Only after validation passes, the LLM generates a natural follow-up question.

### Sandbox Isolation

For each sample, the pipeline:
- Copies the original `.sqlite` file to a temporary sandbox
- Applies the LLM-generated altering SQL
- Re-executes the gold SQL to get the altered result
- Validates the result diff
- Destroys the sandbox (regardless of success/failure)

The original databases are **never modified**.

### Validation Rules

The validator checks:
1. **All targeted records are absent** from the altered result
2. **All non-targeted records remain present** in the altered result
3. **New records** that appear (e.g., due to `LIMIT` queries) are logged as warnings but tolerated

If validation fails, the pipeline builds a retry prompt that includes the previous attempt, the validation error, and the actual vs expected result — giving the LLM specific feedback to correct its SQL.

### Skipped Samples

The pipeline automatically skips samples that are not suitable for data debugging:
- **Empty results**: Queries returning 0 rows have no records to remove
- **Query execution errors**: Invalid SQL or missing databases
- **Scalar aggregates**: Queries like `SELECT COUNT(*) FROM ...` without `GROUP BY` return a single number — not meaningful for row-level data debugging

### Alteration Strategies

| Strategy | Probability | Description |
|---|---|---|
| **Delete** | 50% (configurable) | Remove the entire row from the database |
| **Modify** | 50% (configurable) | Change column values so the row no longer matches query conditions (set to NULL, default, or a value that breaks WHERE/JOIN) |

## Output Format

### dataset.json

```json
[
  {
    "id": 1,
    "db_id": "movie_platform",
    "question": "Name movie titles released in year 1945...",
    "evidence": "released in the year 1945 refers to movie_release_year = 1945",
    "gold_sql": "SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1",
    "gold_result": [{"movie_title": "Brief Encounter"}],
    "alteration_type": "delete",
    "targeted_records": [{"movie_title": "Brief Encounter"}],
    "altering_sql": "DELETE FROM movies WHERE movie_title = 'Brief Encounter'",
    "altered_result": [{"movie_title": "Children of Paradise"}],
    "alteration_explanation": "Deleting 'Brief Encounter' removes it from the 1945 movies, causing the next most popular movie to appear.",
    "follow_up_question": "Why does the result show 'Children of Paradise' instead of 'Brief Encounter'?",
  }
]
```

### dataset.jsonl

Same records, one JSON object per line.

### pipeline_stats.json

```json
{
  "total_samples": 100,
  "processed": 100,
  "success": 87,
  "skipped_empty_result": 5,
  "skipped_query_error": 3,
  "skipped_aggregate": 2,
  "failed_validation_after_retries": 2,
  "failed_llm_error": 1,
  "success_rate": "87.0%",
  "elapsed_seconds": 342.5
}
```

## Supported Datasets

The pipeline supports four source datasets selectable via `--dataset`:

| Dataset | Flag | JSON format | Key field |
|---|---|---|---|
| **BIRD-Bench train** (default) | `bird_train` | `train.json` | `SQL` |
| **BIRD-Bench dev** | `bird_dev` | `dev.json` | `SQL` |
| **Spider train** | `spider_train` | `train_spider.json` | `query` |
| **Spider test** | `spider_test` | `test.json` | `query` |

Spider samples are automatically normalized to the BIRD format (`query → SQL`, `evidence = ""`). All downstream pipeline logic is identical regardless of the source dataset.

> **BIRD dev databases**: The `database/` folder must be extracted from `dev_databases.zip` before use. If the directory does not exist, the pipeline exits with a clear error.

### Seed Dataset (BIRD train default)

The default BIRD-Bench training set:
- **~9,428 samples** across **69 SQLite databases**
- Each sample has: `db_id`, `question`, `evidence`, `SQL`
- Databases span domains: movies, restaurants, sports, finance, healthcare, etc.
- Queries range from simple `SELECT ... WHERE` to complex multi-table JOINs, subqueries, and conditional aggregations

## Configuration Reference

All defaults are set in `config.py` and can be overridden via CLI args or environment variables:

| Parameter | Default | Env Var | CLI Flag |
|---|---|---|---|
| OpenAI model | `gpt-4o` | `OPENAI_MODEL` | `--model` |
| API key | — | `OPENAI_API_KEY` | `--api-key` |
| API base URL | (OpenAI default) | `OPENAI_BASE_URL` | `--base-url` |
| Temperature | `0.3` | `OPENAI_TEMPERATURE` | `--temperature` |
| Dataset | `bird_train` | — | `--dataset` |
| Sample count | `100` | — | `--samples` |
| Delete probability | `0.5` | — | `--delete-prob` |
| Max target records | `1` | — | `--max-targets` |
| Max retries | `3` | — | `--max-retries` |
| Random seed | `42` | — | `--seed` |
| Workers | `10` | — | `--workers` |
| Log level | `INFO` | `LOG_LEVEL` | `--log-level` |

### Dataset Path Defaults

Default paths are set in `config.py` relative to the project root and can be overridden via environment variables:

| Dataset | JSON path | DB directory | Env var (db dir) |
|---|---|---|---|
| `bird_train` | `data/bird/train/train.json` | `data/bird/train/train_databases/` | `BIRD_TRAIN_DB_DIR` |
| `bird_dev` | `data/bird_dev/dev.json` | `data/bird_dev/database/` | `BIRD_DEV_DB_DIR` |
| `spider_train` | `data/spider_data/train_spider.json` | `data/spider_data/database/` | `SPIDER_TRAIN_DB_DIR` |
| `spider_test` | `data/spider_data/test.json` | `data/spider_data/test_database/` | `SPIDER_TEST_DB_DIR` |
