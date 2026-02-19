import argparse
import pandas as pd
import re
import os
import json

from src.prompts.prompt_loader import load_prompt
from src.llms.llm_engine import call_model
from src.database_utils.database_manager import get_db_schema_db_id
from src.database_utils.execution import compare_sqls, syntax_check_sql
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv(override=True)

DB_PATH = os.getenv("DB_PATH", "data/dev/dev_databases")
DATASET_PATH = os.getenv("DATASET_PATH", "data/dev/dev.json")
# Global schema cache: db_id -> schema string
SCHEMA_CACHE: dict = {}


def extract_sql_query(text: str) -> str:
    """Extract the last SQL block from a given text, remove all newlines, and collapse multiple spaces."""
    matches = re.findall(r"```sql\s*(.*?)\s*```", text, re.DOTALL)
    sql = matches[-1] if matches else text
    sql = sql.replace('\n', ' ').strip()
    sql = re.sub(r'\s+', ' ', sql)
    return sql

def load_all_schemas(df: pd.DataFrame) -> dict:
    """Preload all schemas for all unique db_ids in the dataset."""
    global SCHEMA_CACHE
    unique_db_ids = df["db_id"].unique()
    print(f"Preloading schemas for {len(unique_db_ids)} databases...")
    
    for db_id in tqdm(unique_db_ids, desc="Loading schemas"):
        try:
            schema = get_db_schema_db_id(
                db_id=db_id,
                bird_database_path=DB_PATH,
                use_fk_constraints=True
            )
            SCHEMA_CACHE[db_id] = schema
        except Exception as e:
            print(f"Error loading schema for db_id '{db_id}': {e}")
            SCHEMA_CACHE[db_id] = ""
    
    return SCHEMA_CACHE


def find_single_wrong_candidate(sample: dict, args: argparse.Namespace, prompt_template: str) -> dict | None:
    """
    For a single sample, try models in order strong -> middle -> weak.
    For each model, generate args.num_generation candidates and return the first incorrect query.
    Return None if all generated candidates across all models are correct.
    """
    db_id = sample["db_id"]
    question = sample["question"]
    evidence = sample.get("evidence", "")
    gold_query = sample["SQL"]
    question_id = sample.get("question_id", "unknown")

    base_schema = SCHEMA_CACHE.get(db_id, "")
    if not base_schema:
        print(f"Warning: No schema found for db_id '{db_id}'")
        return None

    prompt = prompt_template.format(
        DATABASE_SCHEMA=base_schema,
        QUESTION=question,
        HINT=evidence
    )

    models_in_order = [
        args.strong_model_name,
        args.middle_model_name,
        args.weak_model_name,
    ]

    def run_one_candidate(model_name: str, candidate_idx: int) -> dict | None:
        try:
            response = call_model(
                model_name,
                prompt,
                args.temperature,
                args.max_output_tokens
            )
            predicted_query = extract_sql_query(response)

            is_valid_syntax = syntax_check_sql(DB_PATH, db_id, predicted_query)
            exec_res = compare_sqls(DB_PATH, db_id, predicted_query, gold_query).get("exec_res", 0)

            if exec_res != 1:
                label = "valid_but_incorrect" if is_valid_syntax else "invalid_syntax"
                return {
                    "question_id": question_id,
                    "db_id": db_id,
                    "question": question,
                    "evidence": evidence,
                    "gold_query": gold_query,
                    "wrong_query": predicted_query,
                    "source_model": model_name,
                    "candidate_idx": candidate_idx,
                    "label": label,
                }
        except Exception as err:
            print(
                f"Generation/eval failed for question_id '{question_id}' "
                f"using model '{model_name}', candidate {candidate_idx}: {err}"
            )
        return None

    for model_name in models_in_order:
        candidate_workers = min(args.num_generation, args.num_candidate_workers)
        incorrect_results = []

        with ThreadPoolExecutor(max_workers=candidate_workers) as candidate_executor:
            futures = {
                candidate_executor.submit(run_one_candidate, model_name, candidate_idx): candidate_idx
                for candidate_idx in range(args.num_generation)
            }

            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    incorrect_results.append(res)

        if incorrect_results:
            incorrect_results.sort(key=lambda x: x["candidate_idx"])
            return incorrect_results[0]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strong_model_name", default="gemini-3-flash-preview")
    parser.add_argument("--middle_model_name", default="gemini-2.5-flash-lite")
    parser.add_argument("--weak_model_name", default="gemini-2.0-flash")
    parser.add_argument("--output_path", default="debugging_dataset_wrong_queries.json")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_output_tokens", type=int, default=8000)
    parser.add_argument("--prompt_name", default="sql_generation_zero_shot_new")
    parser.add_argument("--num_workers", type=int, default=50)
    parser.add_argument("--num_generation", type=int, default=5)
    parser.add_argument(
        "--num_candidate_workers",
        type=int,
        default=5,
        help="Max threads per sample for parallel candidate generation within each model"
    )
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds for each sample processing")
    args = parser.parse_args()

    df = pd.read_json(DATASET_PATH)
    # df = df.head(5)  # For quick testing; remove or adjust as needed
    
    # Step 1: Preload all schemas once
    load_all_schemas(df)
    
    # Load prompt template
    prompt_template = load_prompt(args.prompt_name)
    
    # Step 2: Prepare samples
    samples = df.to_dict(orient="records")
    for idx, sample in enumerate(samples):
        if "question_id" not in sample:
            sample["question_id"] = idx

    print(
        f"Processing {len(samples)} samples with staged fallback "
        f"(strong -> middle -> weak), up to {args.num_generation} candidates/model"
    )

    # Step 3: Process each sample and keep the first wrong query if found
    wrong_query_records = []
    easy_count = 0
    processed_count = 0
    save_interval = 500

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for sample in samples:
            future = executor.submit(find_single_wrong_candidate, sample, args, prompt_template)
            futures[future] = sample["question_id"]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            try:
                res = fut.result(timeout=args.timeout)
                if res is None:
                    easy_count += 1
                else:
                    wrong_query_records.append(res)
            except Exception as e:
                qid = futures[fut]
                print(f"\nError processing question_id '{qid}': {e}")

            processed_count += 1

            # Save intermediate results every save_interval items
            if processed_count % save_interval == 0:
                print(
                    f"\nSaving intermediate results at "
                    f"{processed_count}/{len(futures)} processed..."
                )
                intermediate_path = args.output_path.replace('.json', f'_intermediate_{processed_count}.json')
                with open(intermediate_path, "w") as f:
                    json.dump(wrong_query_records, f, indent=2)
                print(f"Saved to {intermediate_path}")

    print(
        f"\nProcessing complete: {len(wrong_query_records)} hard samples found, "
        f"{easy_count} easy samples"
    )

    with open(args.output_path, "w") as f:
        json.dump(wrong_query_records, f, indent=2)

if __name__ == "__main__":
    main()
