"""
Comprehensive analysis of agent results across models and datasets.

Metric legend (column headings)
--------------------------------
  AltType  avg_alteration_type_score  — Did the agent correctly identify the
               alteration type? (insertion / deletion / modification)  [0 or 1]
  Explain  avg_explanation_score      — LLM-judge score for the ExplanationAgent's
               root-cause explanation vs. ground truth  [0.0 / 0.5 / 1.0]
  GoldRes  avg_gold_result_score      — Does running the gold SQL on the
               repaired DB return the expected result?  [0 or 1]
  FullRst  avg_full_restore_score     — Were ALL corrupted rows fully restored
               to their original values?  [0 or 1]
  Final    avg_final_score            — Composite score (weighted combination
               of the four above, minus per-tool-call penalties)  [0–1]
  N        number of records in that bucket

Breakdowns:
  1. Overall (per model × dataset)
  2. Database column-count buckets
  3. Database row-count buckets
  4. Gold SQL AST depth buckets
  5. Rows affected by altering_sql buckets
  6. Number of clarification questions asked by the FixAgent
  7. Number of turns used by the ExplanationAgent
  8. Total length of UserAgent responses to the FixAgent

Usage:
    python analyze_results.py
"""

from __future__ import annotations

import json
import sqlite3
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import sqlglot
import sqlglot.expressions as exp

# ── Paths ─────────────────────────────────────────────────────────────────────
AGENTS_DIR = Path(__file__).resolve().parent
DATA_ROOT = AGENTS_DIR.parent / "data_debugging_scenario"

RUNS: dict[str, dict[str, Any]] = {
    "bird_gemini":  {
        "results": AGENTS_DIR / "bird_output" / "results.json",
        "db_dir":  DATA_ROOT / "data" / "dev" / "dev_databases",
        "dataset": "bird",
    },
    "bird_claude":  {
        "results": AGENTS_DIR / "bird_output_claude" / "results.json",
        "db_dir":  DATA_ROOT / "data" / "dev" / "dev_databases",
        "dataset": "bird",
    },
    "spider_gemini": {
        "results": AGENTS_DIR / "spider_output" / "results.json",
        "db_dir":  DATA_ROOT / "data" / "spider_data" / "spider_data" / "test_databases",
        "dataset": "spider",
    },
    "spider_claude": {
        "results": AGENTS_DIR / "spider_output_claude" / "results.json",
        "db_dir":  DATA_ROOT / "data" / "spider_data" / "spider_data" / "test_databases",
        "dataset": "spider",
    },
    "bird_claude_dataset_gemini": {
        "results": AGENTS_DIR / "bird_output_claude_dataset_gemini" / "results.json",
        "db_dir":  DATA_ROOT / "data" / "dev" / "dev_databases",
        "dataset": "bird",
    },
    "bird_claude_dataset_claude": {
        "results": AGENTS_DIR / "bird_output_claude_claude" / "results.json",
        "db_dir":  DATA_ROOT / "data" / "dev" / "dev_databases",
        "dataset": "bird",
    },
}

METRICS = [
    "alteration_type_score",
    "explanation_score",
    "gold_result_score",
    "full_restore_score",
    "final_score",
]

# ── DB info cache ─────────────────────────────────────────────────────────────

_db_info_cache: dict[tuple[str, str], dict] = {}


def get_db_info(db_id: str, db_dir: Path) -> dict:
    key = (db_id, str(db_dir))
    if key in _db_info_cache:
        return _db_info_cache[key]

    # Try both <db_id>/<db_id>.sqlite and direct <db_id>.sqlite
    candidates = [
        db_dir / db_id / f"{db_id}.sqlite",
        db_dir / f"{db_id}.sqlite",
    ]
    db_path = next((p for p in candidates if p.exists()), None)

    if db_path is None:
        info = {"total_columns": None, "total_rows": None}
        _db_info_cache[key] = info
        return info

    try:
        con = sqlite3.connect(str(db_path))
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        total_cols = 0
        total_rows = 0
        for (tname,) in tables:
            cols = con.execute(f"PRAGMA table_info('{tname}')").fetchall()
            total_cols += len(cols)
            try:
                count = con.execute(f"SELECT COUNT(*) FROM \"{tname}\"").fetchone()[0]
                total_rows += count
            except Exception:
                pass
        con.close()
        info = {"total_columns": total_cols, "total_rows": total_rows}
    except Exception:
        info = {"total_columns": None, "total_rows": None}

    _db_info_cache[key] = info
    return info


# ── AST depth ─────────────────────────────────────────────────────────────────

def ast_depth(sql: str) -> int:
    try:
        tree = sqlglot.parse_one(sql)
        if tree is None:
            return 0

        def _depth(node) -> int:
            children = list(node.args.values())
            child_nodes = []
            for c in children:
                if isinstance(c, exp.Expression):
                    child_nodes.append(c)
                elif isinstance(c, list):
                    child_nodes.extend(x for x in c if isinstance(x, exp.Expression))
            if not child_nodes:
                return 1
            return 1 + max(_depth(ch) for ch in child_nodes)

        return _depth(tree)
    except Exception:
        return 0


# ── Bucketing helpers ─────────────────────────────────────────────────────────

def bucket_columns(n: int | None) -> str:
    if n is None:
        return "unknown"
    if n <= 10:
        return "1-10 cols"
    if n <= 30:
        return "11-30 cols"
    if n <= 60:
        return "31-60 cols"
    return "61+ cols"


def bucket_rows(n: int | None) -> str:
    if n is None:
        return "unknown"
    if n <= 500:
        return "1-500 rows"
    if n <= 5_000:
        return "501-5k rows"
    if n <= 50_000:
        return "5k-50k rows"
    return "50k+ rows"


def bucket_ast(d: int) -> str:
    if d == 0:
        return "unknown"
    if d <= 5:
        return "depth 1-5"
    if d <= 10:
        return "depth 6-10"
    if d <= 15:
        return "depth 11-15"
    return "depth 16+"


def bucket_affected(n: int) -> str:
    if n == 0:
        return "0 rows affected"
    if n == 1:
        return "1 row affected"
    if n <= 5:
        return "2-5 rows affected"
    if n <= 20:
        return "6-20 rows affected"
    return "21+ rows affected"


def bucket_questions(n: int) -> str:
    """Number of clarification questions the FixAgent asked the UserAgent."""
    if n == 0:
        return "0 questions (no clarification)"
    if n == 1:
        return "1 question"
    return "2+ questions"


def bucket_expl_turns(n: int) -> str:
    """Number of run_query turns the ExplanationAgent used."""
    if n == 0:
        return "0 turns (fallback)"
    if n <= 2:
        return "1-2 turns"
    if n <= 4:
        return "3-4 turns"
    return "5-6 turns"


def bucket_user_response_len(chars: int) -> str:
    """Total character length of all UserAgent responses to the FixAgent."""
    if chars == 0:
        return "0 chars (no questions asked)"
    if chars <= 150:
        return "1-150 chars (brief)"
    if chars <= 400:
        return "151-400 chars (moderate)"
    if chars <= 800:
        return "401-800 chars (detailed)"
    return "800+ chars (very detailed)"


# ── Aggregation ───────────────────────────────────────────────────────────────

def empty_agg() -> dict:
    return {"n": 0, **{m: 0.0 for m in METRICS}}


def add_record(agg: dict, ev: dict) -> None:
    agg["n"] += 1
    for m in METRICS:
        agg[m] += ev.get(m, 0.0)


def finalise(agg: dict) -> dict:
    n = agg["n"]
    if n == 0:
        return {**agg, **{f"avg_{m}": None for m in METRICS}}
    return {
        "n": n,
        **{f"avg_{m}": round(agg[m] / n, 4) for m in METRICS},
    }


# ── Reporting helpers ─────────────────────────────────────────────────────────

def print_table(
    title: str,
    breakdown: dict[str, dict[str, dict]],
    bucket_order: list[str] | None = None,
) -> None:
    """breakdown[run_name][bucket] = finalised dict"""
    runs = list(breakdown.keys())
    all_buckets: list[str] = []
    for b in (bucket_order or []):
        if b in {bk for rd in breakdown.values() for bk in rd}:
            all_buckets.append(b)
    seen = set(all_buckets)
    for rd in breakdown.values():
        for bk in rd:
            if bk not in seen:
                all_buckets.append(bk)
                seen.add(bk)

    col_w = 22
    run_w = 10

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    metric_labels = [
        ("avg_alteration_type_score", "AltType"),
        ("avg_explanation_score",     "Explain"),
        ("avg_gold_result_score",     "GoldRes"),
        ("avg_full_restore_score",    "FullRst"),
        ("avg_final_score",           "Final  "),
    ]

    # Header
    header = f"{'Bucket':<{col_w}}" + "".join(
        f"{'Model':<{run_w}}" + "".join(f"{lbl:>9}" for _, lbl in metric_labels) + "  N  "
        for _ in [None]
    )
    # One row per bucket × run
    print(f"{'Bucket':<{col_w}} {'Model':<20}" + "".join(f"{lbl:>9}" for _, lbl in metric_labels) + "   N")
    print("-" * 80)

    for bk in all_buckets:
        first = True
        for run in runs:
            d = breakdown[run].get(bk)
            if d is None:
                continue
            prefix = f"{bk:<{col_w}}" if first else " " * col_w
            first = False
            vals = "".join(
                f"{d.get(mk, 0):9.4f}" if d.get(mk) is not None else f"{'N/A':>9}"
                for mk, _ in metric_labels
            )
            print(f"{prefix} {run:<20}{vals}  {d['n']:>4}")
        if not first:
            print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Accumulators: [run_name][bucket] -> agg dict
    overall: dict[str, dict] = {}
    by_cols: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(empty_agg))
    by_rows: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(empty_agg))
    by_ast:  dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(empty_agg))
    by_affected: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(empty_agg))
    by_questions:   dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(empty_agg))
    by_expl_turns:  dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(empty_agg))
    by_user_resp:   dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(empty_agg))

    for run_name, cfg in RUNS.items():
        print(f"Loading {run_name} ...")
        with open(cfg["results"], encoding="utf-8") as f:
            records = json.load(f)

        run_agg = empty_agg()
        db_dir: Path = cfg["db_dir"]

        for rec in records:
            ev = rec["evaluation"]
            db_id = rec["db_id"]

            # ── overall ────────────────────────────────────────────────────
            add_record(run_agg, ev)

            # ── db info ────────────────────────────────────────────────────
            info = get_db_info(db_id, db_dir)
            bc = bucket_columns(info["total_columns"])
            br = bucket_rows(info["total_rows"])
            add_record(by_cols[run_name][bc], ev)
            add_record(by_rows[run_name][br], ev)

            # ── AST depth ──────────────────────────────────────────────────
            depth = ast_depth(rec.get("gold_sql", ""))
            ba = bucket_ast(depth)
            add_record(by_ast[run_name][ba], ev)

            # ── rows affected ──────────────────────────────────────────────
            altered = rec.get("altered_result", [])
            # altered_result is a list of result rows from the gold query after alteration
            # The number of affected rows = |altered_result| - |gold_result|
            gold = rec.get("gold_result", [])

            # For aggregation queries the result won't directly tell us affected rows,
            # so we approximate via the diff in single-value results
            # or count the number of VALUES clauses in altering_sql
            altering_sql = rec.get("altering_sql", "")
            # Count semicolon-separated statements as a proxy for affected rows
            stmts = [s.strip() for s in altering_sql.split(";") if s.strip()]
            n_affected = len(stmts) if stmts else 0

            # Better: for INSERT/DELETE/UPDATE the affected count is inferrable
            # from the difference in COUNT(*)-style results — but only when the
            # result values are whole integers (not floats/ratios/percentages).
            if (
                len(altered) == 1
                and len(gold) == 1
                and len(altered[0]) == 1
                and len(gold[0]) == 1
            ):
                vals = list(altered[0].values())
                gvals = list(gold[0].values())
                try:
                    av, gv = vals[0], gvals[0]
                    # Only use the diff when both values are whole numbers
                    if isinstance(av, (int, float)) and isinstance(gv, (int, float)):
                        if float(av) == int(av) and float(gv) == int(gv):
                            diff = abs(int(av) - int(gv))
                            if diff > 0:
                                n_affected = diff
                except (TypeError, ValueError):
                    pass  # keep the stmt-count fallback

            baf = bucket_affected(n_affected)
            add_record(by_affected[run_name][baf], ev)

            # ── clarification questions asked by FixAgent ──────────────────
            n_questions = rec["fix"].get("questions_asked", 0)
            bq = bucket_questions(n_questions)
            add_record(by_questions[run_name][bq], ev)

            # ── ExplanationAgent turns ─────────────────────────────────────
            expl_turns = rec["explanation"].get("turns_used", 0)
            bet = bucket_expl_turns(expl_turns)
            add_record(by_expl_turns[run_name][bet], ev)

            # ── UserAgent response length (to FixAgent questions) ──────────
            fix_conv = rec["fix"].get("conversation", [])
            user_resp_chars = sum(
                len(m.get("content", ""))
                for m in fix_conv
                if m.get("role") == "UserAgent"
            )
            bur = bucket_user_response_len(user_resp_chars)
            add_record(by_user_resp[run_name][bur], ev)

        overall[run_name] = finalise(run_agg)

    # ── Print overall ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  OVERALL RESULTS")
    print(f"{'='*80}")
    metric_labels = [
        ("avg_alteration_type_score", "AltType"),
        ("avg_explanation_score",     "Explain"),
        ("avg_gold_result_score",     "GoldRes"),
        ("avg_full_restore_score",    "FullRst"),
        ("avg_final_score",           "Final  "),
    ]
    print(f"{'Model':<22}" + "".join(f"{lbl:>9}" for _, lbl in metric_labels) + "   N")
    print("-" * 80)
    for run_name, d in overall.items():
        vals = "".join(
            f"{d.get(mk, 0):9.4f}" if d.get(mk) is not None else f"{'N/A':>9}"
            for mk, _ in metric_labels
        )
        print(f"{run_name:<22}{vals}  {d['n']:>4}")

    # ── Finalise bucketed dicts ────────────────────────────────────────────────
    def finalise_breakdown(bd: dict) -> dict:
        return {run: {bk: finalise(agg) for bk, agg in buckets.items()} for run, buckets in bd.items()}

    cols_final = finalise_breakdown(by_cols)
    rows_final = finalise_breakdown(by_rows)
    ast_final  = finalise_breakdown(by_ast)
    aff_final  = finalise_breakdown(by_affected)

    col_order = ["1-10 cols", "11-30 cols", "31-60 cols", "61+ cols", "unknown"]
    row_order = ["1-500 rows", "501-5k rows", "5k-50k rows", "50k+ rows", "unknown"]
    ast_order = ["depth 1-5", "depth 6-10", "depth 11-15", "depth 16+", "unknown"]
    aff_order = ["0 rows affected", "1 row affected", "2-5 rows affected",
                 "6-20 rows affected", "21+ rows affected"]

    questions_final  = finalise_breakdown(by_questions)
    expl_turns_final = finalise_breakdown(by_expl_turns)
    user_resp_final  = finalise_breakdown(by_user_resp)

    q_order  = ["0 questions (no clarification)", "1 question", "2+ questions"]
    et_order = ["0 turns (fallback)", "1-2 turns", "3-4 turns", "5-6 turns"]
    ur_order = [
        "0 chars (no questions asked)",
        "1-150 chars (brief)",
        "151-400 chars (moderate)",
        "401-800 chars (detailed)",
        "800+ chars (very detailed)",
    ]

    print_table("BY DATABASE COLUMN COUNT", cols_final, col_order)
    print_table("BY DATABASE ROW COUNT",    rows_final, row_order)
    print_table("BY GOLD SQL AST DEPTH",    ast_final,  ast_order)
    print_table("BY ROWS AFFECTED BY ALTERING SQL", aff_final, aff_order)
    print_table(
        "BY NUMBER OF CLARIFICATION QUESTIONS (FixAgent → UserAgent)\n"
        "  0 questions = FixAgent proceeded without asking anything\n"
        "  1+ questions = FixAgent asked the UserAgent for clarification",
        questions_final, q_order,
    )
    print_table(
        "BY EXPLANATION AGENT TURNS (run_query calls)\n"
        "  0 turns = fallback (max turns hit with no valid response)\n"
        "  1-6 turns = autonomous investigation depth",
        expl_turns_final, et_order,
    )
    print_table(
        "BY USER RESPONSE LENGTH TO FIX AGENT (total chars across all answers)\n"
        "  0 chars = FixAgent asked no questions\n"
        "  Higher = UserAgent provided more detailed / longer answers",
        user_resp_final, ur_order,
    )


if __name__ == "__main__":
    main()
