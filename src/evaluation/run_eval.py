"""
Phase 4 - Evaluation runner
Runs every query in eval/queries.json through the Retriever, computes
Recall@k (for several k) and MRR, and writes a results table to
eval/results.json and eval/results.md.

Usage:
    python src/evaluation/run_eval.py
    python src/evaluation/run_eval.py --k-values 1 3 5 10
"""
import argparse
import json
import statistics
from pathlib import Path

from src.evaluation.metrics import evaluate_query
from src.indexing.search import Retriever

QUERIES_PATH = Path("eval/queries.json")
RESULTS_JSON_PATH = Path("eval/results.json")
RESULTS_MD_PATH = Path("eval/results.md")


def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def run_eval(k_values):
    queries = load_queries(QUERIES_PATH)
    if not queries:
        raise ValueError("eval/queries.json has no queries -- add some hand-labeled queries first.")

    retriever = Retriever()
    max_k = max(k_values)

    per_query_results = []
    for item in queries:
        query_text = item["query"]
        relevant_ids = set(item["relevant_chunk_ids"])

        hits = retriever.search(query_text, top_k=max_k)
        retrieved_ids = [h["chunk_id"] for h in hits]

        metrics = evaluate_query(retrieved_ids, relevant_ids, k_values)
        per_query_results.append({
            "query": query_text,
            "relevant_chunk_ids": list(relevant_ids),
            "retrieved_chunk_ids": retrieved_ids,
            **metrics,
        })

    # Aggregate: mean of each metric across all queries
    aggregate = {}
    for k in k_values:
        key = f"recall@{k}"
        aggregate[key] = statistics.mean(r[key] for r in per_query_results)
    aggregate["mrr"] = statistics.mean(r["reciprocal_rank"] for r in per_query_results)
    aggregate["num_queries"] = len(per_query_results)

    return aggregate, per_query_results


def write_markdown_table(aggregate, k_values, path):
    lines = ["# Evaluation Results\n"]
    lines.append(f"Evaluated on {aggregate['num_queries']} hand-labeled queries.\n")
    lines.append("| Metric | Score |")
    lines.append("|---|---|")
    for k in k_values:
        lines.append(f"| Recall@{k} | {aggregate[f'recall@{k}']:.3f} |")
    lines.append(f"| MRR | {aggregate['mrr']:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10])
    args = parser.parse_args()

    aggregate, per_query_results = run_eval(args.k_values)

    print("=== Aggregate Results ===")
    for k in args.k_values:
        print(f"Recall@{k}: {aggregate[f'recall@{k}']:.3f}")
    print(f"MRR: {aggregate['mrr']:.3f}")
    print(f"(across {aggregate['num_queries']} queries)")

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "per_query": per_query_results}, f, indent=2)
    write_markdown_table(aggregate, args.k_values, RESULTS_MD_PATH)

    print(f"\nSaved detailed results to {RESULTS_JSON_PATH}")
    print(f"Saved README-ready table to {RESULTS_MD_PATH}")


if __name__ == "__main__":
    main()