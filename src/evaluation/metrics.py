"""
Phase 4 - Metrics
Recall@k and Mean Reciprocal Rank (MRR), computed against a hand-labeled
query set where each query has one or more known-relevant chunk_ids.
"""
from typing import List, Set


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of relevant chunks that appear in the top-k retrieved results.

    Note: this is recall over the *labeled relevant set*, not over all
    possible relevant chunks in the corpus (we can't know that without
    labeling everything) -- standard practice for hand-labeled eval sets.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hit_count = len(top_k & relevant_ids)
    return hit_count / len(relevant_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """1 / rank of the first relevant chunk found; 0 if none found."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_query(retrieved_ids: List[str], relevant_ids: Set[str], k_values: List[int]) -> dict:
    """Compute Recall@k for each k in k_values, plus reciprocal rank, for one query."""
    return {
        **{f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
    }