"""
Evaluation metrics for Sentence JEPA.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict


def compute_recall(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute Recall@K for retrieval task.

    Given query embeddings and candidate embeddings, compute the
    fraction of queries where the correct candidate (same index)
    appears in the top K retrieved candidates.

    Args:
        query_embeddings: [N, D] - query embeddings
        candidate_embeddings: [N, D] - candidate embeddings (correct match at same index)
        k_values: List of K values to compute recall for

    Returns:
        metrics: Dictionary with 'recall@K' keys
    """
    N, D = query_embeddings.shape
    assert candidate_embeddings.shape == (N, D), "Shape mismatch"

    # Normalize embeddings
    query_norm = F.normalize(query_embeddings, p=2, dim=-1)  # [N, D]
    candidate_norm = F.normalize(candidate_embeddings, p=2, dim=-1)  # [N, D]

    # Compute similarity matrix: [N, N]
    # similarity[i, j] = cosine similarity between query i and candidate j
    similarity = torch.mm(query_norm, candidate_norm.t())  # [N, N]

    # For each query, the correct candidate is at the same index (diagonal)
    # Get ranking of candidates for each query
    # Higher similarity = better match
    # argsort returns indices in ascending order, so we reverse
    sorted_indices = torch.argsort(similarity, dim=-1, descending=True)  # [N, N]

    # Check if correct candidate (index i) is in top K for query i
    metrics = {}
    for k in k_values:
        if k > N:
            k = N

        # Get top K candidates for each query
        top_k_indices = sorted_indices[:, :k]  # [N, K]

        # Check if correct index is in top K
        # Correct index for query i is i (diagonal)
        correct_indices = torch.arange(N, device=query_embeddings.device).unsqueeze(1)  # [N, 1]
        matches = (top_k_indices == correct_indices).any(dim=1)  # [N]

        # Compute recall
        recall = matches.float().mean().item()
        metrics[f'recall@{k}'] = recall

    return metrics


def compute_mean_reciprocal_rank(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        query_embeddings: [N, D] - query embeddings
        candidate_embeddings: [N, D] - candidate embeddings

    Returns:
        mrr: Mean reciprocal rank
    """
    N, D = query_embeddings.shape

    # Normalize embeddings
    query_norm = F.normalize(query_embeddings, p=2, dim=-1)
    candidate_norm = F.normalize(candidate_embeddings, p=2, dim=-1)

    # Compute similarity matrix
    similarity = torch.mm(query_norm, candidate_norm.t())  # [N, N]

    # Get ranking
    sorted_indices = torch.argsort(similarity, dim=-1, descending=True)  # [N, N]

    # Find rank of correct candidate for each query
    correct_indices = torch.arange(N, device=query_embeddings.device)
    ranks = []
    for i in range(N):
        rank = (sorted_indices[i] == correct_indices[i]).nonzero(as_tuple=True)[0].item()
        ranks.append(1.0 / (rank + 1))  # Rank is 0-indexed, so add 1

    mrr = sum(ranks) / len(ranks)
    return mrr


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    # Create dummy embeddings
    N, D = 10, 128
    query = torch.randn(N, D)
    candidates = torch.randn(N, D)

    # Make some queries similar to their correct candidates
    candidates[:3] = query[:3] + torch.randn(3, D) * 0.1

    # Compute recall
    metrics = compute_recall(query, candidates, k_values=[1, 3, 5])
    print(f"Recall metrics: {metrics}")

    # Compute MRR
    mrr = compute_mean_reciprocal_rank(query, candidates)
    print(f"MRR: {mrr:.4f}")

    # Test perfect case (identical)
    metrics_perfect = compute_recall(query, query, k_values=[1, 5])
    print(f"Perfect case recall: {metrics_perfect}")
    assert metrics_perfect['recall@1'] == 1.0

    print("\n✓ Metrics test passed!")
