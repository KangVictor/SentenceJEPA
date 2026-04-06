"""
Evaluation functions for Sentence JEPA.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List
from tqdm import tqdm

from .metrics import compute_recall, compute_mean_reciprocal_rank


@torch.no_grad()
def evaluate_retrieval(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k_values: List[int] = [1, 5, 10],
    max_batches: int = None,
) -> Dict[str, float]:
    """
    Evaluate model on masked sentence retrieval task.

    For each paragraph:
    - Mask one sentence
    - Use predictor to predict masked sentence embedding
    - Use target to get actual embeddings of all sentences
    - Compute retrieval accuracy (Recall@K)

    Args:
        model: HierarchicalSentenceJEPA model
        dataloader: DataLoader yielding batches
        device: Device to run on
        k_values: List of K values for Recall@K
        max_batches: Maximum batches to evaluate (None = all)

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()

    all_pred_embeddings = []
    all_target_embeddings = []

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if max_batches is not None and i >= max_batches:
            break

        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentence_mask = batch['sentence_mask'].to(device)
        mask_idx = batch['mask_idx'].to(device)

        # Forward pass
        z_pred, z_target = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentence_mask=sentence_mask,
            mask_idx=mask_idx,
        )

        all_pred_embeddings.append(z_pred.cpu())
        all_target_embeddings.append(z_target.cpu())

    # Concatenate all embeddings
    pred_embeddings = torch.cat(all_pred_embeddings, dim=0)  # [N, D]
    target_embeddings = torch.cat(all_target_embeddings, dim=0)  # [N, D]

    # Compute retrieval metrics
    metrics = compute_recall(pred_embeddings, target_embeddings, k_values=k_values)

    # Also compute MRR
    mrr = compute_mean_reciprocal_rank(pred_embeddings, target_embeddings)
    metrics['mrr'] = mrr

    return metrics


if __name__ == "__main__":
    print("Evaluation module ready!")
