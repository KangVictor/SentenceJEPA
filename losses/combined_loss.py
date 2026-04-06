"""
Combined loss for Sentence JEPA: JEPA loss + SIGReg.
"""

import torch
from .jepa_loss import jepa_loss
from .sigreg import SIGReg


def combined_loss(pred_embeddings: torch.Tensor,
                  target_embeddings: torch.Tensor,
                  sigreg_module: SIGReg,
                  lambda_sigreg: float = 0.1) -> tuple[torch.Tensor, dict]:
    """
    Compute combined loss: JEPA + SIGReg.

    loss = loss_jepa + λ * loss_sigreg

    Args:
        pred_embeddings: [B, D] - predicted embeddings
        target_embeddings: [B, D] - target embeddings
        sigreg_module: SIGReg module for computing regularization
        lambda_sigreg: Weight for SIGReg loss

    Returns:
        total_loss: scalar tensor
        loss_dict: dictionary with individual losses for logging
    """
    # JEPA loss (normalized MSE)
    loss_jepa = jepa_loss(pred_embeddings, target_embeddings)

    # SIGReg loss (apply to target embeddings)
    loss_sigreg = sigreg_module(target_embeddings)

    # Combined loss
    total_loss = loss_jepa + lambda_sigreg * loss_sigreg

    # Return loss dict for logging
    loss_dict = {
        'total': total_loss.item(),
        'jepa': loss_jepa.item(),
        'sigreg': loss_sigreg.item(),
    }

    return total_loss, loss_dict
