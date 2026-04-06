"""
JEPA Loss: Normalized MSE between predicted and target embeddings.
"""

import torch
import torch.nn.functional as F


def jepa_loss(pred_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized MSE loss for JEPA.

    The loss compares L2-normalized embeddings:
        z_pred_norm = z_pred / ||z_pred||
        z_tgt_norm = z_tgt / ||z_tgt||
        loss = ||z_pred_norm - z_tgt_norm||^2

    Args:
        pred_embeddings: [B, D] - predicted embeddings
        target_embeddings: [B, D] - target embeddings

    Returns:
        loss: scalar tensor
    """
    # Normalize embeddings to unit sphere
    pred_norm = F.normalize(pred_embeddings, p=2, dim=-1)  # [B, D]
    target_norm = F.normalize(target_embeddings, p=2, dim=-1)  # [B, D]

    # Compute MSE on normalized embeddings
    loss = F.mse_loss(pred_norm, target_norm)

    return loss


if __name__ == "__main__":
    # Test JEPA loss
    torch.manual_seed(42)

    # Test with identical embeddings (should have ~0 loss)
    embeds = torch.randn(32, 512)
    loss_identical = jepa_loss(embeds, embeds)
    print(f"JEPA loss for identical embeddings: {loss_identical.item():.6f}")

    # Test with random embeddings (should have higher loss)
    pred = torch.randn(32, 512)
    target = torch.randn(32, 512)
    loss_random = jepa_loss(pred, target)
    print(f"JEPA loss for random embeddings: {loss_random.item():.4f}")

    # Test with opposite embeddings (should have max loss ~4)
    target = torch.randn(32, 512)
    pred = -target
    loss_opposite = jepa_loss(pred, target)
    print(f"JEPA loss for opposite embeddings: {loss_opposite.item():.4f}")
