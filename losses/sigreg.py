"""
SIGReg: Signal Regularization via random projections and Gaussian matching.
Implements Epps-Pulley style test using random projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
import numpy as np


class SIGReg(nn.Module):
    """
    SIGReg regularization using random projections and Epps-Pulley test.

    Ensures that the distribution of embeddings matches a standard Gaussian
    along random projection directions.
    """

    def __init__(self, embedding_dim: int, num_projections: int = 32, projection_dim: int = 128):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_projections: Number of random projections (K)
            projection_dim: Dimension to project to
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_projections = num_projections
        self.projection_dim = projection_dim

        # Random projection matrices - fixed, not learned
        # Shape: [num_projections, projection_dim, embedding_dim]
        self.register_buffer(
            'projection_matrices',
            self._initialize_projections()
        )

    def _initialize_projections(self):
        """Initialize random projection matrices."""
        projections = torch.randn(
            self.num_projections,
            self.projection_dim,
            self.embedding_dim
        )
        # Normalize each projection matrix
        projections = F.normalize(projections, dim=-1)
        return projections

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss.

        Args:
            embeddings: [B, D] - batch of embeddings

        Returns:
            loss: scalar tensor
        """
        B, D = embeddings.shape
        assert D == self.embedding_dim, f"Expected dim {self.embedding_dim}, got {D}"

        # Project embeddings along random directions
        # embeddings: [B, D]
        # projection_matrices: [K, P, D]
        # Result: [K, P, B]
        projected = torch.einsum('kpd,bd->kpb', self.projection_matrices, embeddings)

        # Flatten projections: [K*P, B]
        projected = projected.reshape(-1, B)

        # Compute Epps-Pulley statistic for each projection
        losses = []
        for i in range(projected.shape[0]):
            proj_vals = projected[i]  # [B]
            loss = self._epps_pulley_loss(proj_vals)
            losses.append(loss)

        # Average over all projections
        return torch.stack(losses).mean()

    def _epps_pulley_loss(self, values: torch.Tensor) -> torch.Tensor:
        """
        Compute Epps-Pulley style loss comparing to standard Gaussian.

        Uses a differentiable approximation based on moment matching
        and distribution comparison.

        Args:
            values: [B] - projected values

        Returns:
            loss: scalar
        """
        # Moment matching: penalize deviation from N(0,1)
        mean = values.mean()
        std = values.std(unbiased=False) + 1e-6

        # Loss for mean and variance
        mean_loss = mean ** 2
        var_loss = (std - 1.0) ** 2

        # Normalize values to unit variance for higher-order moments
        normalized = (values - mean) / std

        # Higher-order moments (skewness and kurtosis penalties)
        # Skewness should be ~0, kurtosis should be ~3 for Gaussian
        moment3 = (normalized ** 3).mean()
        moment4 = (normalized ** 4).mean()

        skew_loss = moment3 ** 2
        kurt_loss = (moment4 - 3.0) ** 2

        # Combine losses
        loss = mean_loss + var_loss + 0.1 * skew_loss + 0.1 * kurt_loss

        return loss


def sigreg_loss(embeddings: torch.Tensor,
                num_projections: int = 32,
                projection_dim: int = 128) -> torch.Tensor:
    """
    Functional interface for SIGReg loss.

    Args:
        embeddings: [B, D] - batch of embeddings
        num_projections: Number of random projections
        projection_dim: Dimension of projections

    Returns:
        loss: scalar tensor
    """
    # Create temporary SIGReg module
    sigreg = SIGReg(
        embedding_dim=embeddings.shape[1],
        num_projections=num_projections,
        projection_dim=projection_dim
    ).to(embeddings.device)

    return sigreg(embeddings)


if __name__ == "__main__":
    # Test SIGReg
    torch.manual_seed(42)

    # Test with Gaussian embeddings (should have low loss)
    gaussian_embeds = torch.randn(64, 512)
    loss_gaussian = sigreg_loss(gaussian_embeds, num_projections=16)
    print(f"SIGReg loss for Gaussian embeddings: {loss_gaussian.item():.4f}")

    # Test with non-Gaussian embeddings (should have higher loss)
    non_gaussian_embeds = torch.randn(64, 512) * 3.0 + 2.0  # Wrong mean and variance
    loss_non_gaussian = sigreg_loss(non_gaussian_embeds, num_projections=16)
    print(f"SIGReg loss for non-Gaussian embeddings: {loss_non_gaussian.item():.4f}")
