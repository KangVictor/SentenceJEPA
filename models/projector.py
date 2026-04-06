"""
Projection head (MLP) for mapping contextualized embeddings to final space.
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    MLP projection head.

    Projects contextualized embeddings to a lower-dimensional space
    for contrastive learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings.

        Args:
            x: [B, D_in] or [B, S, D_in]

        Returns:
            projected: [B, D_out] or [B, S, D_out]
        """
        return self.net(x)


if __name__ == "__main__":
    # Test projection head
    print("Testing ProjectionHead...")

    projector = ProjectionHead(
        input_dim=768,
        hidden_dim=1024,
        output_dim=512,
    )

    # Test 2D input
    x_2d = torch.randn(16, 768)
    out_2d = projector(x_2d)
    print(f"2D input shape: {x_2d.shape}")
    print(f"2D output shape: {out_2d.shape}")
    assert out_2d.shape == (16, 512)

    # Test 3D input
    x_3d = torch.randn(4, 5, 768)
    out_3d = projector(x_3d)
    print(f"3D input shape: {x_3d.shape}")
    print(f"3D output shape: {out_3d.shape}")
    assert out_3d.shape == (4, 5, 512)

    print("✓ ProjectionHead test passed!")
