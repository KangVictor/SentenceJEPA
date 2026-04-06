"""
Paragraph-level transformer that processes sentence embeddings.
"""

import torch
import torch.nn as nn
import math


class ParagraphTransformer(nn.Module):
    """
    Transformer encoder that operates on sentence-level embeddings.

    Takes sentence embeddings and produces contextualized sentence representations.
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_sentences: int = 100,
    ):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            max_sentences: Maximum number of sentences (for positional embeddings)
        """
        super().__init__()
        self.d_model = d_model
        self.max_sentences = max_sentences

        # Positional embeddings for sentences
        self.pos_embedding = nn.Parameter(torch.randn(1, max_sentences, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        sentence_embeddings: torch.Tensor,
        sentence_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Process sentence embeddings through transformer.

        Args:
            sentence_embeddings: [B, S, D] - sentence embeddings
            sentence_mask: [B, S] - binary mask (1 = valid, 0 = padding)

        Returns:
            contextualized: [B, S, D] - contextualized embeddings
        """
        B, S, D = sentence_embeddings.shape
        assert S <= self.max_sentences, f"Too many sentences: {S} > {self.max_sentences}"

        # Add positional embeddings
        x = sentence_embeddings + self.pos_embedding[:, :S, :]  # [B, S, D]

        # Create attention mask for transformer
        # PyTorch transformer expects mask where True = ignore
        if sentence_mask is not None:
            # Convert from [B, S] where 1=valid to [B, S] where True=invalid
            attn_mask = (sentence_mask == 0)  # [B, S]
            # Expand for attention: [B, S] -> [B*nhead, S, S]
            # For simplicity, use src_key_padding_mask which expects [B, S]
        else:
            attn_mask = None

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # [B, S, D]

        # Final norm
        x = self.norm(x)

        return x


if __name__ == "__main__":
    # Test paragraph transformer
    print("Testing ParagraphTransformer...")

    model = ParagraphTransformer(
        d_model=768,
        nhead=8,
        num_layers=2,
        dim_feedforward=2048,
    )

    # Create dummy input
    B, S, D = 4, 5, 768
    sentence_embeddings = torch.randn(B, S, D)
    sentence_mask = torch.ones(B, S)
    sentence_mask[0, 4] = 0  # Mask last sentence of first batch

    # Forward pass
    output = model(sentence_embeddings, sentence_mask)
    print(f"Input shape: {sentence_embeddings.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (B, S, D)
    print("✓ ParagraphTransformer test passed!")
