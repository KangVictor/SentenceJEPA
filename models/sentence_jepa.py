"""
Main Hierarchical Sentence JEPA model.

Combines sentence encoder, paragraph transformer, and projection heads
to implement masked sentence prediction in latent space.
"""

import torch
import torch.nn as nn
from typing import Optional

from .sentence_encoder import SentenceEncoder
from .paragraph_transformer import ParagraphTransformer
from .projector import ProjectionHead


class HierarchicalSentenceJEPA(nn.Module):
    """
    Hierarchical Sentence JEPA model.

    Architecture:
    1. Encode sentences with pretrained encoder
    2. Process with paragraph transformer (two branches)
       - Target branch: full paragraph
       - Predictor branch: one sentence masked
    3. Project to latent space with MLPs
    4. Compare predicted vs target for masked sentence
    """

    def __init__(
        self,
        # Sentence encoder config
        sentence_encoder_name: str = "roberta-base",
        sentence_encoder_frozen: bool = True,
        sentence_pooling: str = "mean",
        # Paragraph transformer config
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_sentences: int = 100,
        # Projection head config
        projection_hidden_dim: int = 1024,
        projection_output_dim: int = 512,
        projection_dropout: float = 0.1,
    ):
        """
        Args:
            sentence_encoder_name: HuggingFace model name
            sentence_encoder_frozen: Whether to freeze sentence encoder
            sentence_pooling: Pooling strategy for sentence encoder
            d_model: Paragraph transformer dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            max_sentences: Max sentences for positional embeddings
            projection_hidden_dim: Hidden dim of projection MLP
            projection_output_dim: Output dim of projection MLP
            projection_dropout: Dropout in projection MLP
        """
        super().__init__()

        # 1. Sentence encoder
        self.sentence_encoder = SentenceEncoder(
            model_name=sentence_encoder_name,
            frozen=sentence_encoder_frozen,
            pooling=sentence_pooling,
        )

        encoder_dim = self.sentence_encoder.get_embedding_dim()

        # Optional linear projection if encoder dim != d_model
        if encoder_dim != d_model:
            self.input_projection = nn.Linear(encoder_dim, d_model)
        else:
            self.input_projection = nn.Identity()

        # 2. Paragraph transformers (target and predictor)
        # Share architecture but separate weights
        self.target_transformer = ParagraphTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_sentences=max_sentences,
        )

        self.predictor_transformer = ParagraphTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_sentences=max_sentences,
        )

        # 3. Learned mask embedding
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 4. Projection heads
        self.target_head = ProjectionHead(
            input_dim=d_model,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
            dropout=projection_dropout,
        )

        self.predictor_head = ProjectionHead(
            input_dim=d_model,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
            dropout=projection_dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_mask: torch.Tensor,
        mask_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with masked sentence prediction.

        Args:
            input_ids: [B, S, T] - token IDs for each sentence
            attention_mask: [B, S, T] - attention mask for tokens
            sentence_mask: [B, S] - binary mask for valid sentences
            mask_idx: [B] - index of sentence to mask (for prediction)

        Returns:
            z_pred: [B, D] - predicted embeddings for masked sentence
            z_target: [B, D] - target embeddings for masked sentence
        """
        B, S, T = input_ids.shape

        # 1. Encode sentences
        sentence_embeddings = self.sentence_encoder(input_ids, attention_mask)  # [B, S, D_enc]
        sentence_embeddings = self.input_projection(sentence_embeddings)  # [B, S, D_model]

        # 2a. Target branch (full paragraph)
        target_contextualized = self.target_transformer(
            sentence_embeddings,
            sentence_mask,
        )  # [B, S, D_model]

        # Extract target embeddings for masked positions
        z_target = self._extract_masked_embeddings(
            target_contextualized,
            mask_idx,
        )  # [B, D_model]

        # Project target embeddings
        z_target = self.target_head(z_target)  # [B, D_proj]

        # 2b. Predictor branch (masked paragraph)
        # Replace masked sentence with learned mask embedding
        masked_embeddings = self._apply_mask(
            sentence_embeddings,
            mask_idx,
        )  # [B, S, D_model]

        predictor_contextualized = self.predictor_transformer(
            masked_embeddings,
            sentence_mask,
        )  # [B, S, D_model]

        # Extract predicted embeddings for masked positions
        z_pred = self._extract_masked_embeddings(
            predictor_contextualized,
            mask_idx,
        )  # [B, D_model]

        # Project predicted embeddings
        z_pred = self.predictor_head(z_pred)  # [B, D_proj]

        return z_pred, z_target

    def _apply_mask(
        self,
        sentence_embeddings: torch.Tensor,
        mask_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace specified sentences with mask embedding.

        Args:
            sentence_embeddings: [B, S, D]
            mask_idx: [B] - indices to mask

        Returns:
            masked_embeddings: [B, S, D]
        """
        B, S, D = sentence_embeddings.shape

        # Clone to avoid in-place modification
        masked_embeddings = sentence_embeddings.clone()

        # Replace masked positions with mask embedding
        for i in range(B):
            idx = mask_idx[i].item()
            masked_embeddings[i, idx, :] = self.mask_embedding[0, 0, :]

        return masked_embeddings

    def _extract_masked_embeddings(
        self,
        contextualized: torch.Tensor,
        mask_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract embeddings at masked positions.

        Args:
            contextualized: [B, S, D]
            mask_idx: [B] - indices to extract

        Returns:
            extracted: [B, D]
        """
        B, S, D = contextualized.shape

        # Gather embeddings at mask positions
        # mask_idx: [B] -> [B, 1, 1] -> [B, 1, D]
        mask_idx_expanded = mask_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D)

        # Gather: [B, S, D] + [B, 1, D] -> [B, 1, D]
        extracted = torch.gather(contextualized, dim=1, index=mask_idx_expanded)

        # Squeeze: [B, 1, D] -> [B, D]
        extracted = extracted.squeeze(1)

        return extracted


if __name__ == "__main__":
    # Test Hierarchical Sentence JEPA
    print("Testing HierarchicalSentenceJEPA...")

    model = HierarchicalSentenceJEPA(
        sentence_encoder_name="roberta-base",
        sentence_encoder_frozen=True,
        d_model=768,
        nhead=8,
        num_layers=2,
        projection_output_dim=512,
    )

    # Create dummy batch
    B, S, T = 4, 5, 20
    input_ids = torch.randint(0, 1000, (B, S, T))
    attention_mask = torch.ones(B, S, T)
    sentence_mask = torch.ones(B, S)
    sentence_mask[0, 4] = 0  # Pad last sentence
    mask_idx = torch.tensor([1, 2, 0, 3])  # Different mask for each batch

    # Forward pass
    z_pred, z_target = model(input_ids, attention_mask, sentence_mask, mask_idx)

    print(f"Input shape: {input_ids.shape}")
    print(f"z_pred shape: {z_pred.shape}")
    print(f"z_target shape: {z_target.shape}")
    assert z_pred.shape == (B, 512)
    assert z_target.shape == (B, 512)
    print("✓ HierarchicalSentenceJEPA test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
