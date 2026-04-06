"""
Sentence encoder using HuggingFace transformers.
Encodes individual sentences into fixed-dimensional embeddings.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SentenceEncoder(nn.Module):
    """
    Sentence encoder that wraps a HuggingFace transformer model.

    Takes tokenized sentences and produces sentence-level embeddings
    using mean pooling over tokens.
    """

    def __init__(self, model_name: str = "roberta-base", frozen: bool = True, pooling: str = "mean"):
        """
        Args:
            model_name: HuggingFace model name (e.g., "roberta-base")
            frozen: If True, freeze encoder weights
            pooling: Pooling strategy ("mean", "cls", "max")
        """
        super().__init__()
        self.model_name = model_name
        self.frozen = frozen
        self.pooling = pooling

        # Load pretrained model
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Freeze if requested
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode sentences to embeddings.

        Args:
            input_ids: [B, S, T] - token ids
            attention_mask: [B, S, T] - attention mask

        Returns:
            embeddings: [B, S, D] - sentence embeddings
        """
        B, S, T = input_ids.shape

        # Reshape to process all sentences at once
        input_ids_flat = input_ids.view(B * S, T)  # [B*S, T]
        attention_mask_flat = attention_mask.view(B * S, T)  # [B*S, T]

        # Encode with transformer
        if self.frozen:
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids_flat,
                    attention_mask=attention_mask_flat
                )
        else:
            outputs = self.encoder(
                input_ids=input_ids_flat,
                attention_mask=attention_mask_flat
            )

        # Get hidden states: [B*S, T, D]
        hidden_states = outputs.last_hidden_state

        # Pool to sentence embeddings
        if self.pooling == "mean":
            # Mean pooling with attention mask
            embeddings = self._mean_pooling(hidden_states, attention_mask_flat)  # [B*S, D]
        elif self.pooling == "cls":
            # Use [CLS] token (first token)
            embeddings = hidden_states[:, 0, :]  # [B*S, D]
        elif self.pooling == "max":
            # Max pooling
            embeddings = hidden_states.max(dim=1)[0]  # [B*S, D]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Reshape back to [B, S, D]
        embeddings = embeddings.view(B, S, -1)

        return embeddings

    def _mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling with attention mask.

        Args:
            hidden_states: [B*S, T, D]
            attention_mask: [B*S, T]

        Returns:
            pooled: [B*S, D]
        """
        # Expand mask to match hidden states
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # Sum and divide by number of tokens
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)  # [B*S, D]
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # [B*S, D]

        return sum_embeddings / sum_mask

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.hidden_size


if __name__ == "__main__":
    # Test sentence encoder
    print("Testing SentenceEncoder...")

    encoder = SentenceEncoder(model_name="roberta-base", frozen=True)
    print(f"Embedding dim: {encoder.get_embedding_dim()}")

    # Create dummy input
    B, S, T = 2, 3, 10
    input_ids = torch.randint(0, 1000, (B, S, T))
    attention_mask = torch.ones(B, S, T)

    # Forward pass
    embeddings = encoder(input_ids, attention_mask)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {embeddings.shape}")
    assert embeddings.shape == (B, S, encoder.get_embedding_dim())
    print("✓ SentenceEncoder test passed!")
