"""
Use a trained Sentence JEPA model for inference.

Examples of what you can do with the trained model:
1. Generate sentence embeddings
2. Find similar sentences
3. Predict masked sentences
4. Sentence clustering

Usage:
    python scripts/use_model.py \
        --checkpoint checkpoints/best_model.pt \
        --config configs/base.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HierarchicalSentenceJEPA
from transformers import AutoTokenizer
import numpy as np


def load_model(checkpoint_path: str, config: dict, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    # Create model with same config as training
    model = HierarchicalSentenceJEPA(
        sentence_encoder_name=config['model']['sentence_encoder']['model_name'],
        sentence_encoder_frozen=config['model']['sentence_encoder']['frozen'],
        sentence_pooling=config['model']['sentence_encoder']['pooling'],
        d_model=config['model']['paragraph_transformer']['d_model'],
        nhead=config['model']['paragraph_transformer']['nhead'],
        num_layers=config['model']['paragraph_transformer']['num_layers'],
        dim_feedforward=config['model']['paragraph_transformer']['dim_feedforward'],
        dropout=config['model']['paragraph_transformer']['dropout'],
        projection_hidden_dim=config['model']['projection']['hidden_dim'],
        projection_output_dim=config['model']['projection']['output_dim'],
        projection_dropout=config['model']['projection']['dropout'],
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Trained for {checkpoint['global_step']} steps")
    print(f"  Best Recall@1: {checkpoint.get('best_recall', 'N/A')}")

    return model


def encode_paragraph(model, tokenizer, paragraph_text: str, device: str = 'cuda'):
    """
    Encode a paragraph into contextualized sentence embeddings.

    Args:
        model: Trained HierarchicalSentenceJEPA model
        tokenizer: Tokenizer
        paragraph_text: Paragraph text
        device: Device

    Returns:
        embeddings: [num_sentences, 512] embeddings
        sentences: List of sentences
    """
    from data.dataset import split_into_sentences

    # Split into sentences
    sentences = split_into_sentences(paragraph_text, use_spacy=False)

    if len(sentences) < 2:
        print("Warning: Paragraph has fewer than 2 sentences")
        return None, sentences

    # Tokenize
    tokenized = []
    for sent in sentences:
        tokens = tokenizer(
            sent,
            max_length=64,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        tokenized.append(tokens)

    # Stack into batch
    input_ids = torch.stack([t['input_ids'].squeeze(0) for t in tokenized]).unsqueeze(0)  # [1, S, T]
    attention_mask = torch.stack([t['attention_mask'].squeeze(0) for t in tokenized]).unsqueeze(0)  # [1, S, T]
    sentence_mask = torch.ones(1, len(sentences))  # [1, S]

    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    sentence_mask = sentence_mask.to(device)

    # Create dummy mask_idx (we'll use target branch for all sentences)
    mask_idx = torch.tensor([0]).to(device)

    # Forward pass through model
    with torch.no_grad():
        # Get contextualized representations
        sentence_embeddings = model.sentence_encoder(input_ids, attention_mask)  # [1, S, D]
        sentence_embeddings = model.input_projection(sentence_embeddings)  # [1, S, D]

        # Process with target transformer (full context)
        contextualized = model.target_transformer(sentence_embeddings, sentence_mask)  # [1, S, D]

        # Project to embedding space
        embeddings = model.target_head(contextualized)  # [1, S, 512]

    # Remove batch dimension
    embeddings = embeddings.squeeze(0).cpu().numpy()  # [S, 512]

    return embeddings, sentences


def find_similar_sentences(embeddings1, sentences1, embeddings2, sentences2, top_k=3):
    """
    Find most similar sentences between two paragraphs.

    Args:
        embeddings1: [S1, 512] embeddings from paragraph 1
        sentences1: List of sentences from paragraph 1
        embeddings2: [S2, 512] embeddings from paragraph 2
        sentences2: List of sentences from paragraph 2
        top_k: Number of similar pairs to return

    Returns:
        List of (sent1_idx, sent2_idx, similarity) tuples
    """
    # Normalize embeddings
    emb1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    emb2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity = np.dot(emb1_norm, emb2_norm.T)  # [S1, S2]

    # Find top-k similar pairs
    flat_indices = similarity.flatten().argsort()[-top_k:][::-1]
    pairs = []

    for idx in flat_indices:
        i = idx // similarity.shape[1]
        j = idx % similarity.shape[1]
        sim = similarity[i, j]
        pairs.append((i, j, sim))

    return pairs


def main():
    parser = argparse.ArgumentParser(description='Use trained Sentence JEPA model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(args.checkpoint, config, device=device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['sentence_encoder']['model_name'])

    # Example usage
    print("\n" + "="*60)
    print("Example: Encoding Paragraphs")
    print("="*60)

    paragraph1 = """
    Machine learning is a subset of artificial intelligence.
    It focuses on building systems that learn from data.
    These systems improve their performance over time.
    """

    paragraph2 = """
    Deep learning uses neural networks with many layers.
    These networks can learn hierarchical representations.
    They have achieved state-of-the-art results in many tasks.
    """

    # Encode paragraphs
    print("\nEncoding paragraph 1...")
    emb1, sent1 = encode_paragraph(model, tokenizer, paragraph1, device=device)

    print("\nEncoding paragraph 2...")
    emb2, sent2 = encode_paragraph(model, tokenizer, paragraph2, device=device)

    print(f"\nParagraph 1: {len(sent1)} sentences, embeddings shape: {emb1.shape}")
    print(f"Paragraph 2: {len(sent2)} sentences, embeddings shape: {emb2.shape}")

    # Find similar sentences
    print("\n" + "="*60)
    print("Finding Similar Sentences")
    print("="*60)

    pairs = find_similar_sentences(emb1, sent1, emb2, sent2, top_k=3)

    print(f"\nTop {len(pairs)} most similar sentence pairs:")
    for i, (idx1, idx2, sim) in enumerate(pairs, 1):
        print(f"\n{i}. Similarity: {sim:.4f}")
        print(f"   Paragraph 1, sentence {idx1+1}: {sent1[idx1].strip()}")
        print(f"   Paragraph 2, sentence {idx2+1}: {sent2[idx2].strip()}")

    print("\n" + "="*60)
    print("Example Complete")
    print("="*60)
    print("\nYou can now use this model to:")
    print("  - Generate sentence embeddings")
    print("  - Find similar sentences")
    print("  - Cluster sentences")
    print("  - Semantic search")


if __name__ == "__main__":
    main()
