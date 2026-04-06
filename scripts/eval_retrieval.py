"""
Evaluation script for masked sentence retrieval.

Usage:
    python scripts/eval_retrieval.py --checkpoint checkpoints/best_model.pt --config configs/base.yaml --data data/sample_data.txt
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HierarchicalSentenceJEPA
from data import ParagraphDataset, SentenceJEPACollator
from train import evaluate_retrieval


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate Sentence JEPA model on retrieval')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to config file')
    parser.add_argument('--data', type=str, default='data/sample_data.txt',
                        help='Path to evaluation data file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for evaluation')
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset
    print(f"\nLoading data from: {args.data}")
    dataset = ParagraphDataset.from_text_file(
        file_path=args.data,
        min_sentences=config['data']['min_sentences'],
        max_sentences=config['data']['max_sentences'],
        use_spacy=False,
    )
    print(f"Dataset size: {len(dataset)}")

    # Create collator
    collator = SentenceJEPACollator(
        tokenizer_name=config['model']['sentence_encoder']['model_name'],
        max_tokens_per_sentence=config['data']['max_tokens_per_sentence'],
        prefer_interior_mask=config['data']['prefer_interior_mask'],
        interior_prob=config['data']['interior_prob'],
    )

    # Create dataloader
    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Create model
    print("\nCreating model...")
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
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        print(f"Global step: {checkpoint['global_step']}")

    # Evaluate
    print("\n=== Running Evaluation ===\n")
    metrics = evaluate_retrieval(
        model=model,
        dataloader=dataloader,
        device=device,
        k_values=config['evaluation']['recall_k'],
        max_batches=None,
    )

    # Print results
    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key:>12}: {value:.4f}")
    print()


if __name__ == "__main__":
    main()
