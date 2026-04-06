"""
Training script for Sentence JEPA with HuggingFace datasets.

Supports popular datasets like Wikipedia, C4, BookCorpus, etc.

Usage:
    # Wikipedia
    python scripts/train_hf.py --dataset wikipedia --streaming

    # C4
    python scripts/train_hf.py --dataset c4 --streaming --max-samples 100000

    # Custom HuggingFace dataset
    python scripts/train_hf.py --dataset custom --hf-name "username/dataset" --text-column "content"
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
from data import SentenceJEPACollator
from data.hf_dataset import (
    load_wikipedia_dataset,
    load_c4_dataset,
    load_bookcorpus_dataset,
    load_from_disk_dataset,
    HFParagraphDataset,
    HFParagraphDatasetMapStyle,
)
from train import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_dataset(args, config):
    """Load the appropriate HuggingFace dataset."""
    print(f"\n{'='*60}")
    print(f"Loading HuggingFace Dataset: {args.dataset}")
    print(f"{'='*60}\n")

    min_sentences = config['data']['min_sentences']
    max_sentences = config['data']['max_sentences']

    if args.dataset == 'wikipedia':
        dataset = load_wikipedia_dataset(
            language=args.wiki_lang,
            date=args.wiki_date,
            streaming=args.streaming,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            max_samples=args.max_samples,
        )

    elif args.dataset == 'c4':
        dataset = load_c4_dataset(
            streaming=args.streaming,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            max_samples=args.max_samples,
            split='train',
        )

    elif args.dataset == 'bookcorpus':
        dataset = load_bookcorpus_dataset(
            streaming=args.streaming,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            max_samples=args.max_samples,
        )

    elif args.dataset == 'preprocessed':
        if not args.dataset_path:
            raise ValueError("Must provide --dataset-path for preprocessed dataset")

        import pickle
        print(f"Loading preprocessed dataset from: {args.dataset_path}")

        with open(args.dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        print(f"✓ Loaded {len(dataset)} preprocessed paragraphs")

        # Load metadata if available
        metadata_path = args.dataset_path + '.metadata'
        from pathlib import Path
        if Path(metadata_path).exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"  Original samples: {metadata.get('total_samples_processed', 'N/A')}")
            print(f"  Min sentences: {metadata.get('min_sentences', 'N/A')}")
            print(f"  Max sentences: {metadata.get('max_sentences', 'N/A')}")

        # Wrap in a simple dataset class
        from torch.utils.data import Dataset as TorchDataset
        class PreprocessedDataset(TorchDataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        dataset = PreprocessedDataset(dataset)

    elif args.dataset == 'from-disk':
        if not args.dataset_path:
            raise ValueError("Must provide --dataset-path for from-disk dataset")

        dataset = load_from_disk_dataset(
            dataset_path=args.dataset_path,
            text_column=args.text_column or 'text',
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            max_samples=args.max_samples,
            use_streaming=args.streaming,
        )

    elif args.dataset == 'custom':
        if not args.hf_name:
            raise ValueError("Must provide --hf-name for custom dataset")

        from datasets import load_dataset

        print(f"Loading custom dataset: {args.hf_name}")
        hf_dataset = load_dataset(
            args.hf_name,
            split=args.split,
            streaming=args.streaming,
        )

        text_column = args.text_column or 'text'

        if args.streaming:
            dataset = HFParagraphDataset(
                dataset=hf_dataset,
                text_column=text_column,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
                max_samples=args.max_samples,
            )
        else:
            dataset = HFParagraphDatasetMapStyle(
                dataset=hf_dataset,
                text_column=text_column,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
            )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Train Sentence JEPA on HuggingFace datasets')

    # Dataset selection
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['wikipedia', 'c4', 'bookcorpus', 'custom', 'from-disk', 'preprocessed'],
                        help='Which HuggingFace dataset to use')

    # General dataset options
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming mode (recommended for large datasets)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use (None = all)')

    # Wikipedia-specific options
    parser.add_argument('--wiki-lang', type=str, default='en',
                        help='Wikipedia language code')
    parser.add_argument('--wiki-date', type=str, default='20220301',
                        help='Wikipedia dump date')

    # Custom dataset options
    parser.add_argument('--hf-name', type=str, default=None,
                        help='HuggingFace dataset name (for custom datasets)')
    parser.add_argument('--text-column', type=str, default='text',
                        help='Name of text column in custom dataset')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use')

    # From-disk dataset options
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to pre-downloaded dataset directory (for from-disk)')

    # Training options
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--val-split', type=float, default=0.01,
                        help='Fraction of data to use for validation (only for non-streaming)')

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

    # Get dataset
    train_dataset = get_dataset(args, config)

    # For validation, we'll create a separate dataset
    # For streaming datasets, we'll skip validation or use a small held-out portion
    val_dataset = None
    if not args.streaming:
        print("\nSplitting dataset into train/val...")
        from torch.utils.data import random_split
        total_size = len(train_dataset)
        val_size = int(total_size * args.val_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Train size: {train_size}, Val size: {val_size}")
    else:
        print("\nStreaming mode: validation will be skipped")
        print("For evaluation, consider using --max-samples to create a separate eval set")

    # Create collator
    print("\nCreating data collator...")
    collator = SentenceJEPACollator(
        tokenizer_name=config['model']['sentence_encoder']['model_name'],
        max_tokens_per_sentence=config['data']['max_tokens_per_sentence'],
        prefer_interior_mask=config['data']['prefer_interior_mask'],
        interior_prob=config['data']['interior_prob'],
        mask_ratio=config['data'].get('mask_ratio', None),
    )

    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=not args.streaming,  # Can't shuffle streaming datasets
        collate_fn=collator,
        num_workers=0,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr_sentence_encoder=config['training']['lr_sentence_encoder'],
        lr_rest=config['training']['lr_rest'],
        num_epochs=config['training']['num_epochs'],
        warmup_steps=config['training']['warmup_steps'],
        lambda_sigreg=config['loss']['lambda_sigreg'],
        num_projections=config['loss']['sigreg']['num_projections'],
        projection_dim=config['loss']['sigreg']['projection_dim'],
        gradient_clip=config['training']['gradient_clip'],
        device=device,
        log_every=config['training']['log_every'],
        eval_every=config['training']['eval_every'],
        save_every=config['training']['save_every'],
        checkpoint_dir=config['paths']['checkpoint_dir'],
        recall_k=config['evaluation']['recall_k'],
    )

    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")

    print(f"Dataset: {args.dataset}")
    print(f"Streaming: {args.streaming}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples:,}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Device: {device}")
    print()

    trainer.train()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Recall@1: {trainer.best_recall:.4f}")
    print(f"Checkpoints saved to: {config['paths']['checkpoint_dir']}")


if __name__ == "__main__":
    main()
