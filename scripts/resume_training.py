"""
Resume training from a checkpoint.

Usage:
    # Resume exactly where you left off
    python scripts/resume_training.py \
        --checkpoint checkpoints/checkpoint_step_5000.pt \
        --config configs/base.yaml \
        --data-path /content/drive/MyDrive/SentenceJEPA

    # Fine-tune on new data
    python scripts/resume_training.py \
        --checkpoint checkpoints/best_model.pt \
        --config configs/base.yaml \
        --data-path /path/to/new/data \
        --reset-optimizer  # Start fresh optimizer state
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HierarchicalSentenceJEPA
from data import SentenceJEPACollator
from data.hf_dataset import load_from_disk_dataset
from train import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to training data')

    # Data options
    parser.add_argument('--dataset-type', type=str, default='from-disk',
                        choices=['from-disk', 'preprocessed'],
                        help='Type of dataset')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to use')

    # Resume options
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Reset optimizer state (for fine-tuning on new data)')
    parser.add_argument('--reset-scheduler', action='store_true',
                        help='Reset scheduler state')
    parser.add_argument('--reset-steps', action='store_true',
                        help='Reset global step counter')
    parser.add_argument('--new-lr', type=float, default=None,
                        help='Override learning rate (for fine-tuning)')

    # Training options
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--additional-epochs', type=int, default=None,
                        help='Additional epochs to train (overrides config)')

    args = parser.parse_args()

    print("="*60)
    print("Resume Training from Checkpoint")
    print("="*60)

    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)

    # Override config if specified
    if args.additional_epochs:
        config['training']['num_epochs'] = args.additional_epochs
        print(f"  Overriding num_epochs to: {args.additional_epochs}")

    if args.new_lr:
        config['training']['lr_rest'] = args.new_lr
        print(f"  Overriding learning rate to: {args.new_lr}")

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading dataset from: {args.data_path}")

    if args.dataset_type == 'preprocessed':
        import pickle
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)

        from torch.utils.data import Dataset as TorchDataset
        class PreprocessedDataset(TorchDataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        dataset = PreprocessedDataset(data)
        print(f"Loaded {len(dataset)} preprocessed samples")

    else:  # from-disk
        dataset = load_from_disk_dataset(
            dataset_path=args.data_path,
            text_column='text',
            min_sentences=config['data']['min_sentences'],
            max_sentences=config['data']['max_sentences'],
            max_samples=args.max_samples,
            use_streaming=False,
        )

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create collator
    collator = SentenceJEPACollator(
        tokenizer_name=config['model']['sentence_encoder']['model_name'],
        max_tokens_per_sentence=config['data']['max_tokens_per_sentence'],
        prefer_interior_mask=config['data']['prefer_interior_mask'],
        interior_prob=config['data']['interior_prob'],
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

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

    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model weights loaded")

    # Load optimizer state (unless reset)
    if not args.reset_optimizer:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
    else:
        print("✗ Optimizer state reset (starting fresh)")

    # Load scheduler state (unless reset)
    if not args.reset_scheduler:
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✓ Scheduler state loaded")
    else:
        print("✗ Scheduler state reset (starting fresh)")

    # Load training progress (unless reset)
    if not args.reset_steps:
        trainer.global_step = checkpoint['global_step']
        trainer.epoch = checkpoint['epoch']
        trainer.best_recall = checkpoint.get('best_recall', 0.0)
        print(f"✓ Training progress loaded (step {trainer.global_step}, epoch {trainer.epoch})")
    else:
        print("✗ Training progress reset (starting from step 0)")

    print(f"\nCheckpoint info:")
    print(f"  Original global step: {checkpoint['global_step']}")
    print(f"  Original epoch: {checkpoint['epoch']}")
    print(f"  Best recall: {checkpoint.get('best_recall', 'N/A')}")

    # Show what will happen
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Starting from step: {trainer.global_step}")
    print(f"Starting from epoch: {trainer.epoch}")
    print(f"Will train for: {config['training']['num_epochs']} epochs")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['lr_rest']}")
    print(f"Device: {device}")

    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")

    trainer.train()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final global step: {trainer.global_step}")
    print(f"Best Recall@1: {trainer.best_recall:.4f}")
    print(f"Checkpoints saved to: {config['paths']['checkpoint_dir']}")


if __name__ == "__main__":
    main()
