"""
Training script for Sentence JEPA model.

Usage:
    python scripts/train.py --config configs/base.yaml --data data/sample_data.txt
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HierarchicalSentenceJEPA
from data import ParagraphDataset, SentenceJEPACollator
from train import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_sample_data(output_path: str):
    """Create sample training data for demonstration."""
    sample_paragraphs = [
        "Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data. These systems improve their performance over time without being explicitly programmed.",

        "Deep learning uses neural networks with multiple layers. These networks can learn complex patterns in data. They have revolutionized fields like computer vision and natural language processing.",

        "Natural language processing enables computers to understand human language. It involves various tasks like translation, summarization, and sentiment analysis. Modern NLP systems use transformer architectures for better performance.",

        "Computer vision allows machines to interpret visual information. It includes tasks like object detection, image segmentation, and facial recognition. Convolutional neural networks are commonly used in computer vision applications.",

        "Reinforcement learning trains agents through trial and error. The agent learns to make decisions by receiving rewards or penalties. This approach has achieved remarkable success in game playing and robotics.",

        "Transfer learning leverages knowledge from one task to improve performance on another. It reduces the need for large datasets and computational resources. Pre-trained models are fine-tuned for specific applications.",

        "The attention mechanism helps models focus on relevant parts of the input. It was introduced to improve sequence-to-sequence models. Transformers rely heavily on self-attention for processing sequences.",

        "Generative models can create new data samples similar to training data. They include architectures like GANs and VAEs. These models have applications in image generation, text synthesis, and drug discovery.",

        "Embeddings represent discrete objects as continuous vectors. They capture semantic relationships between entities. Word embeddings and sentence embeddings are fundamental in NLP systems.",

        "Self-supervised learning uses the data itself to create supervisory signals. It reduces reliance on labeled data. Techniques like masked language modeling have proven highly effective.",

        "Few-shot learning aims to learn from very limited examples. It mimics human ability to generalize from few instances. Meta-learning approaches are often used for few-shot scenarios.",

        "Contrastive learning learns representations by comparing similar and dissimilar examples. It has become popular for self-supervised pre-training. SimCLR and MoCo are well-known contrastive methods.",

        "Model interpretability helps us understand how AI systems make decisions. It's crucial for building trust in machine learning applications. Techniques include attention visualization and feature importance analysis.",

        "Data augmentation artificially increases training data size. It helps models generalize better and reduces overfitting. Common techniques include random cropping, rotation, and adding noise.",

        "Regularization prevents models from overfitting to training data. Techniques include dropout, weight decay, and early stopping. Proper regularization is essential for good generalization performance.",
    ] * 10  # Repeat for more training data

    with open(output_path, 'w') as f:
        f.write('\n\n'.join(sample_paragraphs))

    print(f"Sample data created at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Sentence JEPA model')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to config file')
    parser.add_argument('--data', type=str, default='data/sample_data.txt',
                        help='Path to training data file')
    parser.add_argument('--create-sample-data', action='store_true',
                        help='Create sample data file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Create sample data if requested
    if args.create_sample_data:
        Path(args.data).parent.mkdir(parents=True, exist_ok=True)
        create_sample_data(args.data)

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
        use_spacy=False,  # Use regex for simplicity
    )

    # Split into train and val
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
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
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
    print("\nStarting training...")
    trainer.train()

    print("\nTraining complete!")
    print(f"Best Recall@1: {trainer.best_recall:.4f}")
    print(f"Checkpoints saved to: {config['paths']['checkpoint_dir']}")


if __name__ == "__main__":
    main()
