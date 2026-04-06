"""
More rigorous evaluation: retrieve masked sentence among OTHER sentences from THE SAME paragraph.

This tests if the model can distinguish between different sentences in the same context,
not just match topics across different paragraphs.

Usage:
    python scripts/eval_within_paragraph.py --checkpoint checkpoints/best_model_15.pt \
        --config configs/base.yaml --data data/test_corpus.txt
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
import torch.nn.functional as F


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@torch.no_grad()
def evaluate_within_paragraph(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate retrieval within the same paragraph.

    For each paragraph with N sentences:
    - Mask sentence i
    - Predict embedding for sentence i
    - Compare against actual embeddings of ALL N sentences in that paragraph
    - Check if sentence i ranks highest

    This is MUCH harder than cross-paragraph retrieval.
    """
    model.eval()

    total_samples = 0
    correct_at_1 = 0
    correct_at_2 = 0
    correct_at_3 = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentence_mask = batch['sentence_mask'].to(device)
        mask_idx = batch['mask_idx'].to(device)

        B, S, T = input_ids.shape

        # For each item in batch
        for b in range(B):
            # Get number of valid sentences
            num_sentences = sentence_mask[b].sum().item()

            if num_sentences < 3:
                continue  # Skip paragraphs with too few sentences

            # Get masked sentence index
            masked_idx = mask_idx[b].item()

            # Forward pass for prediction
            z_pred, _ = model(
                input_ids[b:b+1],
                attention_mask[b:b+1],
                sentence_mask[b:b+1],
                mask_idx[b:b+1],
            )  # [1, D]

            # Get embeddings for ALL sentences in this paragraph (as targets)
            # We need to encode each sentence position as if it were masked
            candidate_embeddings = []

            for sent_idx in range(num_sentences):
                # Create a batch where this sentence is the "target"
                _, z_target = model(
                    input_ids[b:b+1],
                    attention_mask[b:b+1],
                    sentence_mask[b:b+1],
                    torch.tensor([sent_idx], device=device),
                )  # [1, D]
                candidate_embeddings.append(z_target)

            # Stack candidates: [num_sentences, D]
            candidates = torch.cat(candidate_embeddings, dim=0)

            # Compute similarities
            z_pred_norm = F.normalize(z_pred, p=2, dim=-1)  # [1, D]
            candidates_norm = F.normalize(candidates, p=2, dim=-1)  # [N, D]

            similarities = torch.mm(z_pred_norm, candidates_norm.t())  # [1, N]
            similarities = similarities[0]  # [N]

            # Rank candidates
            ranked_indices = torch.argsort(similarities, descending=True)

            # Check if correct sentence is in top K
            rank_of_correct = (ranked_indices == masked_idx).nonzero(as_tuple=True)[0].item()

            if rank_of_correct == 0:
                correct_at_1 += 1
                correct_at_2 += 1
                correct_at_3 += 1
            elif rank_of_correct == 1:
                correct_at_2 += 1
                correct_at_3 += 1
            elif rank_of_correct == 2:
                correct_at_3 += 1

            total_samples += 1

            if total_samples % 50 == 0:
                print(f"Processed: {total_samples}, Recall@1: {correct_at_1/total_samples:.4f}")

    return {
        'total': total_samples,
        'recall@1': correct_at_1 / total_samples,
        'recall@2': correct_at_2 / total_samples,
        'recall@3': correct_at_3 / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate within-paragraph retrieval')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--data', type=str, default='data/test_corpus.txt')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=1)  # Must be 1 for this eval
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Within-Paragraph Retrieval Evaluation")
    print(f"{'='*60}\n")

    # Load config
    config = load_config(args.config)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create dataset
    print(f"\nLoading data: {args.data}")
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

    # Create dataloader (batch_size MUST be 1)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
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
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Evaluate
    print("\n" + "="*60)
    print("Evaluating...")
    print("="*60 + "\n")
    print("This will be MUCH slower than cross-paragraph evaluation")
    print("because we need to encode each sentence separately.\n")

    metrics = evaluate_within_paragraph(model, dataloader, device)

    # Print results
    print("\n" + "="*60)
    print("Within-Paragraph Retrieval Results")
    print("="*60)
    print(f"\nTotal paragraphs evaluated: {metrics['total']}")
    print(f"\nRecall@1: {metrics['recall@1']:.4f}")
    print(f"Recall@2: {metrics['recall@2']:.4f}")
    print(f"Recall@3: {metrics['recall@3']:.4f}")
    print("\nThis measures: Can the model identify the correct")
    print("masked sentence among ALL sentences in the SAME paragraph?")
    print("(Much harder than cross-paragraph topic matching!)")
    print()


if __name__ == "__main__":
    main()
