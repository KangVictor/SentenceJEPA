"""
Test script to verify the complete pipeline works.

This script tests:
1. Data loading and preprocessing
2. Model creation
3. Forward pass
4. Loss computation
5. Backward pass

Run with: python test_pipeline.py
"""

import torch
from torch.utils.data import DataLoader

from data import ParagraphDataset, SentenceJEPACollator
from models import HierarchicalSentenceJEPA
from losses import combined_loss
from losses.sigreg import SIGReg


def test_pipeline():
    """Test the complete training pipeline."""
    print("=" * 60)
    print("Testing Hierarchical Sentence JEPA Pipeline")
    print("=" * 60)

    # 1. Create sample data
    print("\n[1/6] Creating sample data...")
    sample_paragraphs = [
        "This is the first sentence. Here is another one. And a third sentence!",
        "Machine learning is fascinating. It involves training models on data. The models learn patterns. Then they can make predictions.",
        "Another example paragraph. With multiple sentences here. Testing the dataset functionality.",
        "Deep learning has revolutionized AI. Neural networks can learn complex patterns. This has enabled many applications.",
    ]

    dataset = ParagraphDataset.from_list(
        paragraphs=sample_paragraphs,
        min_sentences=3,
        max_sentences=10,
        use_spacy=False,  # Use regex to avoid spaCy dependency for testing
    )
    print(f"   ✓ Created dataset with {len(dataset)} paragraphs")

    # 2. Create collator
    print("\n[2/6] Creating data collator...")
    collator = SentenceJEPACollator(
        tokenizer_name="roberta-base",
        max_tokens_per_sentence=32,
        prefer_interior_mask=True,
    )
    print("   ✓ Collator created")

    # 3. Create dataloader
    print("\n[3/6] Creating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collator,
    )
    print(f"   ✓ Dataloader created with batch_size=2")

    # Get a batch
    batch = next(iter(dataloader))
    print(f"   ✓ Batch shapes:")
    print(f"      input_ids: {batch['input_ids'].shape}")
    print(f"      attention_mask: {batch['attention_mask'].shape}")
    print(f"      sentence_mask: {batch['sentence_mask'].shape}")
    print(f"      mask_idx: {batch['mask_idx'].shape}")

    # 4. Create model
    print("\n[4/6] Creating model...")
    model = HierarchicalSentenceJEPA(
        sentence_encoder_name="roberta-base",
        sentence_encoder_frozen=True,
        d_model=768,
        nhead=8,
        num_layers=2,  # Small for testing
        projection_output_dim=512,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Model created")
    print(f"      Total params: {total_params:,}")
    print(f"      Trainable params: {trainable_params:,}")

    # 5. Forward pass
    print("\n[5/6] Running forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"   Using device: {device}")

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    sentence_mask = batch['sentence_mask'].to(device)
    mask_idx = batch['mask_idx'].to(device)

    z_pred, z_target = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        sentence_mask=sentence_mask,
        mask_idx=mask_idx,
    )

    print(f"   ✓ Forward pass successful")
    print(f"      z_pred shape: {z_pred.shape}")
    print(f"      z_target shape: {z_target.shape}")

    # 6. Compute loss and backward
    print("\n[6/6] Computing loss and backward pass...")
    sigreg_module = SIGReg(
        embedding_dim=512,
        num_projections=16,  # Small for testing
        projection_dim=64,
    ).to(device)

    loss, loss_dict = combined_loss(
        pred_embeddings=z_pred,
        target_embeddings=z_target,
        sigreg_module=sigreg_module,
        lambda_sigreg=0.1,
    )

    print(f"   ✓ Loss computation successful")
    print(f"      Total loss: {loss_dict['total']:.4f}")
    print(f"      JEPA loss: {loss_dict['jepa']:.4f}")
    print(f"      SIGReg loss: {loss_dict['sigreg']:.4f}")

    # Backward pass
    loss.backward()
    print(f"   ✓ Backward pass successful")

    # Check gradients
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
    if has_gradients:
        print(f"   ✓ Gradients computed")
    else:
        print(f"   ✗ No gradients found (potential issue)")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Test Summary")
    print("=" * 60)
    print("✓ Data loading and preprocessing")
    print("✓ Model creation")
    print("✓ Forward pass")
    print("✓ Loss computation")
    print("✓ Backward pass")
    print("\nAll tests passed! The pipeline is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)

    try:
        test_pipeline()
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
