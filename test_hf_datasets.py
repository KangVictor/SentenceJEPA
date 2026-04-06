"""
Test script for HuggingFace datasets integration.

This script verifies that HuggingFace datasets work correctly
with the Sentence JEPA pipeline.

Run with: python test_hf_datasets.py
"""

import torch
from torch.utils.data import DataLoader

print("="*60)
print("Testing HuggingFace Datasets Integration")
print("="*60)

# Test 1: Import
print("\n[1/4] Testing imports...")
try:
    from data.hf_dataset import (
        load_wikipedia_dataset,
        load_c4_dataset,
        HFParagraphDataset,
    )
    from data import SentenceJEPACollator
    print("   ✓ Imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("\n   Please install datasets: pip install datasets")
    exit(1)

# Test 2: Load small Wikipedia sample
print("\n[2/4] Loading Wikipedia sample (5 paragraphs)...")
try:
    dataset = load_wikipedia_dataset(
        streaming=True,
        max_samples=5,  # Just 5 for testing
    )
    print("   ✓ Dataset loaded")
except Exception as e:
    print(f"   ✗ Failed to load dataset: {e}")
    print("\n   Note: This requires internet connection to download from HuggingFace")
    exit(1)

# Test 3: Iterate through samples
print("\n[3/4] Checking samples...")
samples = []
try:
    for i, item in enumerate(dataset):
        samples.append(item)
        print(f"   Sample {i+1}:")
        print(f"      Sentences: {len(item['sentences'])}")
        print(f"      First sentence: {item['sentences'][0][:60]}...")
        if i >= 2:  # Check first 3
            break
    print("   ✓ Samples look good")
except Exception as e:
    print(f"   ✗ Error processing samples: {e}")
    exit(1)

# Test 4: Create dataloader and get batch
print("\n[4/4] Testing with dataloader and collator...")
try:
    # Recreate dataset (can't reuse streaming iterator)
    dataset = load_wikipedia_dataset(
        streaming=True,
        max_samples=4,
    )

    collator = SentenceJEPACollator(
        tokenizer_name="roberta-base",
        max_tokens_per_sentence=32,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collator,
    )

    # Get one batch
    batch = next(iter(dataloader))

    print(f"   Batch shapes:")
    print(f"      input_ids: {batch['input_ids'].shape}")
    print(f"      attention_mask: {batch['attention_mask'].shape}")
    print(f"      sentence_mask: {batch['sentence_mask'].shape}")
    print(f"      mask_idx: {batch['mask_idx'].shape}")

    # Verify shapes are correct
    B, S, T = batch['input_ids'].shape
    assert batch['attention_mask'].shape == (B, S, T)
    assert batch['sentence_mask'].shape == (B, S)
    assert batch['mask_idx'].shape == (B,)

    print("   ✓ Dataloader working correctly")

except Exception as e:
    print(f"   ✗ Dataloader test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "="*60)
print("Test Summary")
print("="*60)
print("✓ HuggingFace datasets import")
print("✓ Wikipedia dataset loading")
print("✓ Sample processing")
print("✓ Dataloader integration")
print("\nAll tests passed! You can now train with HuggingFace datasets.")
print("\nTry:")
print("  python scripts/train_hf.py --dataset wikipedia --streaming --max-samples 10000")
print("="*60)
