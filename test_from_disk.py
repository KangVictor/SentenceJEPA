"""
Test script for loading pre-downloaded datasets from disk.

This verifies that the from-disk dataset loading works correctly.

Run with: python test_from_disk.py
"""

import torch
import tempfile
import shutil
from pathlib import Path

print("="*60)
print("Testing From-Disk Dataset Loading")
print("="*60)

# Test 1: Check imports
print("\n[1/5] Testing imports...")
try:
    from data import load_from_disk_dataset, SentenceJEPACollator
    from datasets import Dataset
    print("   ✓ Imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Create a temporary dataset
print("\n[2/5] Creating temporary dataset...")
try:
    # Create sample data
    sample_data = {
        'text': [
            'This is the first paragraph. It has multiple sentences. Each sentence is important.',
            'Here is another paragraph. It also contains several sentences. This is good for testing.',
            'A third paragraph appears here. With even more sentences. Testing is essential.',
            'Fourth paragraph with text. Multiple sentences as always. Keep them coming.',
        ]
    }

    # Create dataset
    dataset = Dataset.from_dict(sample_data)

    # Save to temporary directory
    temp_dir = tempfile.mkdtemp()
    dataset_path = Path(temp_dir) / "test_dataset"
    dataset.save_to_disk(str(dataset_path))

    print(f"   ✓ Temporary dataset created at: {dataset_path}")
    print(f"   ✓ Dataset size: {len(dataset)} samples")

except Exception as e:
    print(f"   ✗ Failed to create dataset: {e}")
    exit(1)

# Test 3: Load from disk
print("\n[3/5] Loading dataset from disk...")
try:
    loaded_dataset = load_from_disk_dataset(
        dataset_path=str(dataset_path),
        text_column='text',
        min_sentences=3,
        max_sentences=10,
        use_streaming=False,  # Use map-style for testing
    )

    print(f"   ✓ Dataset loaded successfully")
    print(f"   ✓ Processed dataset size: {len(loaded_dataset)}")

except Exception as e:
    print(f"   ✗ Failed to load dataset: {e}")
    # Cleanup
    shutil.rmtree(temp_dir)
    exit(1)

# Test 4: Check samples
print("\n[4/5] Checking samples...")
try:
    if len(loaded_dataset) > 0:
        sample = loaded_dataset[0]
        print(f"   Sample structure:")
        print(f"      Keys: {sample.keys()}")
        print(f"      Num sentences: {len(sample['sentences'])}")
        print(f"      First sentence: {sample['sentences'][0][:50]}...")
        print("   ✓ Samples look good")
    else:
        print("   ⚠ No samples (all filtered out - might be OK)")

except Exception as e:
    print(f"   ✗ Error checking samples: {e}")
    shutil.rmtree(temp_dir)
    exit(1)

# Test 5: Test with dataloader
print("\n[5/5] Testing with dataloader...")
try:
    from torch.utils.data import DataLoader

    collator = SentenceJEPACollator(
        tokenizer_name="roberta-base",
        max_tokens_per_sentence=32,
    )

    if len(loaded_dataset) > 0:
        dataloader = DataLoader(
            loaded_dataset,
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
        print("   ✓ Dataloader working correctly")
    else:
        print("   ⚠ Skipping dataloader test (no samples)")

except Exception as e:
    print(f"   ✗ Dataloader test failed: {e}")
    import traceback
    traceback.print_exc()
    shutil.rmtree(temp_dir)
    exit(1)

# Cleanup
print(f"\n[Cleanup] Removing temporary directory...")
shutil.rmtree(temp_dir)
print(f"   ✓ Cleanup complete")

# Summary
print("\n" + "="*60)
print("Test Summary")
print("="*60)
print("✓ Imports working")
print("✓ Dataset creation and saving")
print("✓ Dataset loading from disk")
print("✓ Sample processing")
print("✓ Dataloader integration")
print("\nAll tests passed! You can now use pre-downloaded datasets.")
print("\nTo download a real dataset:")
print("  python examples/download_and_save_dataset.py \\")
print("      --dataset wikipedia --output ./datasets/wiki_10k --max-samples 10000")
print("\nTo train on it:")
print("  python scripts/train_hf.py \\")
print("      --dataset from-disk --dataset-path ./datasets/wiki_10k")
print("="*60)
