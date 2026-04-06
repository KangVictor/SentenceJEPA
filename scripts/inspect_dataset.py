"""
Inspect a saved HuggingFace dataset to understand its structure.

This helps diagnose issues when loading datasets from disk.

Usage:
    python scripts/inspect_dataset.py --path /path/to/dataset
"""

import argparse
from datasets import load_from_disk


def inspect_dataset(dataset_path: str):
    """Inspect a dataset saved to disk."""
    print("="*60)
    print("Dataset Inspector")
    print("="*60)

    print(f"\nLoading dataset from: {dataset_path}")

    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        return

    print("✓ Dataset loaded successfully\n")

    # Basic info
    print("="*60)
    print("Basic Information")
    print("="*60)

    if hasattr(dataset, '__len__'):
        print(f"Total samples: {len(dataset)}")
    else:
        print(f"Total samples: Unknown (streaming/iterable)")

    # Features
    if hasattr(dataset, 'features'):
        print(f"\nFeatures:")
        for key, value in dataset.features.items():
            print(f"  - {key}: {value}")
    else:
        print(f"\nFeatures: Not available")

    # Column names
    if hasattr(dataset, 'column_names'):
        print(f"\nColumn names: {dataset.column_names}")
    else:
        print(f"\nColumn names: Not available")

    # Dataset type
    print(f"\nDataset type: {type(dataset)}")
    print(f"Dataset class: {dataset.__class__.__name__}")

    # Check if it's a DatasetDict
    if hasattr(dataset, 'keys'):
        print(f"\nDatasetDict splits: {list(dataset.keys())}")
        print(f"\nNote: This is a DatasetDict with multiple splits.")
        print(f"You need to specify which split to use.")
        print(f"\nExample: dataset['train'] or dataset['validation']")

        # Show info for each split
        for split_name in dataset.keys():
            split_dataset = dataset[split_name]
            print(f"\n  Split '{split_name}':")
            print(f"    Samples: {len(split_dataset) if hasattr(split_dataset, '__len__') else 'Unknown'}")
            if hasattr(split_dataset, 'column_names'):
                print(f"    Columns: {split_dataset.column_names}")
        return

    # Sample inspection
    print("\n" + "="*60)
    print("Sample Inspection")
    print("="*60)

    try:
        if hasattr(dataset, '__len__') and len(dataset) > 0:
            print(f"\nInspecting first sample...")
            sample = dataset[0]

            print(f"\nSample type: {type(sample)}")

            if isinstance(sample, dict):
                print(f"Sample keys: {list(sample.keys())}")
                print(f"\nSample contents:")
                for key, value in sample.items():
                    if isinstance(value, str):
                        preview = value[:100] + "..." if len(value) > 100 else value
                        print(f"  {key}: {preview}")
                    else:
                        print(f"  {key}: {type(value)} - {value}")
            elif isinstance(sample, str):
                print(f"\n✗ ERROR: Sample is a string, not a dictionary!")
                print(f"Sample preview: {sample[:200]}...")
                print(f"\nThis dataset structure is not compatible.")
                print(f"Expected: Dictionary with 'text' field")
                print(f"Got: Plain string")
            else:
                print(f"\nSample: {sample}")

            # Check multiple samples if available
            if hasattr(dataset, '__len__') and len(dataset) > 1:
                print(f"\nChecking sample consistency (first 3 samples)...")
                for i in range(min(3, len(dataset))):
                    sample = dataset[i]
                    if isinstance(sample, dict):
                        print(f"  Sample {i}: dict with keys {list(sample.keys())}")
                    else:
                        print(f"  Sample {i}: {type(sample)}")

        else:
            print("\nDataset is empty or length unknown")

    except Exception as e:
        print(f"\n✗ Error inspecting samples: {e}")
        import traceback
        traceback.print_exc()

    # Recommendations
    print("\n" + "="*60)
    print("Recommendations")
    print("="*60)

    if hasattr(dataset, 'column_names'):
        text_columns = [col for col in dataset.column_names if 'text' in col.lower() or 'content' in col.lower()]

        if text_columns:
            print(f"\nLikely text columns: {text_columns}")
            print(f"\nTo use this dataset, try:")
            for col in text_columns:
                print(f"  python scripts/train_hf.py \\")
                print(f"      --dataset from-disk \\")
                print(f"      --dataset-path {dataset_path} \\")
                print(f"      --text-column {col}")
        else:
            print(f"\nAvailable columns: {dataset.column_names}")
            print(f"\nTo use this dataset, specify the text column:")
            print(f"  python scripts/train_hf.py \\")
            print(f"      --dataset from-disk \\")
            print(f"      --dataset-path {dataset_path} \\")
            print(f"      --text-column YOUR_COLUMN_NAME")
    else:
        print("\nCould not determine column structure.")
        print("The dataset may need to be re-saved in the correct format.")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Inspect a saved HuggingFace dataset')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to the saved dataset')

    args = parser.parse_args()
    inspect_dataset(args.path)


if __name__ == "__main__":
    main()
