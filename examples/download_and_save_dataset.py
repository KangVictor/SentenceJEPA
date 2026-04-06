"""
Example: Download a HuggingFace dataset and save it to disk.

This is useful when:
1. You want to work offline later
2. You're on limited bandwidth
3. You want to process/filter the dataset once and reuse it
4. You're working with Google Colab or similar environments

Usage:
    python examples/download_and_save_dataset.py --dataset wikipedia --output ./my_dataset --max-samples 10000
"""

import argparse
from datasets import load_dataset
from pathlib import Path


def download_and_save(
    dataset_name: str,
    output_path: str,
    max_samples: int = None,
    split: str = 'train',
    streaming: bool = False,
):
    """
    Download a HuggingFace dataset and save it to disk.

    Args:
        dataset_name: Name of the dataset (e.g., 'wikipedia', 'c4')
        output_path: Path to save the dataset
        max_samples: Maximum number of samples to download (None = all)
        split: Dataset split to download
        streaming: Use streaming mode for download
    """
    print(f"Downloading {dataset_name}...")

    # Load dataset
    if dataset_name == 'wikipedia':
        dataset = load_dataset(
            'wikipedia',
            '20220301.en',
            split=split,
            streaming=streaming,
        )
    elif dataset_name == 'c4':
        dataset = load_dataset(
            'allenai/c4',
            'en',
            split=split,
            streaming=streaming,
        )
    elif dataset_name == 'bookcorpus':
        dataset = load_dataset(
            'bookcorpus',
            split=split,
            streaming=streaming,
        )
    else:
        # Generic dataset
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
        )

    # Limit samples if requested
    if max_samples is not None:
        print(f"Limiting to {max_samples} samples...")
        if streaming:
            dataset = dataset.take(max_samples)
        else:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Convert streaming to regular dataset if needed
    if streaming and max_samples is not None:
        print("Converting streaming dataset to regular dataset...")
        from datasets import Dataset
        data = list(dataset)
        dataset = Dataset.from_list(data)

    # Save to disk
    print(f"Saving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_path)

    print(f"✓ Dataset saved to: {output_path}")
    print(f"  Total samples: {len(dataset)}")
    print(f"\nTo use this dataset for training:")
    print(f"  python scripts/train_hf.py \\")
    print(f"      --dataset from-disk \\")
    print(f"      --dataset-path {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Download and save HuggingFace dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['wikipedia', 'c4', 'bookcorpus', 'custom'],
                        help='Dataset to download')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path to save dataset')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to download (None = all)')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to download')
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming mode for download')

    args = parser.parse_args()

    download_and_save(
        dataset_name=args.dataset,
        output_path=args.output,
        max_samples=args.max_samples,
        split=args.split,
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()
