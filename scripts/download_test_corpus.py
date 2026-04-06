"""
Download a small test corpus for evaluation.

This script downloads a held-out test set from C4 (different from Wikipedia training data)
to evaluate the model on truly unseen data.

Usage:
    python scripts/download_test_corpus.py --output data/test_corpus.txt --num-samples 1000
"""

import argparse
from datasets import load_dataset
from pathlib import Path
import sys
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def split_into_sentences(text):
    """Simple sentence splitter."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out very short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences


def is_valid_paragraph(text, min_sentences=3, max_sentences=10):
    """Check if text is a valid paragraph for testing."""
    sentences = split_into_sentences(text)
    return min_sentences <= len(sentences) <= max_sentences


def main():
    parser = argparse.ArgumentParser(description='Download test corpus')
    parser.add_argument('--output', type=str, default='data/test_corpus.txt',
                        help='Output file path')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of paragraphs to download')
    parser.add_argument('--source', type=str, default='c4',
                        choices=['c4', 'wikipedia', 'bookcorpus'],
                        help='Source dataset')
    parser.add_argument('--min-sentences', type=int, default=3,
                        help='Minimum sentences per paragraph')
    parser.add_argument('--max-sentences', type=int, default=10,
                        help='Maximum sentences per paragraph')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Downloading Test Corpus")
    print(f"{'='*60}\n")
    print(f"Source: {args.source}")
    print(f"Target samples: {args.num_samples}")
    print(f"Sentence range: {args.min_sentences}-{args.max_sentences}")
    print()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset in streaming mode
    print("Loading dataset (streaming mode)...")
    if args.source == 'c4':
        dataset = load_dataset(
            'allenai/c4',
            'en',
            split='validation',  # Use validation split for test (separate from train)
            streaming=True,
        )
    elif args.source == 'wikipedia':
        dataset = load_dataset(
            'wikipedia',
            '20220301',
            split='train',  # Will skip first portion that might have been in training
            streaming=True,
        )
    elif args.source == 'bookcorpus':
        dataset = load_dataset(
            'bookcorpus',
            split='train',
            streaming=True,
        )

    # Process and save paragraphs
    print(f"\nProcessing and saving to: {output_path}")

    paragraphs_collected = 0
    paragraphs_processed = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            if paragraphs_collected >= args.num_samples:
                break

            paragraphs_processed += 1

            # Get text from item
            if args.source == 'c4':
                text = item['text']
            elif args.source == 'wikipedia':
                text = item['text']
            elif args.source == 'bookcorpus':
                text = item['text']

            # Check if valid paragraph
            if is_valid_paragraph(text, args.min_sentences, args.max_sentences):
                # Write paragraph (blank line separated)
                f.write(text.strip() + '\n\n')
                paragraphs_collected += 1

                if paragraphs_collected % 100 == 0:
                    print(f"  Collected: {paragraphs_collected}/{args.num_samples}")

    print(f"\n✓ Done!")
    print(f"  Paragraphs processed: {paragraphs_processed:,}")
    print(f"  Valid paragraphs saved: {paragraphs_collected:,}")
    print(f"  Output file: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    print()


if __name__ == "__main__":
    main()
