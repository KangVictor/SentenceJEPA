"""
Preprocess a HuggingFace dataset once and save it for fast reuse.

This script:
1. Loads your dataset
2. Splits text into sentences
3. Filters by sentence count
4. Saves processed data in pickle format for instant loading

Usage:
    python scripts/preprocess_dataset.py \
        --input /content/drive/MyDrive/SentenceJEPA \
        --output /content/drive/MyDrive/SentenceJEPA_processed \
        --max-samples 100000
"""

import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.hf_dataset import split_into_sentences


def preprocess_dataset(
    input_path: str,
    output_path: str,
    text_column: str = 'text',
    min_sentences: int = 3,
    max_sentences: int = 10,
    min_paragraph_length: int = 100,
    max_samples: int = None,
    use_spacy: bool = False,
):
    """
    Preprocess a dataset and save for fast loading.

    Args:
        input_path: Path to input dataset
        output_path: Path to save processed data
        text_column: Column containing text
        min_sentences: Minimum sentences per paragraph
        max_sentences: Maximum sentences to keep
        min_paragraph_length: Minimum paragraph length
        max_samples: Maximum samples to process (None = all)
        use_spacy: Use spaCy for sentence splitting
    """
    print("="*60)
    print("Dataset Preprocessing")
    print("="*60)

    # Load dataset
    print(f"\nLoading dataset from: {input_path}")
    dataset = load_from_disk(input_path)

    # Handle DatasetDict
    if hasattr(dataset, 'keys'):
        print(f"Detected DatasetDict with splits: {list(dataset.keys())}")
        if 'train' in dataset.keys():
            dataset = dataset['train']
            print(f"Using 'train' split")
        else:
            first_split = list(dataset.keys())[0]
            dataset = dataset[first_split]
            print(f"Using '{first_split}' split")

    total_samples = len(dataset)
    print(f"Total samples in dataset: {total_samples:,}")

    if max_samples:
        total_samples = min(total_samples, max_samples)
        print(f"Will process first {total_samples:,} samples")

    # Check column exists
    if hasattr(dataset, 'column_names'):
        if text_column not in dataset.column_names:
            print(f"\nERROR: Column '{text_column}' not found!")
            print(f"Available columns: {dataset.column_names}")
            return
        print(f"Using text column: '{text_column}'")

    # Process dataset
    print(f"\nProcessing dataset...")
    print(f"  Min sentences: {min_sentences}")
    print(f"  Max sentences: {max_sentences}")
    print(f"  Min paragraph length: {min_paragraph_length}")
    print(f"  Use spaCy: {use_spacy}")

    processed_data = []
    filtered_too_short = 0
    filtered_too_few_sentences = 0

    for i in tqdm(range(total_samples), desc="Processing"):
        try:
            item = dataset[i]
            text = item[text_column]

            # Split into paragraphs
            paragraphs = text.split('\n\n')
            if len(paragraphs) == 1:
                paragraphs = text.split('\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

            # Process each paragraph
            for paragraph in paragraphs:
                # Filter short paragraphs
                if len(paragraph) < min_paragraph_length:
                    filtered_too_short += 1
                    continue

                # Split into sentences
                sentences = split_into_sentences(paragraph, use_spacy=use_spacy)

                # Filter by sentence count
                if len(sentences) < min_sentences:
                    filtered_too_few_sentences += 1
                    continue

                # Truncate if too many
                if max_sentences and len(sentences) > max_sentences:
                    sentences = sentences[:max_sentences]

                # Add to processed data
                processed_data.append({
                    'paragraph': paragraph,
                    'sentences': sentences,
                })

        except Exception as e:
            print(f"\nWarning: Error processing sample {i}: {e}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print("Processing Summary")
    print("="*60)
    print(f"Samples processed: {total_samples:,}")
    print(f"Paragraphs extracted: {len(processed_data):,}")
    print(f"Filtered (too short): {filtered_too_short:,}")
    print(f"Filtered (too few sentences): {filtered_too_few_sentences:,}")
    print(f"Final dataset size: {len(processed_data):,} paragraphs")

    if len(processed_data) == 0:
        print("\nWARNING: No paragraphs remained after filtering!")
        print("Consider lowering min_sentences or min_paragraph_length")
        return

    # Save processed data
    print(f"\nSaving processed data to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)

    # Save metadata
    metadata = {
        'input_path': input_path,
        'text_column': text_column,
        'min_sentences': min_sentences,
        'max_sentences': max_sentences,
        'min_paragraph_length': min_paragraph_length,
        'total_samples_processed': total_samples,
        'num_paragraphs': len(processed_data),
        'use_spacy': use_spacy,
    }

    metadata_path = str(output_path) + '.metadata'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"✓ Processed data saved")
    print(f"✓ Metadata saved to: {metadata_path}")

    # Show file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"\nFile size: {file_size:.2f} MB")

    print(f"\n{'='*60}")
    print("To use this preprocessed dataset:")
    print("="*60)
    print(f"python scripts/train_hf.py \\")
    print(f"    --dataset preprocessed \\")
    print(f"    --dataset-path {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset for fast loading')

    parser.add_argument('--input', type=str, required=True,
                        help='Path to input dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save processed data')

    parser.add_argument('--text-column', type=str, default='text',
                        help='Column containing text')
    parser.add_argument('--min-sentences', type=int, default=3,
                        help='Minimum sentences per paragraph')
    parser.add_argument('--max-sentences', type=int, default=10,
                        help='Maximum sentences per paragraph')
    parser.add_argument('--min-paragraph-length', type=int, default=100,
                        help='Minimum paragraph length')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to process')
    parser.add_argument('--use-spacy', action='store_true',
                        help='Use spaCy for sentence splitting')

    args = parser.parse_args()

    preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
        text_column=args.text_column,
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        min_paragraph_length=args.min_paragraph_length,
        max_samples=args.max_samples,
        use_spacy=args.use_spacy,
    )


if __name__ == "__main__":
    main()
