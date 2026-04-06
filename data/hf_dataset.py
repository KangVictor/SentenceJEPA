"""
HuggingFace Dataset wrapper for Sentence JEPA.

Supports popular datasets like Wikipedia, C4, BookCorpus, etc.
"""

from torch.utils.data import IterableDataset, Dataset
from typing import Optional, List
import re


def split_into_sentences(text: str, use_spacy: bool = True) -> List[str]:
    """Split text into sentences (same as dataset.py)."""
    if use_spacy:
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            except OSError:
                use_spacy = False
        except ImportError:
            use_spacy = False

    if use_spacy:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
    else:
        # Fallback: simple regex-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


class HFParagraphDataset(IterableDataset):
    """
    Wrapper for HuggingFace datasets.

    Supports streaming mode for large datasets.
    Automatically extracts paragraphs and splits into sentences.
    """

    def __init__(
        self,
        dataset,
        text_column: str = 'text',
        min_sentences: int = 3,
        max_sentences: Optional[int] = 10,
        min_paragraph_length: int = 100,
        max_samples: Optional[int] = None,
        use_spacy: bool = False,
    ):
        """
        Args:
            dataset: HuggingFace dataset (can be streaming)
            text_column: Name of text column in dataset
            min_sentences: Minimum sentences per paragraph
            max_sentences: Maximum sentences to keep
            min_paragraph_length: Minimum characters in paragraph
            max_samples: Maximum number of samples to use (None = all)
            use_spacy: Use spaCy for sentence splitting
        """
        self.dataset = dataset
        self.text_column = text_column
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.min_paragraph_length = min_paragraph_length
        self.max_samples = max_samples
        self.use_spacy = use_spacy

    def __iter__(self):
        """Iterate over paragraphs from the dataset."""
        count = 0

        for item in self.dataset:
            if self.max_samples is not None and count >= self.max_samples:
                break

            # Extract text
            text = item[self.text_column]

            # Split into paragraphs (by double newline or other markers)
            paragraphs = self._split_into_paragraphs(text)

            for paragraph in paragraphs:
                # Filter short paragraphs
                if len(paragraph) < self.min_paragraph_length:
                    continue

                # Split into sentences
                sentences = split_into_sentences(paragraph, use_spacy=self.use_spacy)

                # Filter by sentence count
                if len(sentences) < self.min_sentences:
                    continue

                # Truncate if too many sentences
                if self.max_sentences is not None and len(sentences) > self.max_sentences:
                    sentences = sentences[:self.max_sentences]

                yield {
                    'paragraph': paragraph,
                    'sentences': sentences,
                }

                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    break

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newline
        paragraphs = text.split('\n\n')

        # Also try single newline if no double newlines found
        if len(paragraphs) == 1:
            paragraphs = text.split('\n')

        # Clean and filter
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs


class HFParagraphDatasetMapStyle(Dataset):
    """
    Map-style version of HuggingFace dataset wrapper.

    Use this when you want to load the entire dataset into memory
    and have random access. Better for smaller datasets.
    """

    def __init__(
        self,
        dataset,
        text_column: str = 'text',
        min_sentences: int = 3,
        max_sentences: Optional[int] = 10,
        min_paragraph_length: int = 100,
        use_spacy: bool = False,
    ):
        """
        Args:
            dataset: HuggingFace dataset (non-streaming)
            text_column: Name of text column
            min_sentences: Minimum sentences per paragraph
            max_sentences: Maximum sentences to keep
            min_paragraph_length: Minimum characters in paragraph
            use_spacy: Use spaCy for sentence splitting
        """
        self.text_column = text_column
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.min_paragraph_length = min_paragraph_length
        self.use_spacy = use_spacy

        # Process dataset into paragraphs
        print("Processing HuggingFace dataset into paragraphs...")
        self.data = []

        for item in dataset:
            text = item[text_column]
            paragraphs = self._split_into_paragraphs(text)

            for paragraph in paragraphs:
                if len(paragraph) < self.min_paragraph_length:
                    continue

                sentences = split_into_sentences(paragraph, use_spacy=use_spacy)

                if len(sentences) < self.min_sentences:
                    continue

                if self.max_sentences is not None and len(sentences) > self.max_sentences:
                    sentences = sentences[:self.max_sentences]

                self.data.append({
                    'paragraph': paragraph,
                    'sentences': sentences,
                })

        print(f"Processed {len(self.data)} paragraphs from HuggingFace dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = text.split('\n\n')
        if len(paragraphs) == 1:
            paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs


# Convenience functions for popular datasets

def load_wikipedia_dataset(
    language: str = 'en',
    date: str = '20220301',
    streaming: bool = True,
    min_sentences: int = 3,
    max_sentences: int = 10,
    max_samples: Optional[int] = None,
    split: str = 'train',
):
    """
    Load Wikipedia dataset.

    Args:
        language: Language code (e.g., 'en', 'es', 'fr')
        date: Wikipedia dump date (e.g., '20220301')
        streaming: Use streaming mode (recommended for large datasets)
        min_sentences: Minimum sentences per paragraph
        max_sentences: Maximum sentences per paragraph
        max_samples: Maximum samples to use (None = all)
        split: Dataset split ('train' for Wikipedia)

    Returns:
        HFParagraphDataset instance
    """
    from datasets import load_dataset

    print(f"Loading Wikipedia ({language}, {date}) dataset...")
    print(f"Streaming mode: {streaming}")

    dataset = load_dataset(
        'wikipedia',
        f'{date}.{language}',
        split=split,
        streaming=streaming,
    )

    if streaming:
        return HFParagraphDataset(
            dataset=dataset,
            text_column='text',
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            max_samples=max_samples,
        )
    else:
        return HFParagraphDatasetMapStyle(
            dataset=dataset,
            text_column='text',
            min_sentences=min_sentences,
            max_sentences=max_sentences,
        )


def load_c4_dataset(
    streaming: bool = True,
    min_sentences: int = 3,
    max_sentences: int = 10,
    max_samples: Optional[int] = None,
    split: str = 'train',
):
    """
    Load C4 (Colossal Clean Crawled Corpus) dataset.

    Args:
        streaming: Use streaming mode (highly recommended - C4 is huge!)
        min_sentences: Minimum sentences per paragraph
        max_sentences: Maximum sentences per paragraph
        max_samples: Maximum samples to use
        split: Dataset split ('train' or 'validation')

    Returns:
        HFParagraphDataset instance
    """
    from datasets import load_dataset

    print(f"Loading C4 dataset (split: {split})...")
    print(f"Streaming mode: {streaming}")

    dataset = load_dataset(
        'allenai/c4',
        'en',
        split=split,
        streaming=streaming,
    )

    if streaming:
        return HFParagraphDataset(
            dataset=dataset,
            text_column='text',
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            max_samples=max_samples,
        )
    else:
        return HFParagraphDatasetMapStyle(
            dataset=dataset,
            text_column='text',
            min_sentences=min_sentences,
            max_sentences=max_sentences,
        )


def load_bookcorpus_dataset(
    streaming: bool = True,
    min_sentences: int = 3,
    max_sentences: int = 10,
    max_samples: Optional[int] = None,
):
    """
    Load BookCorpus dataset.

    Args:
        streaming: Use streaming mode
        min_sentences: Minimum sentences per paragraph
        max_sentences: Maximum sentences per paragraph
        max_samples: Maximum samples to use

    Returns:
        HFParagraphDataset instance
    """
    from datasets import load_dataset

    print("Loading BookCorpus dataset...")
    print(f"Streaming mode: {streaming}")

    dataset = load_dataset(
        'bookcorpus',
        split='train',
        streaming=streaming,
    )

    if streaming:
        return HFParagraphDataset(
            dataset=dataset,
            text_column='text',
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            max_samples=max_samples,
        )
    else:
        return HFParagraphDatasetMapStyle(
            dataset=dataset,
            text_column='text',
            min_sentences=min_sentences,
            max_sentences=max_sentences,
        )


def load_from_disk_dataset(
    dataset_path: str,
    text_column: str = 'text',
    min_sentences: int = 3,
    max_sentences: Optional[int] = 10,
    max_samples: Optional[int] = None,
    use_streaming: bool = False,
):
    """
    Load a pre-downloaded HuggingFace dataset from disk.

    Use this when you've already downloaded a dataset with:
        dataset.save_to_disk("/path/to/dataset")

    Or when loading from:
        dataset = load_from_disk("/path/to/dataset")

    Args:
        dataset_path: Path to the saved dataset directory
        text_column: Name of text column
        min_sentences: Minimum sentences per paragraph
        max_sentences: Maximum sentences per paragraph
        max_samples: Maximum samples to use (None = all)
        use_streaming: Treat as streaming (useful for very large datasets)

    Returns:
        HFParagraphDataset or HFParagraphDatasetMapStyle instance
    """
    from datasets import load_from_disk

    print(f"Loading dataset from disk: {dataset_path}")
    hf_dataset = load_from_disk(dataset_path)

    # Check if it's a DatasetDict (has multiple splits)
    if hasattr(hf_dataset, 'keys'):
        print(f"\n  Detected DatasetDict with splits: {list(hf_dataset.keys())}")
        print(f"  Using 'train' split by default...")

        if 'train' in hf_dataset.keys():
            hf_dataset = hf_dataset['train']
        else:
            # Use first available split
            first_split = list(hf_dataset.keys())[0]
            print(f"  'train' split not found, using '{first_split}' instead")
            hf_dataset = hf_dataset[first_split]

    print(f"\nDataset info:")
    print(f"  Total samples: {len(hf_dataset) if hasattr(hf_dataset, '__len__') else 'Unknown (streaming)'}")

    # Debug: Check dataset structure
    if hasattr(hf_dataset, 'features'):
        print(f"  Features: {hf_dataset.features}")
    else:
        print(f"  Features: N/A")

    # Check if dataset has the text column
    if hasattr(hf_dataset, 'column_names'):
        print(f"  Column names: {hf_dataset.column_names}")
        if text_column not in hf_dataset.column_names:
            print(f"\n  WARNING: Column '{text_column}' not found!")
            print(f"  Available columns: {hf_dataset.column_names}")
            print(f"  Please use --text-column to specify the correct column name.")
            raise ValueError(f"Column '{text_column}' not found in dataset. Available: {hf_dataset.column_names}")

    # Special handling: If dataset appears to be a single element or has wrong structure
    if hasattr(hf_dataset, '__len__') and len(hf_dataset) == 1:
        print(f"\n  Note: Dataset has only 1 sample. Checking structure...")
        try:
            sample = hf_dataset[0]
            if isinstance(sample, str):
                print(f"  ERROR: Dataset contains strings, not dictionaries!")
                print(f"  The dataset at this path may not be in the correct format.")
                print(f"  Expected format: Dataset with dictionary items containing a '{text_column}' field")
                raise ValueError(f"Dataset has incorrect structure. Expected dict with '{text_column}' field, got string.")
            elif isinstance(sample, dict):
                if text_column not in sample:
                    print(f"  ERROR: Sample doesn't have '{text_column}' field!")
                    print(f"  Available fields: {list(sample.keys())}")
                    raise ValueError(f"Sample missing '{text_column}' field. Available: {list(sample.keys())}")
                else:
                    print(f"  Structure looks OK, but only 1 sample will be processed.")
        except Exception as e:
            if "incorrect structure" in str(e) or "missing" in str(e):
                raise
            print(f"  Could not verify structure: {e}")

    if use_streaming:
        # Treat as streaming even though it's on disk
        # Useful for very large datasets
        return HFParagraphDataset(
            dataset=hf_dataset,
            text_column=text_column,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            max_samples=max_samples,
        )
    else:
        # Load into memory (map-style)
        return HFParagraphDatasetMapStyle(
            dataset=hf_dataset,
            text_column=text_column,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
        )


if __name__ == "__main__":
    # Test with a small sample
    print("Testing HuggingFace dataset wrapper...")

    # Test with Wikipedia (small sample)
    dataset = load_wikipedia_dataset(
        streaming=True,
        max_samples=10,  # Just 10 samples for testing
    )

    print("\nFirst 3 items:")
    for i, item in enumerate(dataset):
        if i >= 3:
            break
        print(f"\nItem {i+1}:")
        print(f"  Num sentences: {len(item['sentences'])}")
        print(f"  First sentence: {item['sentences'][0][:80]}...")

    print("\n✓ HuggingFace dataset wrapper test passed!")
