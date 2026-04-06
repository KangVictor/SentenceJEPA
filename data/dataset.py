"""
Dataset for loading and preprocessing paragraphs.
Splits paragraphs into sentences using spaCy or simple heuristics.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Optional
import re


def split_into_sentences(text: str, use_spacy: bool = True) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Input paragraph text
        use_spacy: If True, use spaCy; otherwise use simple regex

    Returns:
        sentences: List of sentence strings
    """
    if use_spacy:
        try:
            import spacy
            # Try to load spaCy model
            try:
                nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            except OSError:
                print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                print("Falling back to regex sentence splitting...")
                use_spacy = False
        except ImportError:
            print("Warning: spaCy not installed. Falling back to regex sentence splitting...")
            use_spacy = False

    if use_spacy:
        # Use spaCy sentencizer
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
    else:
        # Fallback: simple regex-based splitting
        # Split on period, exclamation, question mark followed by space/newline
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


class ParagraphDataset(Dataset):
    """
    Dataset that loads paragraphs and splits them into sentences.

    Each item is a paragraph with its sentences.
    Filters out paragraphs with too few sentences.
    """

    def __init__(
        self,
        paragraphs: List[str],
        min_sentences: int = 3,
        max_sentences: Optional[int] = 10,
        use_spacy: bool = True,
    ):
        """
        Args:
            paragraphs: List of paragraph texts
            min_sentences: Minimum number of sentences per paragraph
            max_sentences: Maximum number of sentences to keep (truncate if more)
            use_spacy: Whether to use spaCy for sentence splitting
        """
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.use_spacy = use_spacy

        # Process and filter paragraphs
        self.data = []
        for para in paragraphs:
            sentences = split_into_sentences(para, use_spacy=use_spacy)

            # Filter by minimum sentences
            if len(sentences) < min_sentences:
                continue

            # Truncate if too many sentences
            if max_sentences is not None and len(sentences) > max_sentences:
                sentences = sentences[:max_sentences]

            self.data.append({
                'paragraph': para,
                'sentences': sentences,
            })

        print(f"Loaded {len(self.data)} paragraphs (filtered from {len(paragraphs)})")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single paragraph with its sentences.

        Returns:
            item: {
                'paragraph': str,
                'sentences': List[str],
            }
        """
        return self.data[idx]

    @classmethod
    def from_text_file(
        cls,
        file_path: str,
        min_sentences: int = 3,
        max_sentences: Optional[int] = 10,
        use_spacy: bool = True,
        paragraph_separator: str = "\n\n",
    ):
        """
        Load paragraphs from a text file.

        Args:
            file_path: Path to text file
            min_sentences: Minimum sentences per paragraph
            max_sentences: Maximum sentences per paragraph
            use_spacy: Use spaCy for sentence splitting
            paragraph_separator: Separator between paragraphs

        Returns:
            dataset: ParagraphDataset instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into paragraphs
        paragraphs = text.split(paragraph_separator)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return cls(
            paragraphs=paragraphs,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            use_spacy=use_spacy,
        )

    @classmethod
    def from_list(
        cls,
        paragraphs: List[str],
        min_sentences: int = 3,
        max_sentences: Optional[int] = 10,
        use_spacy: bool = True,
    ):
        """
        Create dataset from list of paragraphs.

        Args:
            paragraphs: List of paragraph strings
            min_sentences: Minimum sentences per paragraph
            max_sentences: Maximum sentences per paragraph
            use_spacy: Use spaCy for sentence splitting

        Returns:
            dataset: ParagraphDataset instance
        """
        return cls(
            paragraphs=paragraphs,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            use_spacy=use_spacy,
        )


if __name__ == "__main__":
    # Test dataset
    print("Testing ParagraphDataset...")

    # Sample paragraphs
    paragraphs = [
        "This is the first sentence. Here is another one. And a third sentence!",
        "Short para.",  # Will be filtered out
        "Machine learning is fascinating. It involves training models on data. The models learn patterns. Then they can make predictions.",
        "Another example paragraph. With multiple sentences here. Testing the dataset functionality.",
    ]

    dataset = ParagraphDataset.from_list(
        paragraphs=paragraphs,
        min_sentences=3,
        max_sentences=10,
        use_spacy=False,  # Use regex for testing
    )

    print(f"Dataset size: {len(dataset)}")

    # Print first item
    item = dataset[0]
    print(f"\nFirst item:")
    print(f"  Paragraph: {item['paragraph'][:50]}...")
    print(f"  Sentences: {item['sentences']}")
    print(f"  Num sentences: {len(item['sentences'])}")

    print("\n✓ ParagraphDataset test passed!")
