"""
Collator for batching and tokenizing paragraphs with sentence masking.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from typing import List, Dict
import random


class SentenceJEPACollator:
    """
    Collator for Sentence JEPA training.

    Handles:
    - Tokenizing sentences
    - Padding to batch
    - Selecting which sentence to mask
    - Creating appropriate masks
    """

    def __init__(
        self,
        tokenizer_name: str = "roberta-base",
        max_tokens_per_sentence: int = 64,
        prefer_interior_mask: bool = True,
        interior_prob: float = 0.8,
        mask_ratio: float = None,
    ):
        """
        Args:
            tokenizer_name: HuggingFace tokenizer name
            max_tokens_per_sentence: Maximum tokens per sentence
            prefer_interior_mask: Prefer masking interior sentences over edges
            interior_prob: Probability of masking interior vs edge sentence
            mask_ratio: Fraction of sentences to mask (e.g., 0.3 = 30%).
                       If None, masks only 1 sentence per paragraph (original behavior).
                       If set, expands batch by creating multiple training examples.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens_per_sentence = max_tokens_per_sentence
        self.prefer_interior_mask = prefer_interior_mask
        self.interior_prob = interior_prob
        self.mask_ratio = mask_ratio

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of paragraphs.

        Args:
            batch: List of items from ParagraphDataset
                Each item: {'paragraph': str, 'sentences': List[str]}

        Returns:
            collated: {
                'input_ids': [B', S, T] - token IDs (B' may be > B if mask_ratio is set)
                'attention_mask': [B', S, T] - token attention mask
                'sentence_mask': [B', S] - sentence validity mask
                'mask_idx': [B'] - index of masked sentence per batch
            }
        """
        # Expand batch if mask_ratio is set
        if self.mask_ratio is not None:
            import math
            expanded_batch = []
            for item in batch:
                n_sents = len(item['sentences'])
                # Calculate how many sentences to mask (use ceiling to get at least mask_ratio%)
                num_to_mask = max(1, math.ceil(n_sents * self.mask_ratio))

                # Select which sentences to mask
                mask_candidates = self._select_multiple_mask_indices(n_sents, num_to_mask)

                # Create one training example for each masked sentence
                for mask_idx in mask_candidates:
                    expanded_batch.append({
                        'sentences': item['sentences'],
                        'paragraph': item['paragraph'],
                        '_mask_idx': mask_idx,  # Pre-selected mask index
                    })
            batch = expanded_batch

        batch_size = len(batch)

        # Get sentences for each item
        all_sentences = [item['sentences'] for item in batch]
        num_sentences = [len(sents) for sents in all_sentences]
        max_sentences = max(num_sentences)

        # Tokenize all sentences
        tokenized_batch = []
        for sentences in all_sentences:
            tokenized_sentences = []
            for sent in sentences:
                # Tokenize sentence
                tokens = self.tokenizer(
                    sent,
                    max_length=self.max_tokens_per_sentence,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )
                tokenized_sentences.append({
                    'input_ids': tokens['input_ids'],
                    'attention_mask': tokens['attention_mask'],
                })
            tokenized_batch.append(tokenized_sentences)

        # Select masked sentence index for each item
        mask_indices = []
        for i, n_sents in enumerate(num_sentences):
            # Use pre-selected mask if available (from mask_ratio expansion)
            if '_mask_idx' in batch[i]:
                mask_idx = batch[i]['_mask_idx']
            else:
                mask_idx = self._select_mask_index(n_sents)
            mask_indices.append(mask_idx)

        # Pad and collate
        # First, collect all tokenized data and find global max_tokens
        all_input_ids = []
        all_attention_masks = []
        all_sentence_masks = []

        for i in range(batch_size):
            n_sents = num_sentences[i]
            tokenized_sents = tokenized_batch[i]

            # Pad sentences to max_sentences
            input_ids_padded = []
            attention_mask_padded = []
            sentence_mask = []

            for j in range(max_sentences):
                if j < n_sents:
                    # Real sentence
                    input_ids_padded.append(torch.tensor(tokenized_sents[j]['input_ids']))
                    attention_mask_padded.append(torch.tensor(tokenized_sents[j]['attention_mask']))
                    sentence_mask.append(1)
                else:
                    # Padding sentence
                    # Use pad_token_id (usually 1 for RoBERTa)
                    pad_id = self.tokenizer.pad_token_id
                    input_ids_padded.append(torch.tensor([pad_id]))
                    attention_mask_padded.append(torch.tensor([0]))
                    sentence_mask.append(0)

            all_input_ids.append(input_ids_padded)
            all_attention_masks.append(attention_mask_padded)
            all_sentence_masks.append(sentence_mask)

        # Find global max_tokens across entire batch
        max_tokens = 0
        for input_ids_list in all_input_ids:
            for ids in input_ids_list:
                max_tokens = max(max_tokens, len(ids))
        max_tokens = min(max_tokens, self.max_tokens_per_sentence)

        # Now pad everything to the same max_tokens
        input_ids_batch = []
        attention_mask_batch = []
        sentence_mask_batch = []

        for i in range(batch_size):
            input_ids_padded_uniform = []
            attention_mask_padded_uniform = []

            for ids, mask in zip(all_input_ids[i], all_attention_masks[i]):
                # Truncate if needed
                ids = ids[:max_tokens]
                mask = mask[:max_tokens]

                # Pad to max_tokens
                pad_length = max_tokens - len(ids)
                if pad_length > 0:
                    pad_id = self.tokenizer.pad_token_id
                    ids = torch.cat([ids, torch.full((pad_length,), pad_id, dtype=ids.dtype)])
                    mask = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])

                input_ids_padded_uniform.append(ids)
                attention_mask_padded_uniform.append(mask)

            # Stack to [S, T]
            input_ids_item = torch.stack(input_ids_padded_uniform)  # [S, T]
            attention_mask_item = torch.stack(attention_mask_padded_uniform)  # [S, T]
            sentence_mask_item = torch.tensor(all_sentence_masks[i])  # [S]

            input_ids_batch.append(input_ids_item)
            attention_mask_batch.append(attention_mask_item)
            sentence_mask_batch.append(sentence_mask_item)

        # Stack to batch: [B, S, T]
        input_ids = torch.stack(input_ids_batch)
        attention_mask = torch.stack(attention_mask_batch)
        sentence_mask = torch.stack(sentence_mask_batch)
        mask_idx = torch.tensor(mask_indices)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentence_mask': sentence_mask,
            'mask_idx': mask_idx,
        }

    def _select_mask_index(self, num_sentences: int) -> int:
        """
        Select which sentence to mask.

        Args:
            num_sentences: Number of sentences in paragraph

        Returns:
            mask_idx: Index of sentence to mask (0-indexed)
        """
        if num_sentences <= 2:
            # If only 2 sentences, mask randomly
            return random.randint(0, num_sentences - 1)

        if self.prefer_interior_mask and random.random() < self.interior_prob:
            # Mask interior sentence (not first or last)
            return random.randint(1, num_sentences - 2)
        else:
            # Mask any sentence
            return random.randint(0, num_sentences - 1)

    def _select_multiple_mask_indices(self, num_sentences: int, num_to_mask: int) -> List[int]:
        """
        Select multiple sentences to mask.

        Args:
            num_sentences: Number of sentences in paragraph
            num_to_mask: Number of sentences to mask

        Returns:
            mask_indices: List of indices to mask
        """
        if num_sentences <= 2:
            # If only 2 sentences, mask one randomly
            return [random.randint(0, num_sentences - 1)]

        # Ensure we don't try to mask more sentences than available
        num_to_mask = min(num_to_mask, num_sentences)

        if self.prefer_interior_mask:
            # Prefer masking interior sentences
            interior_indices = list(range(1, num_sentences - 1))
            edge_indices = [0, num_sentences - 1]

            # Calculate how many interior vs edge to mask
            num_interior_available = len(interior_indices)
            num_interior_to_mask = min(num_to_mask, num_interior_available)

            # Randomly sample interior sentences
            if num_interior_to_mask > 0:
                selected_interior = random.sample(interior_indices, num_interior_to_mask)
            else:
                selected_interior = []

            # If we need more, sample from edges
            num_edge_to_mask = num_to_mask - num_interior_to_mask
            if num_edge_to_mask > 0:
                selected_edges = random.sample(edge_indices, min(num_edge_to_mask, len(edge_indices)))
            else:
                selected_edges = []

            return sorted(selected_interior + selected_edges)
        else:
            # Randomly sample any sentences
            all_indices = list(range(num_sentences))
            return sorted(random.sample(all_indices, num_to_mask))


if __name__ == "__main__":
    # Test collator
    print("Testing SentenceJEPACollator...")

    from data.dataset import ParagraphDataset

    # Create sample dataset
    paragraphs = [
        "This is the first sentence. Here is another one. And a third sentence!",
        "Machine learning is fascinating. It involves training models on data. The models learn patterns. Then they can make predictions.",
        "Another example paragraph. With multiple sentences here. Testing the dataset functionality.",
    ]

    dataset = ParagraphDataset.from_list(
        paragraphs=paragraphs,
        min_sentences=3,
        use_spacy=False,
    )

    # Create collator
    collator = SentenceJEPACollator(
        tokenizer_name="roberta-base",
        max_tokens_per_sentence=32,
        prefer_interior_mask=True,
    )

    # Create batch
    batch = [dataset[i] for i in range(len(dataset))]
    collated = collator(batch)

    print(f"Batch size: {len(batch)}")
    print(f"input_ids shape: {collated['input_ids'].shape}")
    print(f"attention_mask shape: {collated['attention_mask'].shape}")
    print(f"sentence_mask shape: {collated['sentence_mask'].shape}")
    print(f"mask_idx shape: {collated['mask_idx'].shape}")
    print(f"mask_idx values: {collated['mask_idx']}")

    print("\n✓ SentenceJEPACollator test passed!")
