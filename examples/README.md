# Examples

This directory contains example scripts and workflows for using Sentence JEPA.

## Download and Save Dataset

Download a HuggingFace dataset once and reuse it multiple times.

**Why?**
- Work offline after initial download
- Avoid re-downloading for multiple experiments
- Share datasets between projects
- Useful for Colab/limited bandwidth environments

**Example: Download Wikipedia**
```bash
# Download 10K Wikipedia articles
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./datasets/wiki_10k \
    --max-samples 10000

# Train on saved dataset
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_10k
```

**Example: Download C4 subset**
```bash
# Download 50K C4 samples
python examples/download_and_save_dataset.py \
    --dataset c4 \
    --output ./datasets/c4_50k \
    --max-samples 50000 \
    --streaming

# Train on saved dataset
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/c4_50k \
    --config configs/large_scale.yaml
```

## Programmatic Usage

```python
from datasets import load_from_disk
from data import load_from_disk_dataset, SentenceJEPACollator
from torch.utils.data import DataLoader

# Load from disk
dataset = load_from_disk_dataset(
    dataset_path='./datasets/wiki_10k',
    text_column='text',
    min_sentences=3,
    max_samples=None,  # Use all
)

# Create dataloader
collator = SentenceJEPACollator(tokenizer_name='roberta-base')
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

# Train
for batch in dataloader:
    # Your training code
    pass
```

## Google Colab Example

```python
# In Colab notebook

# 1. Download and save to Google Drive
!python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output /content/drive/MyDrive/datasets/wiki_data \
    --max-samples 10000

# 2. Train on saved dataset (reusable across sessions!)
!python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/datasets/wiki_data \
    --device cuda
```

## More Examples Coming Soon

- Custom preprocessing pipelines
- Multi-dataset training
- Fine-tuning workflows
- Evaluation examples
