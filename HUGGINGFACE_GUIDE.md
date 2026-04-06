# Training with HuggingFace Datasets

Complete guide for training Sentence JEPA on popular HuggingFace datasets.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Supported Datasets](#supported-datasets)
3. [Streaming vs Non-Streaming](#streaming-vs-non-streaming)
4. [Custom Datasets](#custom-datasets)
5. [Examples](#examples)
6. [Tips & Best Practices](#tips--best-practices)

---

## Quick Start

### Install Dependencies

```bash
source .venv/bin/activate
pip install datasets  # HuggingFace datasets library
```

### Train on Wikipedia (Small Sample)

```bash
python scripts/train_hf.py \
    --dataset wikipedia \
    --streaming \
    --max-samples 10000
```

### Train on C4 (Large Scale)

```bash
python scripts/train_hf.py \
    --dataset c4 \
    --streaming \
    --config configs/large_scale.yaml \
    --device cuda
```

---

## Supported Datasets

### 1. Wikipedia

**Description**: Wikipedia articles across many languages

**Size**: ~6M articles (English)

**Best for**: General knowledge, factual text

```bash
# English Wikipedia
python scripts/train_hf.py --dataset wikipedia --streaming

# Spanish Wikipedia
python scripts/train_hf.py \
    --dataset wikipedia \
    --wiki-lang es \
    --wiki-date 20220301 \
    --streaming

# Other languages: fr, de, zh, ja, etc.
```

**Pros:**
- High quality, well-structured text
- Multiple languages available
- Good for domain-general representations

**Cons:**
- Encyclopedic style (formal)
- May lack conversational language

### 2. C4 (Colossal Clean Crawled Corpus)

**Description**: Web pages from Common Crawl

**Size**: ~365GB (156B tokens)

**Best for**: Diverse web text, large-scale training

```bash
python scripts/train_hf.py \
    --dataset c4 \
    --streaming \
    --config configs/large_scale.yaml
```

**Pros:**
- Massive scale
- Diverse domains and styles
- Good for robust representations

**Cons:**
- Variable quality (web text)
- Requires more filtering
- Very large (use streaming!)

### 3. BookCorpus

**Description**: Collection of unpublished books

**Size**: ~5GB (11,038 books)

**Best for**: Long-form narrative text, coherent paragraphs

```bash
python scripts/train_hf.py \
    --dataset bookcorpus \
    --streaming
```

**Pros:**
- Coherent, well-written text
- Good paragraph structure
- Narrative flow

**Cons:**
- Smaller than C4/Wikipedia
- Fiction-heavy (may not generalize to all domains)

---

## Streaming vs Non-Streaming

### Streaming Mode (Recommended)

**Use when:**
- Dataset is large (>1GB)
- Limited memory
- Want to start training immediately

**Advantages:**
- No need to download entire dataset
- Constant memory usage
- Start training immediately

**Disadvantages:**
- Can't shuffle data (processes in order)
- No random access
- Can't compute exact dataset size

```bash
# Streaming example
python scripts/train_hf.py \
    --dataset wikipedia \
    --streaming \
    --max-samples 100000  # Process first 100k samples
```

### Non-Streaming Mode

**Use when:**
- Dataset is small (<1GB)
- Want data shuffling
- Need validation split
- Have enough memory

**Advantages:**
- Random access to samples
- Better shuffling
- Can split train/val easily
- Know exact size

**Disadvantages:**
- Must download entire dataset
- Higher memory usage
- Slower startup

```bash
# Non-streaming example (small dataset)
python scripts/train_hf.py \
    --dataset wikipedia \
    --max-samples 10000 \
    --val-split 0.1
```

---

## Custom Datasets

### Using Any HuggingFace Dataset

```bash
python scripts/train_hf.py \
    --dataset custom \
    --hf-name "username/dataset-name" \
    --text-column "content" \
    --streaming
```

### Using Pre-Downloaded Datasets

If you've already downloaded a dataset to disk:

```bash
# Step 1: Download and save once
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./datasets/wiki_10k \
    --max-samples 10000

# Step 2: Train (can reuse multiple times!)
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_10k
```

**Benefits:**
- ✅ Work offline
- ✅ Faster startup (no download time)
- ✅ Reuse across experiments
- ✅ Share with teammates
- ✅ Perfect for limited bandwidth

### Programmatic Usage

```python
from datasets import load_dataset
from data.hf_dataset import HFParagraphDataset
from data import SentenceJEPACollator
from torch.utils.data import DataLoader

# 1. Load HuggingFace dataset
hf_dataset = load_dataset('your/dataset', streaming=True, split='train')

# 2. Wrap with our paragraph dataset
dataset = HFParagraphDataset(
    dataset=hf_dataset,
    text_column='text',  # Column name with text
    min_sentences=3,
    max_sentences=10,
    max_samples=100000,  # Optional limit
)

# 3. Create dataloader
collator = SentenceJEPACollator(tokenizer_name='roberta-base')
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

# 4. Train as usual
for batch in dataloader:
    # Training code...
    pass
```

### Creating Your Own Dataset

Upload to HuggingFace Hub:

```python
from datasets import Dataset, DatasetDict

# Your data
data = {
    'text': [
        'First paragraph here...',
        'Second paragraph...',
        # ...
    ]
}

# Create dataset
dataset = Dataset.from_dict(data)

# Push to hub (requires authentication)
dataset.push_to_hub('your-username/your-dataset')

# Then train with:
# python scripts/train_hf.py --dataset custom --hf-name "your-username/your-dataset"
```

---

## Examples

### Example 1: Wikipedia English (Quick Test)

Train on 10K Wikipedia articles to test setup:

```bash
python scripts/train_hf.py \
    --dataset wikipedia \
    --streaming \
    --max-samples 10000 \
    --config configs/base.yaml \
    --device cuda
```

**Expected time**: ~30 minutes on GPU

### Example 2: Wikipedia Full Scale

Train on full English Wikipedia:

```bash
python scripts/train_hf.py \
    --dataset wikipedia \
    --streaming \
    --config configs/large_scale.yaml \
    --device cuda
```

**Expected time**: ~10-15 hours on single GPU

**Tips:**
- Use multiple GPUs if available
- Monitor checkpoints every 1000 steps
- Watch for convergence after ~50K samples

### Example 3: C4 Large Scale

Train on C4 for maximum diversity:

```bash
python scripts/train_hf.py \
    --dataset c4 \
    --streaming \
    --max-samples 500000 \
    --config configs/large_scale.yaml \
    --device cuda
```

**Expected time**: ~40 hours on single GPU for 500K samples

**Tips:**
- C4 is huge - use `--max-samples` to limit
- Increase batch size if you have GPU memory
- Use gradient accumulation for effective larger batches

### Example 4: Multilingual Wikipedia

Train on Spanish Wikipedia:

```bash
python scripts/train_hf.py \
    --dataset wikipedia \
    --wiki-lang es \
    --streaming \
    --max-samples 50000 \
    --config configs/base.yaml
```

Available languages: en, es, fr, de, it, pt, ru, ja, zh, ko, ar, etc.

### Example 5: Custom Dataset

```bash
# Example: OpenWebText
python scripts/train_hf.py \
    --dataset custom \
    --hf-name "openwebtext" \
    --streaming \
    --config configs/large_scale.yaml
```

### Example 6: From Pre-Downloaded Dataset

Work offline or reuse downloaded datasets:

```bash
# Step 1: Download once (requires internet)
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./datasets/wiki_10k \
    --max-samples 10000

# Step 2: Train (works offline!)
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_10k

# Step 3: Train again with different config (no re-download!)
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_10k \
    --config configs/large_scale.yaml
```

**Perfect for:**
- Google Colab (save to Drive, reuse across sessions)
- Limited bandwidth environments
- Offline development
- Sharing datasets with team

### Example 7: Mixed Datasets

Train on multiple datasets sequentially:

```bash
# Phase 1: BookCorpus (narrative)
python scripts/train_hf.py \
    --dataset bookcorpus \
    --streaming \
    --max-samples 50000 \
    --config configs/base.yaml

# Phase 2: Continue with Wikipedia (factual)
python scripts/train_hf.py \
    --dataset wikipedia \
    --streaming \
    --max-samples 50000 \
    --config configs/base.yaml
    # Add: --resume-from checkpoints/checkpoint_step_XXX.pt

# Phase 3: Fine-tune on C4 (diverse)
python scripts/train_hf.py \
    --dataset c4 \
    --streaming \
    --max-samples 100000 \
    --config configs/base.yaml
    # Add: --resume-from checkpoints/checkpoint_step_YYY.pt
```

---

## Tips & Best Practices

### 1. Start Small, Scale Up

```bash
# Step 1: Test with 1K samples
python scripts/train_hf.py --dataset wikipedia --streaming --max-samples 1000

# Step 2: If working, try 10K
python scripts/train_hf.py --dataset wikipedia --streaming --max-samples 10000

# Step 3: Scale to full dataset
python scripts/train_hf.py --dataset wikipedia --streaming
```

### 2. Monitor Quality

Check samples during training to ensure quality:

```python
from data.hf_dataset import load_wikipedia_dataset

dataset = load_wikipedia_dataset(streaming=True, max_samples=10)

for i, item in enumerate(dataset):
    print(f"\nSample {i+1}:")
    print(f"Sentences: {len(item['sentences'])}")
    print(f"First: {item['sentences'][0]}")
    if i >= 5:
        break
```

### 3. Optimize Hyperparameters

For large HF datasets, adjust config:

```yaml
# configs/hf_large.yaml
training:
  batch_size: 64
  num_epochs: 2  # Fewer epochs with more data
  warmup_steps: 5000  # Longer warmup

data:
  max_sentences: 8  # Slightly fewer for longer docs
  max_tokens_per_sentence: 64

loss:
  lambda_sigreg: 0.15  # Slightly more regularization
```

### 4. Use Checkpoints Frequently

For long training runs:

```yaml
training:
  save_every: 500   # Save every 500 steps
  eval_every: 2000  # Eval every 2000 steps (if using val)
```

### 5. GPU Memory Management

If OOM (Out of Memory):

```yaml
training:
  batch_size: 16  # Reduce from 64
  # Or add gradient accumulation in trainer
```

### 6. Streaming Best Practices

When using streaming mode:

1. **Can't shuffle**: Data processes in order
   - Solution: HF datasets often pre-shuffled
   - Or: Use `dataset.shuffle(seed=42, buffer_size=10000)`

2. **Can't split train/val**: No random access
   - Solution: Create separate validation dataset
   ```bash
   # Train on train split
   python scripts/train_hf.py --dataset c4 --split train --streaming

   # Eval on validation split (separate run)
   python scripts/train_hf.py --dataset c4 --split validation --streaming --max-samples 1000
   ```

3. **Unknown dataset size**: Can't compute epochs exactly
   - Solution: Use `--max-samples` to control amount
   - Or: Let it run for fixed time/steps

### 7. Multi-Dataset Training

Combine datasets for diversity:

```python
from datasets import interleave_datasets, load_dataset

# Load multiple datasets
wiki = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)
c4 = load_dataset('allenai/c4', 'en', split='train', streaming=True)

# Interleave them
combined = interleave_datasets([wiki, c4], probabilities=[0.5, 0.5])

# Wrap and train
from data.hf_dataset import HFParagraphDataset
dataset = HFParagraphDataset(combined, ...)
```

---

## Troubleshooting

### Issue: "Dataset not found"
**Solution**: Install datasets library
```bash
pip install datasets
```

### Issue: "Connection timeout"
**Solution**: HF datasets download from cloud. Check internet connection or:
```bash
export HF_DATASETS_OFFLINE=1  # Use cached version
```

### Issue: "Too slow to load"
**Solution**: Use streaming mode
```bash
python scripts/train_hf.py --dataset wikipedia --streaming
```

### Issue: "Low quality paragraphs"
**Solution**: Adjust filtering parameters
```python
dataset = HFParagraphDataset(
    ...,
    min_paragraph_length=200,  # Increase min length
    min_sentences=4,            # Require more sentences
)
```

### Issue: "Training diverges"
**Solution**:
- Increase warmup steps for large datasets
- Reduce learning rate
- Check data quality (print samples)

---

## Performance Comparison

Training speed on different datasets (single RTX 3090):

| Dataset | Streaming | Samples/sec | Time for 100K |
|---------|-----------|-------------|---------------|
| Wikipedia | Yes | ~150 | ~11 min |
| Wikipedia | No | ~200 | ~8 min |
| C4 | Yes | ~100 | ~17 min |
| BookCorpus | Yes | ~180 | ~9 min |

*Streaming is slightly slower but uses constant memory*

---

## Advanced: Custom Processing

Add custom preprocessing:

```python
from data.hf_dataset import HFParagraphDataset

class CustomHFDataset(HFParagraphDataset):
    def __iter__(self):
        for item in super().__iter__():
            # Custom processing
            paragraph = item['paragraph']

            # Example: Remove URLs
            import re
            paragraph = re.sub(r'http\S+', '', paragraph)

            # Example: Filter by length
            if len(paragraph.split()) < 50:
                continue

            # Update sentences
            sentences = split_into_sentences(paragraph)
            item['sentences'] = sentences

            yield item

# Use custom dataset
dataset = CustomHFDataset(hf_dataset, ...)
```

---

## Summary

**Quick Reference:**

| Use Case | Command |
|----------|---------|
| Quick test | `--dataset wikipedia --streaming --max-samples 10000` |
| Full Wikipedia | `--dataset wikipedia --streaming` |
| Large scale | `--dataset c4 --streaming --config configs/large_scale.yaml` |
| Multilingual | `--dataset wikipedia --wiki-lang es --streaming` |
| Custom dataset | `--dataset custom --hf-name "user/dataset"` |

**Best Practices:**
1. ✅ Use streaming for large datasets
2. ✅ Start with `--max-samples` to test
3. ✅ Monitor first few batches for quality
4. ✅ Save checkpoints frequently
5. ✅ Use GPU for faster training

For more details, see:
- [Main README](README.md)
- [Large Dataset Guide](LARGE_DATASET_GUIDE.md)
- [HuggingFace Datasets Docs](https://huggingface.co/docs/datasets)
