# Training with Pre-Downloaded Datasets (Offline Mode)

Complete guide for working with pre-downloaded HuggingFace datasets.

## Why Use Pre-Downloaded Datasets?

✅ **Work offline** - Train without internet after initial download
✅ **Faster startup** - No download time, train immediately
✅ **Reusable** - Use same data for multiple experiments
✅ **Bandwidth-friendly** - Download once, use forever
✅ **Team sharing** - Share processed datasets
✅ **Google Colab** - Save to Drive, persist across sessions

## Quick Start

### 1. Download and Save Dataset

```bash
# Download 10K Wikipedia articles
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./datasets/wiki_10k \
    --max-samples 10000
```

### 2. Train on Saved Dataset

```bash
# Train (works offline!)
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_10k
```

That's it! The dataset is now saved locally and can be reused.

## Detailed Examples

### Example 1: Wikipedia Dataset

```bash
# Download
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./datasets/wikipedia_en_10k \
    --max-samples 10000

# Train
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wikipedia_en_10k \
    --config configs/base.yaml \
    --device cuda
```

### Example 2: C4 Dataset (Large)

```bash
# Download subset (50K samples)
python examples/download_and_save_dataset.py \
    --dataset c4 \
    --output ./datasets/c4_50k \
    --max-samples 50000 \
    --streaming

# Train with large-scale config
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/c4_50k \
    --config configs/large_scale.yaml \
    --device cuda
```

### Example 3: Multiple Experiments

Download once, experiment multiple times:

```bash
# Download once
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./datasets/wiki_data \
    --max-samples 20000

# Experiment 1: Small model
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_data \
    --config configs/base.yaml

# Experiment 2: Large model (same data!)
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_data \
    --config configs/large_scale.yaml

# Experiment 3: Different hyperparameters
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_data \
    --config configs/custom.yaml
```

## Google Colab Usage

Perfect for Colab notebooks where you want to persist data across sessions:

```python
# In Colab notebook

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Download and save to Drive (ONCE)
!python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output /content/drive/MyDrive/datasets/wiki_10k \
    --max-samples 10000

# 3. Train (can reuse in future sessions!)
!python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/datasets/wiki_10k \
    --device cuda

# 4. In future sessions, just run step 3 again!
# No need to re-download
```

**Benefits for Colab:**
- ✅ Survives session restarts
- ✅ Shared across notebooks
- ✅ No re-downloading
- ✅ Faster experimentation

## Programmatic Usage

For custom workflows:

```python
from datasets import load_dataset

# Download and save
dataset = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)
dataset = dataset.take(10000)  # Limit samples

# Convert to regular dataset
from datasets import Dataset
data = list(dataset)
dataset = Dataset.from_list(data)

# Save to disk
dataset.save_to_disk('./datasets/my_wiki_data')

# Later, load and use
from data import load_from_disk_dataset, SentenceJEPACollator
from torch.utils.data import DataLoader

dataset = load_from_disk_dataset(
    dataset_path='./datasets/my_wiki_data',
    min_sentences=3,
)

collator = SentenceJEPACollator(tokenizer_name='roberta-base')
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

for batch in dataloader:
    # Train
    pass
```

## Using Existing Downloaded Datasets

If you already have a dataset saved with `load_from_disk()`:

```python
# You already have this:
from datasets import load_dataset
dataset = load_dataset('wikipedia', '20220301.en')
dataset.save_to_disk('/path/to/my_dataset')

# Or loaded like this:
from datasets import load_from_disk
dataset = load_from_disk('/content/drive/MyDrive/SentenceJEPA')

# Just train with it!
```

```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA
```

## Advanced: Custom Processing

Pre-process and save:

```python
from datasets import load_dataset, Dataset

# Load dataset
dataset = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)

# Custom filtering/processing
def process_example(example):
    text = example['text']
    # Your custom processing
    text = text.lower()  # Example: lowercase
    text = clean_text(text)  # Your function
    return {'text': text}

# Apply processing
dataset = dataset.map(process_example)

# Take subset
dataset = dataset.take(50000)

# Save
processed_data = list(dataset)
final_dataset = Dataset.from_list(processed_data)
final_dataset.save_to_disk('./datasets/custom_processed')

# Train on processed data
```

```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/custom_processed
```

## Dataset Organization

Recommended structure:

```
project/
├── datasets/
│   ├── wiki_10k/           # Small Wikipedia
│   ├── wiki_100k/          # Medium Wikipedia
│   ├── c4_50k/             # C4 subset
│   ├── custom_processed/   # Your processed data
│   └── bookcorpus_20k/     # BookCorpus subset
├── scripts/
├── configs/
└── ...
```

## Disk Space Requirements

Approximate sizes:

| Dataset | Samples | Disk Space |
|---------|---------|------------|
| Wikipedia | 10K | ~500 MB |
| Wikipedia | 100K | ~5 GB |
| C4 | 10K | ~300 MB |
| C4 | 100K | ~3 GB |
| BookCorpus | 10K | ~800 MB |

**Tip:** Start with small samples (10K) to test, then scale up.

## Troubleshooting

### "Dataset not found at path"
**Check the path:**
```bash
ls -la /path/to/dataset
# Should see: dataset_info.json, data-*.arrow files
```

### "Text column not found"
**Check column names:**
```python
from datasets import load_from_disk
ds = load_from_disk('/path/to/dataset')
print(ds.column_names)  # ['text', 'title', 'id', ...]
```

**Use correct column:**
```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /path/to/dataset \
    --text-column content  # If not 'text'
```

### "Out of disk space"
**Use smaller subset:**
```bash
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./datasets/wiki_small \
    --max-samples 5000  # Reduce from 10000
```

### "Slow to load"
**Use streaming mode:**
```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /path/to/large_dataset \
    --streaming  # Treats on-disk data as streaming
```

## Comparison: Streaming vs Pre-Downloaded

| Feature | Streaming (online) | Pre-Downloaded (offline) |
|---------|-------------------|-------------------------|
| Internet required | Yes (during training) | No (after initial download) |
| Disk space | Minimal | Moderate |
| Startup time | Slower (downloading) | Fast (instant) |
| Reusability | Re-download each time | Reuse unlimited |
| Best for | Quick tests, exploration | Production, experiments |

## Tips

1. **Start with small samples** (10K) to verify everything works
2. **Save to external storage** (Drive, NAS) for team sharing
3. **Document your datasets** - Note how they were processed
4. **Version control** - Keep track of different dataset versions
5. **Compression** - Saved datasets are already compressed by HF

## Command Reference

**Download:**
```bash
python examples/download_and_save_dataset.py \
    --dataset {wikipedia|c4|bookcorpus|custom} \
    --output PATH \
    --max-samples N \
    [--streaming]
```

**Train:**
```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path PATH \
    [--text-column COLUMN] \
    [--config CONFIG] \
    [--device cuda]
```

## Summary

**Workflow:**
1. Download once: `download_and_save_dataset.py`
2. Train unlimited times: `train_hf.py --dataset from-disk`
3. Share with team: Copy dataset directory
4. Experiment freely: No bandwidth concerns!

**Perfect for:**
- 🎓 Students with limited bandwidth
- 💻 Offline development
- 🔬 Research requiring reproducibility
- 👥 Team collaborations
- ☁️ Cloud environments (Colab, Kaggle)

---

**See also:**
- [HuggingFace Guide](HUGGINGFACE_GUIDE.md) - Complete HF reference
- [Large Dataset Guide](LARGE_DATASET_GUIDE.md) - Scaling strategies
- [Cheatsheet](CHEATSHEET.md) - Quick commands
