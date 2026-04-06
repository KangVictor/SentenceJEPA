# Pre-Downloaded Dataset Support - Complete Guide

✅ **Feature Added:** You can now train on pre-downloaded HuggingFace datasets!

## What's New

Support for loading datasets that you've already downloaded and saved to disk, including datasets saved with `load_from_disk()`.

### Before (Only Online)
```bash
# Had to download every time
python scripts/train_hf.py --dataset wikipedia --streaming
```

### After (Offline Supported!)
```bash
# Download once
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./datasets/wiki_data \
    --max-samples 10000

# Train unlimited times (no re-download!)
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_data
```

## Quick Start

### Step 1: Install datasets library (if not already installed)

```bash
source .venv/bin/activate
pip install datasets
```

### Step 2: Download and save a dataset

```bash
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./datasets/wiki_10k \
    --max-samples 10000
```

### Step 3: Train on the saved dataset

```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_10k \
    --device cuda
```

## Your Use Case (Google Colab / Drive)

Perfect for your scenario where you have a dataset saved at `/content/drive/MyDrive/SentenceJEPA`:

```python
# In Colab notebook

# If you already saved it with load_from_disk:
from datasets import load_from_disk
ds = load_from_disk("/content/drive/MyDrive/SentenceJEPA")

# Now train with it directly:
!python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA \
    --device cuda
```

**Benefits:**
- ✅ Works across Colab sessions (no re-download)
- ✅ Saved on Google Drive (persistent)
- ✅ No bandwidth usage after initial save
- ✅ Fast startup (no download time)

## What Was Added

### 1. New Function: `load_from_disk_dataset()`

**Location:** `data/hf_dataset.py`

```python
from data import load_from_disk_dataset

dataset = load_from_disk_dataset(
    dataset_path='/path/to/saved/dataset',
    text_column='text',
    min_sentences=3,
    max_sentences=10,
    max_samples=None,  # Use all samples
    use_streaming=False,  # Map-style or streaming
)
```

### 2. Training Script Support

**Updated:** `scripts/train_hf.py`

New `--dataset from-disk` option:

```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /path/to/dataset \
    --text-column text \
    --config configs/base.yaml
```

### 3. Download Helper Script

**New file:** `examples/download_and_save_dataset.py`

```bash
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./datasets/wiki_10k \
    --max-samples 10000
```

Supports: wikipedia, c4, bookcorpus, or any custom dataset.

### 4. Test Script

**New file:** `test_from_disk.py`

```bash
python test_from_disk.py
```

Verifies the from-disk loading works correctly.

## All Available Commands

### Download Datasets

```bash
# Wikipedia
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./datasets/wiki_data \
    --max-samples 10000

# C4
python examples/download_and_save_dataset.py \
    --dataset c4 \
    --output ./datasets/c4_data \
    --max-samples 50000 \
    --streaming

# BookCorpus
python examples/download_and_save_dataset.py \
    --dataset bookcorpus \
    --output ./datasets/books_data \
    --max-samples 5000
```

### Train on Saved Datasets

```bash
# Basic training
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_data

# With custom config
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_data \
    --config configs/large_scale.yaml

# With GPU
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wiki_data \
    --device cuda

# Custom text column
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/custom_data \
    --text-column content
```

## Complete Example Workflow

### Scenario: Train on Wikipedia, experiment with different configs

```bash
# 1. Download once (takes time)
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./datasets/wikipedia_20k \
    --max-samples 20000

# 2. Experiment 1: Base config
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wikipedia_20k \
    --config configs/base.yaml

# 3. Experiment 2: Large scale config (same data, no re-download!)
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wikipedia_20k \
    --config configs/large_scale.yaml

# 4. Experiment 3: Custom hyperparameters
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./datasets/wikipedia_20k \
    --config configs/custom.yaml
```

**Time saved:** No re-downloading between experiments!

## Programmatic Usage

```python
# Download and save
from datasets import load_dataset

dataset = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)
dataset = dataset.take(10000)

from datasets import Dataset
data = list(dataset)
dataset = Dataset.from_list(data)
dataset.save_to_disk('./datasets/my_data')

# Later: Load and train
from data import load_from_disk_dataset, SentenceJEPACollator
from torch.utils.data import DataLoader

dataset = load_from_disk_dataset(
    dataset_path='./datasets/my_data',
    min_sentences=3,
)

collator = SentenceJEPACollator(tokenizer_name='roberta-base')
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

for batch in dataloader:
    # Training code
    pass
```

## Documentation

Four new/updated docs:

1. **[OFFLINE_TRAINING.md](OFFLINE_TRAINING.md)** - Complete offline training guide
2. **[HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md)** - Updated with from-disk examples
3. **[CHEATSHEET.md](CHEATSHEET.md)** - Added from-disk commands
4. **[examples/README.md](examples/README.md)** - Example workflows

## Common Use Cases

### Use Case 1: Limited Bandwidth
```bash
# At work/university (good internet)
python examples/download_and_save_dataset.py --dataset wikipedia --output ~/datasets/wiki

# At home (limited internet)
python scripts/train_hf.py --dataset from-disk --dataset-path ~/datasets/wiki
```

### Use Case 2: Google Colab
```python
# Session 1: Download to Drive
!python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output /content/drive/MyDrive/datasets/wiki \
    --max-samples 10000

# Sessions 2, 3, 4, ... : Reuse
!python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/datasets/wiki \
    --device cuda
```

### Use Case 3: Team Sharing
```bash
# Team member 1: Download
python examples/download_and_save_dataset.py --dataset c4 --output /shared/datasets/c4_50k --max-samples 50000

# Team members 2, 3, 4: Train on shared data
python scripts/train_hf.py --dataset from-disk --dataset-path /shared/datasets/c4_50k
```

### Use Case 4: Reproducible Research
```bash
# Save exact dataset used in paper
python examples/download_and_save_dataset.py \
    --dataset wikipedia \
    --output ./paper_datasets/wiki_v1 \
    --max-samples 10000

# Document it
echo "Wikipedia 20220301.en, first 10K samples" > ./paper_datasets/wiki_v1/README.txt

# Future researchers can use exact same data
python scripts/train_hf.py --dataset from-disk --dataset-path ./paper_datasets/wiki_v1
```

## Compatibility

Works with any dataset saved using:
- `dataset.save_to_disk(path)` (HuggingFace datasets)
- `examples/download_and_save_dataset.py` (our helper)
- Manual download and save

**Requirements:**
- Dataset must have a `text` column (or specify with `--text-column`)
- Must be in HuggingFace datasets format

## Testing

```bash
# Test the feature
python test_from_disk.py

# Should output:
# ✓ Imports working
# ✓ Dataset creation and saving
# ✓ Dataset loading from disk
# ✓ Sample processing
# ✓ Dataloader integration
```

## Summary

**What you can now do:**
1. ✅ Download datasets once, use unlimited times
2. ✅ Work offline after initial download
3. ✅ Share datasets with team
4. ✅ Persist datasets across Colab sessions (Drive)
5. ✅ Experiment faster (no re-download time)
6. ✅ Save bandwidth

**Command you need:**
```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /path/to/your/dataset
```

**For your specific case:**
```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA \
    --device cuda
```

That's it! Your pre-downloaded dataset at `/content/drive/MyDrive/SentenceJEPA` will work directly now! 🎉

---

**See also:**
- [OFFLINE_TRAINING.md](OFFLINE_TRAINING.md) - Full offline training guide
- [HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md) - Complete HF reference
- [examples/README.md](examples/README.md) - More examples
