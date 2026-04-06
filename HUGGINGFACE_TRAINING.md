# Training with HuggingFace Datasets - Complete Setup

This guide shows you exactly how to train Sentence JEPA on HuggingFace datasets.

## Prerequisites

You already have:
- ✅ Virtual environment (`.venv`)
- ✅ Basic dependencies installed
- ✅ Pipeline tested and working

## Step 1: Install HuggingFace Datasets

```bash
source .venv/bin/activate
pip install datasets
```

This installs the HuggingFace `datasets` library which gives you access to thousands of datasets.

## Step 2: Test HuggingFace Integration

```bash
python test_hf_datasets.py
```

**Expected output:**
```
Testing HuggingFace Datasets Integration
[1/4] Testing imports...
   ✓ Imports successful
[2/4] Loading Wikipedia sample (5 paragraphs)...
   ✓ Dataset loaded
[3/4] Checking samples...
   Sample 1:
      Sentences: 4
      First sentence: ...
   ✓ Samples look good
[4/4] Testing with dataloader and collator...
   ✓ Dataloader working correctly

All tests passed!
```

**Note:** First run will download model and data from HuggingFace (requires internet).

## Step 3: Train on Wikipedia (Quick Test)

```bash
python scripts/train_hf.py \
    --dataset wikipedia \
    --streaming \
    --max-samples 10000
```

**What this does:**
- Loads Wikipedia dataset in streaming mode
- Processes first 10,000 paragraphs
- Trains for default epochs (20)
- Saves checkpoints to `checkpoints/`

**Expected time:** ~30 minutes on GPU, ~3 hours on CPU

**Expected results:**
- Recall@1: ~0.3-0.4 (with 10K samples)
- Better results with more data

## Step 4: Scale Up (Full Wikipedia)

```bash
python scripts/train_hf.py \
    --dataset wikipedia \
    --streaming \
    --config configs/large_scale.yaml \
    --device cuda
```

**What this does:**
- Processes entire English Wikipedia (~6M articles)
- Uses large-scale config (bigger model, more projections)
- Trains on GPU

**Expected time:** ~10-15 hours on single GPU

**Expected results:**
- Recall@1: 0.7-0.9 (very good!)
- Recall@5: 0.9-0.95

## Available Datasets

### 1. Wikipedia
```bash
# English (default)
python scripts/train_hf.py --dataset wikipedia --streaming

# Spanish
python scripts/train_hf.py --dataset wikipedia --wiki-lang es --streaming

# French
python scripts/train_hf.py --dataset wikipedia --wiki-lang fr --streaming
```

**Size:** ~6M articles (English)
**Good for:** Factual, encyclopedic text

### 2. C4 (Colossal Clean Crawled Corpus)
```bash
python scripts/train_hf.py \
    --dataset c4 \
    --streaming \
    --config configs/large_scale.yaml
```

**Size:** ~365GB (massive!)
**Good for:** Diverse web text, robust representations
**Tip:** Use `--max-samples` to limit amount

### 3. BookCorpus
```bash
python scripts/train_hf.py --dataset bookcorpus --streaming
```

**Size:** ~5GB (11K books)
**Good for:** Narrative text, coherent paragraphs

### 4. Custom Dataset
```bash
python scripts/train_hf.py \
    --dataset custom \
    --hf-name "username/dataset-name" \
    --text-column "text"
```

Browse datasets at: https://huggingface.co/datasets

## Common Options

### Limit Dataset Size
```bash
--max-samples 50000  # Process first 50K paragraphs
```

### Use GPU
```bash
--device cuda
```

### Change Config
```bash
--config configs/large_scale.yaml  # Bigger model
--config configs/base.yaml         # Standard model
```

### Multilingual Wikipedia
```bash
--wiki-lang es  # Spanish
--wiki-lang fr  # French
--wiki-lang de  # German
# See full list: https://en.wikipedia.org/wiki/List_of_Wikipedias
```

## Understanding Streaming Mode

### What is Streaming?
- **Streaming**: Downloads data on-the-fly, processes one sample at a time
- **Non-streaming**: Downloads entire dataset first, then trains

### When to Use Streaming? (Recommended)

✅ **Use streaming when:**
- Dataset is large (>1GB)
- Limited disk space
- Want to start training immediately
- Don't need perfect shuffling

❌ **Don't use streaming when:**
- Dataset is small (<100MB)
- Need validation split
- Want perfect data shuffling
- Have plenty of disk space

### Example: Streaming vs Non-Streaming

```bash
# Streaming (recommended for large datasets)
python scripts/train_hf.py \
    --dataset wikipedia \
    --streaming \
    --max-samples 100000

# Non-streaming (for small datasets)
python scripts/train_hf.py \
    --dataset wikipedia \
    --max-samples 10000 \
    --val-split 0.1
```

## Monitoring Training

### During Training
Watch the output:
```
Epoch 1/20: 100%|██████| loss: 0.2543, jepa: 0.0432, sig: 2.1103, lr: 1.5e-05
```

**What to look for:**
- `loss`: Should decrease over time
- `jepa`: Typically 0.01-0.5 (lower is better)
- `sig`: Typically 0.5-3.0 (stabilizes)
- `lr`: Learning rate (changes with scheduler)

### Check Checkpoints
```bash
ls -lh checkpoints/
```

**Files:**
- `checkpoint_step_1000.pt`: Regular checkpoint
- `checkpoint_step_2000.pt`: Regular checkpoint
- `best_model.pt`: Best model by Recall@1

### Evaluate Model
```bash
python scripts/eval_retrieval.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data data/sample_data.txt
```

## Practical Examples

### Example 1: Quick Test (5 minutes)
```bash
# Test everything works
python scripts/train_hf.py \
    --dataset wikipedia \
    --streaming \
    --max-samples 1000 \
    --config configs/base.yaml
```

### Example 2: Small Experiment (30 minutes)
```bash
# Train on 10K Wikipedia articles
python scripts/train_hf.py \
    --dataset wikipedia \
    --streaming \
    --max-samples 10000 \
    --device cuda
```

### Example 3: Production (10+ hours)
```bash
# Full Wikipedia training
python scripts/train_hf.py \
    --dataset wikipedia \
    --streaming \
    --config configs/large_scale.yaml \
    --device cuda
```

### Example 4: Massive Scale (40+ hours)
```bash
# C4 dataset (huge!)
python scripts/train_hf.py \
    --dataset c4 \
    --streaming \
    --max-samples 500000 \
    --config configs/large_scale.yaml \
    --device cuda
```

## Troubleshooting

### "No module named 'datasets'"
```bash
pip install datasets
```

### "Connection timeout" / "Download failed"
- Check internet connection
- Datasets download from HuggingFace servers
- First download is slow, then cached locally

### "CUDA out of memory"
```bash
# Reduce batch size in config
# Edit configs/base.yaml:
training:
  batch_size: 8  # Reduce from 16
```

### "Training is very slow"
- Use GPU: `--device cuda`
- Increase batch size if you have memory
- Use `--max-samples` for quick tests

### "Samples look low quality"
Check samples:
```python
from data.hf_dataset import load_wikipedia_dataset

dataset = load_wikipedia_dataset(streaming=True, max_samples=5)
for item in dataset:
    print(item['sentences'])
```

Adjust filtering:
```python
dataset = load_wikipedia_dataset(
    min_sentences=4,  # Increase from 3
    max_samples=100,
)
```

## Performance Tips

### 1. Start Small, Scale Up
```bash
# Day 1: Test (5 min)
--max-samples 1000

# Day 2: Small (30 min)
--max-samples 10000

# Day 3: Medium (3 hours)
--max-samples 100000

# Day 4: Full (10+ hours)
# Remove --max-samples
```

### 2. Monitor GPU Usage
```bash
# In another terminal
watch -n 1 nvidia-smi
```

Look for:
- GPU utilization: Should be >80%
- Memory usage: Should be high but not OOM
- Temperature: Should be <85°C

### 3. Optimize Batch Size
```yaml
# If GPU underutilized (memory available)
batch_size: 64  # Increase

# If getting OOM
batch_size: 16  # Decrease
```

### 4. Use Multiple GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_hf.py ...
```

## Next Steps

1. **✅ Test HuggingFace integration**
   ```bash
   python test_hf_datasets.py
   ```

2. **✅ Quick training test**
   ```bash
   python scripts/train_hf.py --dataset wikipedia --streaming --max-samples 1000
   ```

3. **✅ Real training**
   ```bash
   python scripts/train_hf.py --dataset wikipedia --streaming --device cuda
   ```

4. **✅ Evaluate**
   ```bash
   python scripts/eval_retrieval.py --checkpoint checkpoints/best_model.pt
   ```

## More Resources

- **[HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md)** - Comprehensive HF guide
- **[LARGE_DATASET_GUIDE.md](LARGE_DATASET_GUIDE.md)** - Scaling strategies
- **[CHEATSHEET.md](CHEATSHEET.md)** - Quick reference
- **[README.md](README.md)** - Full documentation

## Summary Commands

```bash
# Setup
source .venv/bin/activate
pip install datasets

# Test
python test_hf_datasets.py

# Train (quick test)
python scripts/train_hf.py --dataset wikipedia --streaming --max-samples 1000

# Train (production)
python scripts/train_hf.py --dataset wikipedia --streaming --device cuda

# Evaluate
python scripts/eval_retrieval.py --checkpoint checkpoints/best_model.pt
```

Happy training! 🚀
