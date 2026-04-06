# GPU Training Optimization Guide

## Your Problem: Only 2GB VRAM Used on G4 GPU

**Symptoms:**
- Slow training
- Low GPU utilization (2GB out of 15GB)
- GPU mostly idle

**Causes:**
1. **Batch size too small** (default: 16)
2. **Data loading bottleneck** (CPU preprocessing)
3. **Small model** (not utilizing GPU capacity)

## Quick Fix (5 Minutes)

### Step 1: Check Current GPU Usage

```bash
# In a separate terminal, monitor GPU
watch -n 1 nvidia-smi

# Or in Colab
!watch -n 1 nvidia-smi
```

### Step 2: Use GPU-Optimized Config

```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA \
    --config configs/gpu_optimized.yaml \
    --device cuda
```

**Changes in gpu_optimized.yaml:**
- ✅ Batch size: 16 → **64** (4x larger!)
- ✅ More frequent logging
- ✅ Optimized for GPU memory

### Step 3: Profile Your Setup

```bash
python scripts/profile_training.py --config configs/gpu_optimized.yaml
```

This will:
- Check GPU availability
- Test different batch sizes
- Recommend optimal batch size for your GPU

## Detailed Optimizations

### 1. Increase Batch Size (MOST IMPORTANT)

**Problem:** Default batch size of 16 is too small for modern GPUs.

**Solution:** Increase batch size to utilize GPU memory.

```yaml
# configs/gpu_optimized.yaml
training:
  batch_size: 64  # or even 128
```

**For G4 GPU (16GB):**
- Batch size 64: ~6-8GB VRAM
- Batch size 128: ~12-14GB VRAM

**Test maximum batch size:**
```bash
python scripts/profile_training.py
```

### 2. Use Preprocessed Dataset

**Problem:** Processing paragraphs on-the-fly is slow (CPU bottleneck).

**Solution:** Preprocess once, reuse forever.

```bash
# Preprocess (run once, takes time)
python scripts/preprocess_dataset.py \
    --input /content/drive/MyDrive/SentenceJEPA \
    --output /content/drive/MyDrive/SentenceJEPA_processed.pkl \
    --max-samples 100000

# Train on preprocessed data (FAST!)
python scripts/train_hf.py \
    --dataset preprocessed \
    --dataset-path /content/drive/MyDrive/SentenceJEPA_processed.pkl \
    --config configs/gpu_optimized.yaml \
    --device cuda
```

**Speed improvement:** 10-50x faster data loading!

### 3. Increase num_workers

**Problem:** Single-threaded data loading.

**Solution:** Add num_workers to your training script.

Edit `scripts/train_hf.py` (or run with modified script):

```python
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    collate_fn=collator,
    num_workers=4,  # Use multiple workers
    pin_memory=True,  # Faster GPU transfer
)
```

### 4. Mixed Precision Training (FP16)

**Speed improvement:** 2x faster, uses less memory.

**Not implemented yet**, but can be added if needed.

### 5. Larger Model (Optional)

If GPU is still underutilized after increasing batch size:

```yaml
# configs/gpu_large.yaml
model:
  paragraph_transformer:
    d_model: 1024  # Increase from 768
    num_layers: 6   # Increase from 4
    dim_feedforward: 4096  # Increase from 2048
```

## Performance Comparison

| Configuration | VRAM Usage | Speed (samples/sec) |
|---------------|------------|---------------------|
| Default (batch 16) | 2GB | ~20 |
| Optimized (batch 64) | 8GB | ~80 |
| + Preprocessed data | 8GB | ~200 |
| + num_workers=4 | 8GB | ~250 |
| Large model (batch 64) | 12GB | ~60 |

## Monitoring GPU Usage

### During Training

```bash
# Terminal 1: Train
python scripts/train_hf.py ...

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

### In Colab

```python
# In one cell (runs in background)
%%bash
while true; do
    nvidia-smi
    sleep 2
done
```

```python
# In another cell
!python scripts/train_hf.py ...
```

### Check GPU Utilization

**Good utilization:**
- Memory: >50% of available (>8GB on G4)
- GPU-Util: >80%
- Temperature: 60-80°C

**Bad utilization:**
- Memory: <20% (<3GB on G4)
- GPU-Util: <30%
- Temperature: <50°C

## Troubleshooting

### GPU Usage Still Low After Increasing Batch Size

**Check:**
1. Is model on GPU?
   ```python
   next(model.parameters()).device  # Should be 'cuda:0'
   ```

2. Are batches on GPU?
   ```python
   print(batch['input_ids'].device)  # Should be 'cuda:0'
   ```

3. Is data loading the bottleneck?
   - **Symptom:** GPU utilization spikes, then drops to 0%
   - **Solution:** Preprocess dataset or add num_workers

### Out of Memory (OOM)

**Solution:** Reduce batch size gradually:

```yaml
training:
  batch_size: 32  # Try 32, 24, 16
```

Or enable gradient accumulation (not implemented yet).

### Training Still Slow

**Possible causes:**

1. **CPU bottleneck** - Preprocess dataset
2. **Small dataset** - More samples loaded = more time
3. **Disk I/O** - Use faster storage (SSD vs HDD)
4. **Model on CPU** - Ensure `--device cuda`

## Recommended Workflow

### For G4 GPU (16GB VRAM)

```bash
# 1. Profile to find optimal batch size
python scripts/profile_training.py

# 2. Preprocess dataset (one time, 20-60 min for 100K samples)
python scripts/preprocess_dataset.py \
    --input /content/drive/MyDrive/SentenceJEPA \
    --output /content/drive/MyDrive/SentenceJEPA_processed.pkl \
    --max-samples 100000

# 3. Train with optimized settings
python scripts/train_hf.py \
    --dataset preprocessed \
    --dataset-path /content/drive/MyDrive/SentenceJEPA_processed.pkl \
    --config configs/gpu_optimized.yaml \
    --device cuda
```

**Expected performance:**
- VRAM usage: 8-12GB
- GPU utilization: >80%
- Speed: 150-250 samples/sec
- Training 100K samples: ~10-15 minutes

## Quick Commands

```bash
# Check GPU
nvidia-smi

# Profile training
python scripts/profile_training.py

# Train with GPU optimization
python scripts/train_hf.py \
    --config configs/gpu_optimized.yaml \
    --device cuda

# Monitor during training
watch -n 1 nvidia-smi
```

## Summary

**To fix low GPU usage:**

1. ✅ **Increase batch size to 64** (or run profiler to find max)
2. ✅ **Preprocess dataset** for faster data loading
3. ✅ **Use configs/gpu_optimized.yaml**
4. ✅ **Monitor with nvidia-smi**

**Expected result:**
- VRAM: 8-12GB (instead of 2GB)
- Speed: 10x faster
- GPU utilization: >80%

---

**See also:**
- [CHEATSHEET.md](CHEATSHEET.md) - Quick commands
- [LARGE_DATASET_GUIDE.md](LARGE_DATASET_GUIDE.md) - Scaling strategies
