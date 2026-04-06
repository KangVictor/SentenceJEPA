# Training on Top of Checkpoints

Complete guide for resuming training or fine-tuning from saved checkpoints.

## Table of Contents
1. [Resume Training (Same Data)](#resume-training-same-data)
2. [Fine-tune on New Data](#fine-tune-on-new-data)
3. [Transfer Learning](#transfer-learning)
4. [Understanding Options](#understanding-options)

---

## Resume Training (Same Data)

**Use case:** Training was interrupted, want to continue exactly where you left off.

### Quick Command

```bash
python scripts/resume_training.py \
    --checkpoint checkpoints/checkpoint_step_5000.pt \
    --config configs/base.yaml \
    --data-path /content/drive/MyDrive/SentenceJEPA \
    --device cuda
```

**What this does:**
- ✅ Loads model weights
- ✅ Loads optimizer state (momentum, etc.)
- ✅ Loads scheduler state (learning rate)
- ✅ Continues from same step/epoch
- ✅ Uses same data

**Result:** Training continues seamlessly from step 5000.

---

## Fine-tune on New Data

**Use case:** Have trained model, want to adapt it to new domain/dataset.

### Option 1: Reset Optimizer (Recommended)

```bash
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /path/to/new/data \
    --reset-optimizer \
    --reset-steps \
    --new-lr 5e-5 \
    --device cuda
```

**What this does:**
- ✅ Loads model weights (your learned representations)
- ✗ Resets optimizer (starts fresh momentum)
- ✗ Resets step counter (starts from 0)
- ✅ Uses new learning rate
- ✅ Trains on new data

**Good for:**
- Different domain (e.g., Wikipedia → medical text)
- New dataset with different characteristics
- Starting fresh training phase

### Option 2: Continue Optimizer State

```bash
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /path/to/new/data \
    --new-lr 1e-5 \
    --device cuda
```

**What this does:**
- ✅ Loads everything (model + optimizer)
- ✅ Keeps optimizer momentum
- ✅ Lower learning rate for stability

**Good for:**
- Similar data to original
- Expanding training set
- Gentle fine-tuning

---

## Transfer Learning

**Use case:** Use pretrained model for different but related task.

### Example: Domain Adaptation

```bash
# Train on Wikipedia
python scripts/train_hf.py \
    --dataset from-disk \
    --data-path /content/drive/MyDrive/Wikipedia \
    --config configs/base.yaml

# Fine-tune on medical texts
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /content/drive/MyDrive/MedicalTexts \
    --reset-optimizer \
    --new-lr 1e-5 \
    --additional-epochs 5
```

### Example: Multi-stage Training

```bash
# Stage 1: Train on large general corpus
python scripts/train_hf.py \
    --dataset from-disk \
    --data-path /data/general_corpus \
    --config configs/base.yaml \
    --max-samples 100000

# Stage 2: Fine-tune on domain-specific data
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /data/domain_specific \
    --reset-optimizer \
    --new-lr 5e-6 \
    --additional-epochs 3
```

---

## Understanding Options

### `--checkpoint`
**Required.** Path to checkpoint to load.

```bash
--checkpoint checkpoints/best_model.pt          # Best by validation
--checkpoint checkpoints/checkpoint_step_5000.pt # Specific step
```

### `--reset-optimizer`
**Optional.** Resets optimizer state (momentum, variance).

**Use when:**
- ✅ Fine-tuning on new data
- ✅ Changing learning rate significantly
- ✅ Domain shift between datasets

**Don't use when:**
- ❌ Resuming same training run
- ❌ Just continuing on same data

```bash
--reset-optimizer  # Starts fresh Adam momentum
```

### `--reset-scheduler`
**Optional.** Resets learning rate scheduler.

**Use when:**
- ✅ Want fresh LR warmup
- ✅ Changing training schedule

```bash
--reset-scheduler  # Restarts warmup and cosine schedule
```

### `--reset-steps`
**Optional.** Resets global step counter to 0.

**Use when:**
- ✅ Starting new training phase
- ✅ Want clean logging

**Don't use when:**
- ❌ Tracking total training across runs

```bash
--reset-steps  # Starts counting from 0 again
```

### `--new-lr`
**Optional.** Override learning rate from config.

**Common values:**
- `1e-4`: Normal training (default)
- `5e-5`: Gentle fine-tuning
- `1e-5`: Very gentle fine-tuning
- `5e-6`: Minimal changes (domain adaptation)

```bash
--new-lr 5e-5  # Lower LR for fine-tuning
```

### `--additional-epochs`
**Optional.** Override number of epochs to train.

```bash
--additional-epochs 5  # Train for 5 more epochs
```

---

## Common Scenarios

### Scenario 1: Training Interrupted

**Problem:** Training stopped at step 3500, want to continue.

**Solution:**
```bash
python scripts/resume_training.py \
    --checkpoint checkpoints/checkpoint_step_3000.pt \
    --config configs/base.yaml \
    --data-path /content/drive/MyDrive/SentenceJEPA \
    --device cuda
```

**No special flags needed** - everything resumes automatically.

### Scenario 2: Want to Train Longer

**Problem:** Training finished but want more epochs.

**Solution:**
```bash
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /content/drive/MyDrive/SentenceJEPA \
    --additional-epochs 10 \
    --device cuda
```

### Scenario 3: Fine-tune on New Domain

**Problem:** Trained on Wikipedia, want to adapt to scientific papers.

**Solution:**
```bash
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /data/scientific_papers \
    --reset-optimizer \
    --reset-steps \
    --new-lr 5e-5 \
    --additional-epochs 5 \
    --device cuda
```

### Scenario 4: Increase Model Capacity

**Problem:** Want to unfreeze sentence encoder and train it too.

**Solution:**
```bash
# First, edit configs/base.yaml:
# model:
#   sentence_encoder:
#     frozen: false  # Changed from true

python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /content/drive/MyDrive/SentenceJEPA \
    --reset-optimizer \
    --new-lr 1e-5 \
    --additional-epochs 5 \
    --device cuda
```

**Note:** Unfreezing adds ~125M trainable parameters!

### Scenario 5: Experiment with Hyperparameters

**Problem:** Want to try different batch size or learning rate.

**Solution:**
```bash
# Edit config file first (e.g., configs/finetune.yaml)
# training:
#   batch_size: 128  # Increased from 64
#   lr_rest: 1e-5    # Decreased for stability

python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/finetune.yaml \
    --data-path /content/drive/MyDrive/SentenceJEPA \
    --reset-optimizer \
    --device cuda
```

---

## Programmatic Usage

For custom training loops:

```python
import torch
from train import Trainer
from models import HierarchicalSentenceJEPA

# Create model
model = HierarchicalSentenceJEPA(...)

# Create trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    # ... other args
)

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Load model weights
model.load_state_dict(checkpoint['model_state_dict'])

# Optionally load optimizer/scheduler
trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Optionally load training state
trainer.global_step = checkpoint['global_step']
trainer.epoch = checkpoint['epoch']

# Continue training
trainer.train()
```

### Custom Fine-tuning

```python
# Load only model weights, start fresh optimizer
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Create new optimizer with lower LR
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)

# Create new trainer with new optimizer
trainer = Trainer(
    model=model,
    train_dataloader=new_dataloader,
    # ... other args
)

# Train on new data
trainer.train()
```

---

## Tips & Best Practices

### Learning Rate Guidelines

| Scenario | Recommended LR | Reasoning |
|----------|---------------|-----------|
| **Resume same training** | Keep original | Maintain optimization trajectory |
| **Fine-tune similar data** | `5e-5` | Slight adjustments |
| **Fine-tune different domain** | `1e-5` | Careful domain adaptation |
| **Unfreeze encoder** | `1e-5` | Pretrained weights need gentle updates |

### When to Reset Optimizer

**Reset** (`--reset-optimizer`) if:
- ✅ Data distribution changed significantly
- ✅ Using much different learning rate
- ✅ Starting new training phase

**Keep** (no flag) if:
- ✅ Same data, just continuing
- ✅ Minor adjustments only
- ✅ Want to maintain momentum

### Monitoring Progress

After resuming training, check:

1. **Loss should continue decreasing** (or at least not spike)
2. **Metrics should improve** (Recall@1, etc.)
3. **GPU utilization** should remain high

If loss spikes:
- Lower learning rate
- Reset optimizer
- Check data quality

### Checkpointing Strategy

1. **Keep multiple checkpoints** during training
   ```
   checkpoint_step_1000.pt
   checkpoint_step_2000.pt
   checkpoint_step_3000.pt
   best_model.pt
   ```

2. **Always keep `best_model.pt`** - best by validation metric

3. **For long runs:** Save every 1000 steps (configurable)

4. **After successful training:** Can delete intermediate checkpoints

---

## Troubleshooting

### Issue: Loss spikes after resuming

**Solution:**
```bash
# Lower learning rate
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /data \
    --new-lr 1e-5 \
    --reset-optimizer
```

### Issue: "Model architecture mismatch"

**Cause:** Config doesn't match checkpoint architecture.

**Solution:** Use same config as original training, or:
```python
# Check checkpoint architecture
checkpoint = torch.load('checkpoints/best_model.pt')
for key in checkpoint['model_state_dict'].keys():
    print(key)
```

Match your config to these layers.

### Issue: Training not improving

**Possible causes:**
1. Learning rate too low → Increase LR
2. Learning rate too high → Decrease LR
3. Data too similar to original → Use different data
4. Model already converged → Normal, stop training

### Issue: Out of memory after unfreezing encoder

**Solution:** Reduce batch size:
```yaml
# configs/base.yaml
training:
  batch_size: 32  # Reduce from 64
```

---

## Examples

### Example 1: Multi-Domain Training

```bash
# Stage 1: General corpus (Wikipedia)
python scripts/train_hf.py \
    --dataset from-disk \
    --data-path /data/wikipedia \
    --config configs/base.yaml

# Stage 2: Scientific texts
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /data/scientific \
    --reset-optimizer \
    --new-lr 5e-5 \
    --additional-epochs 3

# Stage 3: Medical texts
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /data/medical \
    --reset-optimizer \
    --new-lr 1e-5 \
    --additional-epochs 3
```

### Example 2: Incremental Training

```bash
# Initial training on 10K samples
python scripts/train_hf.py \
    --dataset from-disk \
    --data-path /data/corpus \
    --max-samples 10000 \
    --config configs/base.yaml

# Expand to 50K samples
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /data/corpus \
    --max-samples 50000

# Expand to 100K samples
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /data/corpus \
    --max-samples 100000
```

### Example 3: Unfreezing Encoder

```bash
# Phase 1: Train with frozen encoder
python scripts/train_hf.py \
    --dataset from-disk \
    --data-path /data/corpus \
    --config configs/base.yaml
# (frozen: true in config)

# Phase 2: Unfreeze and fine-tune entire model
# Edit configs/finetune.yaml to set frozen: false

python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/finetune.yaml \
    --data-path /data/corpus \
    --reset-optimizer \
    --new-lr 1e-5 \
    --additional-epochs 3
```

---

## Quick Reference

```bash
# Resume exact same training
python scripts/resume_training.py \
    --checkpoint checkpoints/checkpoint_step_5000.pt \
    --config configs/base.yaml \
    --data-path /data

# Fine-tune on new data
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /new_data \
    --reset-optimizer \
    --new-lr 5e-5

# Train longer
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /data \
    --additional-epochs 10

# Full reset (keep only weights)
python scripts/resume_training.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data-path /new_data \
    --reset-optimizer \
    --reset-scheduler \
    --reset-steps \
    --new-lr 1e-4
```

---

**See also:**
- [MODEL_CHECKPOINTS.md](MODEL_CHECKPOINTS.md) - What's in checkpoints
- [USE_TRAINED_MODEL.md](USE_TRAINED_MODEL.md) - Using trained models
- [README.md](README.md) - Full documentation
