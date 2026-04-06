# Understanding Model Checkpoints

## What Gets Saved

When training, checkpoints are saved to `checkpoints/` containing:

### 1. Model State Dict (`model_state_dict`)

The complete **HierarchicalSentenceJEPA** model with all trained weights:

- **Sentence Encoder** (RoBERTa-base or similar)
  - ~125M parameters (frozen by default, so not updated)
  - Converts sentences to embeddings

- **Input Projection**
  - Linear layer (if needed) to match dimensions

- **Target Paragraph Transformer**
  - Multi-layer transformer for context
  - ~10M parameters
  - Processes full paragraph

- **Predictor Paragraph Transformer**
  - Separate transformer for masked prediction
  - ~10M parameters
  - Processes paragraph with masked sentence

- **Mask Embedding**
  - Learned mask token
  - Small (~768 dims)

- **Target Projection Head**
  - MLP projecting to 512-dim space
  - ~1M parameters

- **Predictor Projection Head**
  - MLP projecting to 512-dim space
  - ~1M parameters

**Total trained parameters:** ~25M (if encoder frozen) or ~150M (if encoder unfrozen)

### 2. Optimizer State (`optimizer_state_dict`)

AdamW optimizer state:
- Momentum buffers
- Variance buffers
- Learning rate schedules for each parameter group

**Purpose:** Resume training from exact same optimization state

### 3. Scheduler State (`scheduler_state_dict`)

Learning rate scheduler state:
- Current step
- Learning rate value
- Warmup progress

**Purpose:** Continue LR schedule when resuming

### 4. Training Metadata

- `global_step`: Total training steps completed
- `epoch`: Current epoch number
- `best_recall`: Best Recall@1 score achieved

## Checkpoint Files

After training, you'll have:

```
checkpoints/
├── checkpoint_step_1000.pt    # Saved every 1000 steps
├── checkpoint_step_2000.pt
├── checkpoint_step_3000.pt
├── ...
└── best_model.pt              # Best model by Recall@1
```

**File size:** ~100-600MB depending on model size and frozen encoder

## What Can You Do With the Saved Model?

### 1. Resume Training

```python
from train import Trainer

# Create trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    # ... other args
)

# Resume from checkpoint
trainer.load_checkpoint('checkpoints/checkpoint_step_5000.pt')

# Continue training
trainer.train()
```

### 2. Evaluate on New Data

```bash
python scripts/eval_retrieval.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data your_test_data.txt
```

### 3. Extract Sentence Embeddings

```python
import torch
from models import HierarchicalSentenceJEPA

# Load model
model = HierarchicalSentenceJEPA(
    sentence_encoder_name='roberta-base',
    # ... match your training config
)

checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Now use model for inference
# See USE_MODEL.md for examples
```

### 4. Fine-tune on Different Task

Load the checkpoint and continue training with:
- Different dataset
- Different learning rate
- Different config

### 5. Export for Production

Extract just the model (no optimizer/scheduler):

```python
checkpoint = torch.load('checkpoints/best_model.pt')

# Save only model weights
torch.save(
    {'model_state_dict': checkpoint['model_state_dict']},
    'model_only.pt'
)
```

**Smaller file:** ~100-300MB (vs 600MB with optimizer)

## What the Model Learned

The trained model has learned to:

1. **Contextualize sentences** - Understand sentence meaning in paragraph context
2. **Predict masked sentences** - Infer missing sentence from context
3. **Generate embeddings** - Create 512-dim representations capturing:
   - Sentence semantics
   - Contextual information
   - Position in discourse

These embeddings can be used for:
- Sentence similarity
- Sentence retrieval
- Document understanding
- Semantic search
- Clustering similar sentences

## Inspecting a Checkpoint

```python
import torch

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# See what's inside
print("Keys:", checkpoint.keys())
# Output: ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
#          'global_step', 'epoch', 'best_recall']

# Check training progress
print(f"Trained for {checkpoint['global_step']} steps")
print(f"Best Recall@1: {checkpoint['best_recall']:.4f}")

# See model architecture
print("\nModel layers:")
for name in checkpoint['model_state_dict'].keys():
    if 'weight' in name:
        shape = checkpoint['model_state_dict'][name].shape
        print(f"  {name}: {shape}")
```

## Model Architecture Recap

```
Input: Paragraph text
    ↓
[1] Tokenize & Split Sentences
    ↓
[2] Sentence Encoder (RoBERTa) → [B, S, 768]
    ↓
[3] Input Projection (optional) → [B, S, D]
    ↓
    ├─→ [4a] Target Branch
    │        └→ Paragraph Transformer
    │             └→ Target Projection Head → z_target [B, 512]
    │
    └─→ [4b] Predictor Branch (with MASK)
             └→ Paragraph Transformer
                  └→ Predictor Projection Head → z_pred [B, 512]
    ↓
Loss: normalized_MSE(z_pred, z_target) + λ * SIGReg(z_target)
```

**All of this is saved in the checkpoint!**

## Which Checkpoint to Use?

### `best_model.pt`
- **Best performance** on validation set
- Highest Recall@1 score
- **Recommended** for evaluation/inference

### `checkpoint_step_N.pt`
- Saved periodically during training
- Use to resume training
- Use if you want a specific training stage

## Loading for Different Purposes

### For Inference/Evaluation
```python
# Just need model weights
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### For Resuming Training
```python
# Need everything (optimizer, scheduler, etc.)
trainer.load_checkpoint('checkpoints/checkpoint_step_5000.pt')
trainer.train()  # Continues from step 5000
```

### For Transfer Learning
```python
# Load pretrained weights, then modify
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Modify for your task
model.projection_head = CustomHead(...)  # Your custom head
# Train on your data
```

## Checkpoint Best Practices

### During Training

1. **Save periodically** - Every 1000 steps (configurable)
2. **Keep best model** - Based on validation metric
3. **Don't save too frequently** - Wastes disk space

### After Training

1. **Keep best model** - Delete intermediate checkpoints
2. **Save metadata** - Document training config, data used
3. **Test before deleting** - Ensure checkpoint loads correctly

### For Production

1. **Export model only** - Remove optimizer/scheduler
2. **Document config** - Save model architecture details
3. **Version checkpoints** - Keep track of different versions

## File Size Breakdown

Typical checkpoint (~600MB):

- Model weights: ~100MB (frozen encoder) or ~500MB (unfrozen)
- Optimizer state: ~200MB (AdamW stores 2x model params)
- Scheduler state: <1MB
- Metadata: <1KB

**To reduce size:** Save only `model_state_dict` for inference.

## Common Questions

### Q: Can I use this model for sentence embeddings?
**A:** Yes! Load the model and use the projection heads to get 512-dim embeddings.

### Q: Can I fine-tune on my own data?
**A:** Yes! Load the checkpoint and continue training with your dataset.

### Q: What if I change the config?
**A:** The checkpoint stores weights for a specific architecture. Changing `d_model`, `num_layers`, etc. will cause loading to fail unless you modify the architecture to match.

### Q: Can I extract just the sentence encoder?
**A:** Yes! The sentence encoder is part of the model:
```python
sentence_encoder = model.sentence_encoder
```

### Q: How do I know training was successful?
**A:** Check:
- `best_recall` in checkpoint (should be >0.5 for good performance)
- Loss decreased during training
- Evaluation metrics improved

## Summary

**What's saved:**
- ✅ Complete trained model (all layers, all weights)
- ✅ Optimizer state (for resuming training)
- ✅ Scheduler state (learning rate schedule)
- ✅ Training metadata (steps, epochs, metrics)

**What you can do:**
- ✅ Resume training
- ✅ Evaluate on new data
- ✅ Generate sentence embeddings
- ✅ Fine-tune on different tasks
- ✅ Use for inference/production

**Recommended:**
- Use `best_model.pt` for evaluation/inference
- Use `checkpoint_step_N.pt` for resuming training
- Save just model weights for production deployment

---

**See also:**
- [USE_MODEL.md](USE_MODEL.md) - How to use trained models (coming soon)
- [README.md](README.md) - Full documentation
