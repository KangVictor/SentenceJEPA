# Hierarchical Sentence JEPA with SIGReg

A PyTorch implementation of a hierarchical sentence-level Joint-Embedding Predictive Architecture (JEPA) with Signal Regularization (SIGReg).

## Overview

This model learns contextual sentence representations by:
1. Splitting paragraphs into sentences
2. Encoding sentences with a pretrained transformer (e.g., RoBERTa)
3. Processing sentence embeddings with a paragraph-level transformer
4. Masking one sentence and predicting its contextualized representation
5. Training with normalized MSE loss + SIGReg regularization

**Key Features:**
- No token reconstruction - predicts latent embeddings directly
- Two-branch architecture (target + predictor)
- SIGReg regularization for better embedding distributions
- Flexible sentence encoder (HuggingFace models)
- Evaluation via masked sentence retrieval

## Architecture

```
Input: Paragraph text
    ↓
Sentence Splitting (spaCy or regex)
    ↓
Sentence Encoder (RoBERTa) → [B, S, D]
    ↓
    ├─→ Target Branch: Paragraph Transformer → Projection Head → z_target
    └─→ Predictor Branch: [MASK] one sentence → Paragraph Transformer → Projection Head → z_pred
    ↓
Loss = normalized_MSE(z_pred, z_target) + λ * SIGReg(z_target)
```

## Installation

### Quick Setup (Recommended)

```bash
cd SentenceJEPA
./setup.sh
```

This script will:
- Create a virtual environment (`.venv`)
- Install all dependencies
- Optionally install spaCy model for sentence splitting

### Manual Setup

```bash
cd SentenceJEPA
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: Install spaCy model for better sentence splitting
python -m spacy download en_core_web_sm
```

If you skip the spaCy model, the code will fall back to regex-based sentence splitting.

## Project Structure

```
SentenceJEPA/
├── configs/
│   └── base.yaml              # Configuration file
├── data/
│   ├── dataset.py             # Dataset with sentence splitting
│   └── collator.py            # Batch collation and masking
├── models/
│   ├── sentence_encoder.py   # HuggingFace sentence encoder wrapper
│   ├── paragraph_transformer.py  # Paragraph-level transformer
│   ├── projector.py           # MLP projection heads
│   └── sentence_jepa.py       # Main model
├── losses/
│   ├── jepa_loss.py           # Normalized MSE loss
│   ├── sigreg.py              # SIGReg regularization
│   └── combined_loss.py       # Combined loss function
├── train/
│   ├── trainer.py             # Training loop
│   ├── evaluation.py          # Evaluation functions
│   └── metrics.py             # Recall@K metrics
└── scripts/
    ├── train.py               # Training script
    └── eval_retrieval.py      # Evaluation script
```

## Quick Start

### 1. Create sample training data

```bash
python scripts/train.py --create-sample-data --data data/sample_data.txt
```

This creates a sample dataset with AI/ML-related paragraphs.

### 2. Train the model

```bash
python scripts/train.py --config configs/base.yaml --data data/sample_data.txt
```

**Training options:**
- `--config`: Path to configuration file (default: `configs/base.yaml`)
- `--data`: Path to training data (paragraphs separated by `\n\n`)
- `--device`: Device to use (`cuda` or `cpu`)

**What happens during training:**
- Model checkpoints saved to `checkpoints/` every 1000 steps
- Best model (by Recall@1) saved as `checkpoints/best_model.pt`
- Evaluation runs every 500 steps on validation set

### 3. Evaluate the model

```bash
python scripts/eval_retrieval.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data data/sample_data.txt
```

**Evaluation metrics:**
- **Recall@K**: Fraction of queries where correct sentence is in top K retrieved
- **MRR**: Mean Reciprocal Rank

## Configuration

Edit `configs/base.yaml` to customize:

### Model Architecture

```yaml
model:
  sentence_encoder:
    model_name: "roberta-base"  # Any HuggingFace model
    frozen: true                # Freeze encoder weights
    pooling: "mean"             # Pooling: mean, cls, or max

  paragraph_transformer:
    d_model: 768
    nhead: 8
    num_layers: 4
    dim_feedforward: 2048
    dropout: 0.1

  projection:
    hidden_dim: 1024
    output_dim: 512             # Final embedding dimension
    dropout: 0.1
```

### Loss Configuration

```yaml
loss:
  lambda_sigreg: 0.1            # Weight for SIGReg loss
  sigreg:
    num_projections: 32         # Number of random projections
    projection_dim: 128
```

### Training Configuration

```yaml
training:
  batch_size: 16
  num_epochs: 20
  gradient_clip: 1.0

  # Dual learning rates
  lr_sentence_encoder: 1.0e-5   # Lower LR for pretrained encoder
  lr_rest: 1.0e-4               # Higher LR for new parameters

  # Scheduler
  warmup_steps: 500
  scheduler: "cosine"
```

## Data Format

Training data should be a text file with paragraphs separated by double newlines:

```
This is the first paragraph. It has multiple sentences. Each sentence will be encoded separately.

This is the second paragraph. It also contains several sentences. The model learns to predict masked sentences.

This is the third paragraph. Continue with more examples...
```

**Requirements:**
- Minimum 3 sentences per paragraph (configurable via `min_sentences`)
- Will automatically filter out shorter paragraphs

## Training Details

### Optimization
- **Optimizer**: AdamW with weight decay
- **Learning rates**: Separate for sentence encoder (1e-5) and rest (1e-4)
- **Scheduler**: Cosine annealing with linear warmup
- **Gradient clipping**: Max norm 1.0

### Loss Function
- **JEPA Loss**: Normalized MSE between predicted and target embeddings
  ```
  z_pred_norm = normalize(z_pred)
  z_target_norm = normalize(z_target)
  loss_jepa = ||z_pred_norm - z_target_norm||²
  ```

- **SIGReg**: Regularizes target embeddings to follow standard Gaussian
  - Uses random projections (Epps-Pulley style)
  - Encourages well-behaved embedding space

- **Total**: `loss = loss_jepa + λ * loss_sigreg`

### Masking Strategy
- Randomly select one sentence per paragraph to mask
- Option to prefer interior sentences (not first/last) with probability 0.8
- Mask embedding is learned during training

## Evaluation

The model is evaluated on **masked sentence retrieval**:

1. For each paragraph, mask one sentence
2. Use predictor branch to generate embedding for masked position
3. Use target branch to generate embeddings for all sentences
4. Retrieve correct sentence from candidate pool using cosine similarity
5. Compute Recall@1, Recall@5, Recall@10

**Interpreting results:**
- **Recall@1 = 0.8** means 80% of the time, the correct sentence is the top match
- Good performance: Recall@1 > 0.5, Recall@5 > 0.8

## Using Your Own Data

To train on custom data:

### Option 1: Text file with paragraphs

```python
# Prepare your data as paragraphs separated by \n\n
with open('my_data.txt', 'w') as f:
    for paragraph in my_paragraphs:
        f.write(paragraph + '\n\n')

# Train
python scripts/train.py --data my_data.txt
```

### Option 2: Programmatic dataset

```python
from data import ParagraphDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = ParagraphDataset.from_list(
    paragraphs=my_paragraphs,
    min_sentences=3,
    max_sentences=10,
)

# Rest of training code...
```

### Large-Scale Training

For datasets with millions of paragraphs, see the **[Large Dataset Guide](LARGE_DATASET_GUIDE.md)** which covers:
- Streaming datasets for memory efficiency
- Multi-GPU and distributed training
- Mixed precision (FP16) training
- Gradient accumulation strategies
- Performance optimization tips

## Extending the Model

### Use a different sentence encoder

```yaml
# In configs/base.yaml
model:
  sentence_encoder:
    model_name: "sentence-transformers/all-mpnet-base-v2"
    frozen: false  # Fine-tune the encoder
```

### Adjust model size

For smaller/faster model:
```yaml
paragraph_transformer:
  d_model: 512      # Reduce from 768
  num_layers: 2     # Reduce from 4
  nhead: 8
```

For larger model:
```yaml
paragraph_transformer:
  d_model: 1024
  num_layers: 6
  nhead: 16
```

### Add more regularization

```yaml
loss:
  lambda_sigreg: 0.2  # Increase SIGReg weight
  sigreg:
    num_projections: 64  # More projections
```

## Testing Individual Components

Each module includes test code. Run with:

```bash
# Test losses
python losses/jepa_loss.py
python losses/sigreg.py

# Test model components
python models/sentence_encoder.py
python models/paragraph_transformer.py
python models/projector.py
python models/sentence_jepa.py

# Test data pipeline
python data/dataset.py
python data/collator.py

# Test metrics
python train/metrics.py
```

## Common Issues

### Issue: "spaCy model not found"
**Solution**: Either install spaCy model or ignore (will use regex fallback)
```bash
python -m spacy download en_core_web_sm
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size or model size
```yaml
training:
  batch_size: 8  # Reduce from 16
```

### Issue: Training is slow
**Solution**:
- Use GPU if available
- Reduce `num_layers` in paragraph transformer
- Keep sentence encoder frozen
- Use smaller sentence encoder (e.g., distilroberta-base)

## Performance Tips

1. **GPU Training**: Use CUDA for ~10x speedup
   ```bash
   python scripts/train.py --device cuda
   ```

2. **Frozen Encoder**: Keep sentence encoder frozen for faster training and lower memory

3. **Batch Size**: Increase if you have GPU memory available

4. **Data Size**: More diverse paragraphs → better representations

## Citation

If you use this code, please cite the relevant papers:
- JEPA: Yann LeCun's vision paper on Joint-Embedding Predictive Architectures
- SIGReg: Signal regularization techniques for representation learning

## License

MIT License - feel free to use and modify.

## Contact

For questions or issues, please open a GitHub issue or contact the author.
