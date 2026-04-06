# Sentence JEPA - Quick Reference

## Setup

```bash
# Initial setup
./setup.sh

# Activate environment
source .venv/bin/activate

# Install HuggingFace datasets
pip install datasets
```

## Testing

```bash
# Test basic pipeline
python test_pipeline.py

# Test HuggingFace integration
python test_hf_datasets.py
```

## Training Commands

### Sample Data
```bash
# Create and train on sample data
python scripts/train.py --create-sample-data
python scripts/train.py
```

### Your Own Data
```bash
# Prepare: paragraphs separated by \n\n in a .txt file
python scripts/train.py --data path/to/your_data.txt
```

### HuggingFace Datasets

| Dataset | Command |
|---------|---------|
| **Wikipedia (test)** | `python scripts/train_hf.py --dataset wikipedia --streaming --max-samples 10000` |
| **Wikipedia (full)** | `python scripts/train_hf.py --dataset wikipedia --streaming` |
| **C4** | `python scripts/train_hf.py --dataset c4 --streaming --config configs/large_scale.yaml` |
| **BookCorpus** | `python scripts/train_hf.py --dataset bookcorpus --streaming` |
| **Custom** | `python scripts/train_hf.py --dataset custom --hf-name "user/dataset"` |
| **Multilingual** | `python scripts/train_hf.py --dataset wikipedia --wiki-lang es --streaming` |

### With GPU
```bash
python scripts/train_hf.py --dataset wikipedia --streaming --device cuda
```

## Evaluation

```bash
python scripts/eval_retrieval.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data data/your_data.txt
```

## Configuration

### Quick Changes (configs/base.yaml)

**Reduce memory usage:**
```yaml
training:
  batch_size: 8  # Reduce from 16
```

**Smaller/faster model:**
```yaml
model:
  paragraph_transformer:
    num_layers: 2  # Reduce from 4
    d_model: 512   # Reduce from 768
```

**Longer training:**
```yaml
training:
  num_epochs: 50  # Increase from 20
```

**Adjust learning rates:**
```yaml
training:
  lr_sentence_encoder: 1.0e-5  # Lower for pretrained
  lr_rest: 1.0e-4              # Higher for new params
```

## Monitoring

**Check training progress:**
- Checkpoints saved to: `checkpoints/`
- Best model: `checkpoints/best_model.pt`
- Loss should decrease over time
- Recall@1 should increase (>0.5 is good, >0.7 is great)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce `batch_size` in config |
| Training slow | Use `--device cuda`, increase `batch_size` |
| Import error | `pip install -r requirements.txt` |
| Stack size error | **FIXED** - Collator bug resolved |
| Poor quality | Check data, increase `min_sentences` |
| Not converging | Increase `warmup_steps`, reduce LR |

## File Structure

```
SentenceJEPA/
├── configs/
│   ├── base.yaml          # Default config
│   └── large_scale.yaml   # For large datasets
├── scripts/
│   ├── train.py           # Train on text files
│   ├── train_hf.py        # Train on HuggingFace datasets
│   └── eval_retrieval.py  # Evaluate model
├── data/                  # Data pipeline
├── models/                # Model architecture
├── losses/                # Loss functions
└── train/                 # Training loop
```

## Documentation

- **[README.md](README.md)** - Full documentation
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute guide
- **[HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md)** - HF datasets guide
- **[LARGE_DATASET_GUIDE.md](LARGE_DATASET_GUIDE.md)** - Scaling guide

## Key Metrics

**Training:**
- JEPA loss: Should decrease to ~0.1-0.5
- SIGReg loss: Should stabilize ~0.5-2.0
- Total loss: Weighted combination

**Evaluation:**
- Recall@1: >0.5 is good, >0.7 is great
- Recall@5: Should be >0.8
- MRR: Mean reciprocal rank

## Common Workflows

### Quick Experiment
```bash
# 1. Test on small data
python scripts/train_hf.py --dataset wikipedia --streaming --max-samples 1000

# 2. Check it works
python test_pipeline.py

# 3. Scale up
python scripts/train_hf.py --dataset wikipedia --streaming --max-samples 50000
```

### Production Training
```bash
# 1. Large dataset
python scripts/train_hf.py \
    --dataset c4 \
    --streaming \
    --config configs/large_scale.yaml \
    --device cuda

# 2. Monitor checkpoints
ls -lh checkpoints/

# 3. Evaluate best model
python scripts/eval_retrieval.py \
    --checkpoint checkpoints/best_model.pt \
    --data test_data.txt
```

### Multi-GPU Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_hf.py \
    --dataset wikipedia \
    --streaming \
    --config configs/large_scale.yaml
```

## Tips

1. **Start small**: Use `--max-samples 1000` to test
2. **Use streaming**: For datasets >1GB
3. **Monitor GPU**: `nvidia-smi` to check utilization
4. **Save often**: Set `save_every: 500` in config
5. **Check data**: Print first few samples to verify quality

## Performance

| Hardware | Batch Size | Speed (samples/sec) |
|----------|-----------|---------------------|
| CPU (16 cores) | 8 | ~10 |
| RTX 3090 | 32 | ~200 |
| A100 | 64 | ~400 |
| 4x A100 | 256 | ~1500 |

## Getting Help

1. Check documentation in this directory
2. Review inline code comments
3. Run test scripts to verify setup
4. Check HuggingFace datasets docs for data issues

---

**Quick Links:**
- [Full README](README.md)
- [HuggingFace Guide](HUGGINGFACE_GUIDE.md)
- [Large Dataset Guide](LARGE_DATASET_GUIDE.md)
