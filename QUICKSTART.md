# Quick Start Guide

Get up and running with Sentence JEPA in 5 minutes!

## Setup (2 minutes)

```bash
# Quick setup (recommended)
./setup.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: Install spaCy model for better sentence splitting
python -m spacy download en_core_web_sm
```

## Test the Pipeline (1 minute)

```bash
# Activate venv if not already active
source .venv/bin/activate

# Verify everything works
python test_pipeline.py
```

Expected output: All tests should pass ✓

## Train a Model (2 minutes)

```bash
# 1. Create sample training data
python scripts/train.py --create-sample-data

# 2. Train the model (will take a few minutes)
python scripts/train.py --config configs/base.yaml --data data/sample_data.txt

# For quick testing, you can modify configs/base.yaml to reduce num_epochs to 2
```

## Evaluate the Model

```bash
python scripts/eval_retrieval.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml \
    --data data/sample_data.txt
```

## Expected Results

After training on the sample data for a few epochs:
- **Recall@1**: Should be > 0.3 (30% accuracy)
- **Recall@5**: Should be > 0.6 (60% accuracy)

With more data and longer training:
- **Recall@1**: Can reach 0.7-0.9
- **Recall@5**: Can reach 0.9-0.95

## Next Steps

1. **Use your own data**: Replace `data/sample_data.txt` with your paragraphs
2. **Tune hyperparameters**: Edit `configs/base.yaml`
3. **Experiment**: Try different sentence encoders, model sizes, etc.

See [README.md](README.md) for detailed documentation.

## Troubleshooting

**Issue**: "ModuleNotFoundError"
- Solution: Run `pip install -r requirements.txt`

**Issue**: "CUDA out of memory"
- Solution: Reduce `batch_size` in `configs/base.yaml` to 8 or 4

**Issue**: Training is very slow
- Solution: Add `--device cuda` if you have a GPU
- Or reduce `num_layers` and `d_model` in config for a smaller model

## File Overview

- **configs/base.yaml**: All hyperparameters and settings
- **scripts/train.py**: Main training script
- **scripts/eval_retrieval.py**: Evaluation script
- **test_pipeline.py**: Test that everything works
- **models/**: Model architecture components
- **losses/**: Loss functions (JEPA + SIGReg)
- **data/**: Dataset and data loading
- **train/**: Training loop and evaluation

## Support

For detailed information, see [README.md](README.md)

For issues or questions, please open a GitHub issue.
