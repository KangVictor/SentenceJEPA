# Training on Large Datasets

This guide explains how to efficiently train Sentence JEPA on large-scale datasets.

## Table of Contents
1. [Data Preparation](#data-preparation)
2. [Memory Optimization](#memory-optimization)
3. [Training Strategies](#training-strategies)
4. [Distributed Training](#distributed-training)
5. [Example Workflows](#example-workflows)

---

## Data Preparation

### Option 1: Text File (Simple)

For datasets up to ~1GB, use a simple text file:

```python
# prepare_data.py
paragraphs = []
# Load from your source (books, articles, web pages, etc.)
for document in your_documents:
    # Split document into paragraphs
    doc_paragraphs = document.split('\n\n')
    paragraphs.extend(doc_paragraphs)

# Save to file
with open('data/train_data.txt', 'w') as f:
    f.write('\n\n'.join(paragraphs))
```

Then train:
```bash
python scripts/train.py --data data/train_data.txt
```

### Option 2: Streaming Dataset (Memory-Efficient)

For very large datasets (>1GB), use a streaming approach:

```python
# data/streaming_dataset.py
from torch.utils.data import IterableDataset
import json

class StreamingParagraphDataset(IterableDataset):
    """Stream paragraphs from a large file without loading all into memory."""

    def __init__(self, file_path, min_sentences=3, max_sentences=10):
        self.file_path = file_path
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def __iter__(self):
        with open(self.file_path, 'r') as f:
            for line in f:
                # Assuming JSONL format: {"text": "paragraph text..."}
                data = json.loads(line)
                paragraph = data['text']

                # Process paragraph
                from data.dataset import split_into_sentences
                sentences = split_into_sentences(paragraph, use_spacy=False)

                if len(sentences) >= self.min_sentences:
                    if len(sentences) > self.max_sentences:
                        sentences = sentences[:self.max_sentences]

                    yield {
                        'paragraph': paragraph,
                        'sentences': sentences,
                    }

# Usage
dataset = StreamingParagraphDataset('data/large_dataset.jsonl')
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)
```

### Option 3: HuggingFace Datasets

Use HuggingFace datasets for popular corpora:

```python
from datasets import load_dataset

# Load a dataset (e.g., Wikipedia, C4, BookCorpus)
hf_dataset = load_dataset('wikipedia', '20220301.en', streaming=True)

# Create wrapper
class HFParagraphDataset:
    def __init__(self, hf_dataset, min_sentences=3):
        self.hf_dataset = hf_dataset
        self.min_sentences = min_sentences

    def __iter__(self):
        for item in self.hf_dataset:
            text = item['text']
            # Split into paragraphs
            for paragraph in text.split('\n\n'):
                sentences = split_into_sentences(paragraph)
                if len(sentences) >= self.min_sentences:
                    yield {'paragraph': paragraph, 'sentences': sentences}

dataset = HFParagraphDataset(hf_dataset['train'])
```

---

## Memory Optimization

### 1. Reduce Batch Size

```yaml
# configs/base.yaml
training:
  batch_size: 8  # Reduce from 16
  # Or even 4 for very large models
```

### 2. Use Gradient Accumulation

Simulate larger batches without memory overhead:

```python
# In train/trainer.py, modify training loop:
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss, _ = compute_loss(...)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Add to config:
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size: 16
```

### 3. Mixed Precision Training (FP16)

Use automatic mixed precision for 2x speedup and 2x less memory:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():  # Use FP16
        z_pred, z_target = model(...)
        loss, _ = combined_loss(...)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 4. Smaller Model

```yaml
model:
  paragraph_transformer:
    d_model: 512       # Reduce from 768
    num_layers: 2      # Reduce from 4
    nhead: 8
    dim_feedforward: 1024  # Reduce from 2048
```

### 5. Freeze Sentence Encoder

```yaml
model:
  sentence_encoder:
    frozen: true  # Don't update pretrained weights
```

This saves ~125M parameters from backprop.

---

## Training Strategies

### 1. Curriculum Learning

Start with easier examples (short paragraphs), gradually increase difficulty:

```python
# Phase 1: Short paragraphs (3-5 sentences)
dataset_phase1 = ParagraphDataset(
    paragraphs=paragraphs,
    min_sentences=3,
    max_sentences=5,
)
trainer.train()

# Phase 2: Medium paragraphs (5-8 sentences)
dataset_phase2 = ParagraphDataset(
    paragraphs=paragraphs,
    min_sentences=3,
    max_sentences=8,
)
trainer.train()
```

### 2. Learning Rate Scheduling

For large datasets, use a longer warmup:

```yaml
training:
  warmup_steps: 2000  # Increase from 500
  num_epochs: 10      # May need fewer epochs with more data
```

### 3. Early Stopping

Monitor validation loss and stop when not improving:

```python
# In trainer
best_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    val_loss = evaluate(...)

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        save_checkpoint('best_model.pt')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping!")
        break
```

### 4. Checkpointing Strategy

For long training runs, save frequently:

```yaml
training:
  save_every: 500      # Every 500 steps
  eval_every: 1000     # Evaluate every 1000 steps
```

---

## Distributed Training

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --device cuda
```

### Multiple GPUs (DataParallel)

```python
# In scripts/train.py
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

### Multiple GPUs (DistributedDataParallel - Recommended)

Create `scripts/train_distributed.py`:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_distributed(rank, world_size, config):
    setup(rank, world_size)

    # Create model and move to GPU
    model = HierarchicalSentenceJEPA(**config)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Use DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, ...)

    # Train...

    dist.destroy_process_group()

# Launch
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train_distributed,
        args=(world_size, config),
        nprocs=world_size,
    )
```

Run:
```bash
python scripts/train_distributed.py
```

### Multi-Node Training

Use `torchrun`:
```bash
# On each node
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=<main_node_ip> --master_port=29500 \
         scripts/train_distributed.py
```

---

## Example Workflows

### Example 1: Wikipedia Training

```bash
# 1. Download Wikipedia dump
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# 2. Extract text (using WikiExtractor)
pip install wikiextractor
wikiextractor enwiki-latest-pages-articles.xml.bz2 -o wiki_text

# 3. Prepare paragraphs
python prepare_wikipedia.py  # Convert to paragraph format

# 4. Train
python scripts/train.py \
    --data data/wikipedia_paragraphs.txt \
    --config configs/large_scale.yaml
```

### Example 2: Custom Dataset (1M+ Paragraphs)

```python
# prepare_large_dataset.py
import json

# Convert your data to JSONL
with open('data/train_large.jsonl', 'w') as f:
    for paragraph in your_large_corpus:
        json.dump({'text': paragraph}, f)
        f.write('\n')

# Use streaming dataset
from data.streaming_dataset import StreamingParagraphDataset
dataset = StreamingParagraphDataset('data/train_large.jsonl')

# Train with reduced batch size
# In configs/large_scale.yaml:
# batch_size: 4
# gradient_accumulation_steps: 8
```

### Example 3: Multi-GPU Training

```yaml
# configs/multi_gpu.yaml
training:
  batch_size: 32  # Per GPU
  num_epochs: 5

model:
  sentence_encoder:
    frozen: true  # Keep frozen for efficiency
```

```bash
# Train on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py \
    --config configs/multi_gpu.yaml \
    --data data/large_dataset.txt
```

---

## Performance Benchmarks

Approximate training speed (depends on hardware):

| Setup | Batch Size | Paragraphs/sec | Time for 1M paragraphs |
|-------|-----------|----------------|------------------------|
| CPU (16 cores) | 8 | ~10 | ~28 hours |
| Single GPU (RTX 3090) | 32 | ~200 | ~1.4 hours |
| 4x GPU (A100) | 128 | ~1000 | ~17 minutes |
| 8x GPU (A100) + FP16 | 256 | ~2500 | ~7 minutes |

---

## Troubleshooting

### Issue: OOM (Out of Memory)
**Solutions:**
1. Reduce `batch_size`
2. Reduce `max_sentences` or `max_tokens_per_sentence`
3. Use gradient accumulation
4. Enable FP16 training
5. Reduce model size

### Issue: Training is slow
**Solutions:**
1. Use GPU instead of CPU
2. Increase `batch_size` (if memory allows)
3. Set `num_workers > 0` in DataLoader
4. Keep sentence encoder frozen
5. Use FP16 training

### Issue: Model not converging
**Solutions:**
1. Increase `warmup_steps`
2. Reduce learning rate
3. Check data quality (need diverse paragraphs)
4. Increase `lambda_sigreg` for better regularization

### Issue: Validation metrics not improving
**Solutions:**
1. Train longer
2. Use more diverse data
3. Unfreeze sentence encoder (fine-tune)
4. Reduce regularization

---

## Recommended Configs

### Small Dataset (<10K paragraphs)
```yaml
training:
  batch_size: 16
  num_epochs: 20
model:
  sentence_encoder:
    frozen: true
  paragraph_transformer:
    num_layers: 4
```

### Medium Dataset (10K-100K paragraphs)
```yaml
training:
  batch_size: 32
  num_epochs: 10
model:
  sentence_encoder:
    frozen: false  # Fine-tune
    lr_sentence_encoder: 5e-6
  paragraph_transformer:
    num_layers: 6
```

### Large Dataset (>100K paragraphs)
```yaml
training:
  batch_size: 64
  num_epochs: 3
  warmup_steps: 2000
model:
  sentence_encoder:
    frozen: true  # Keep frozen
  paragraph_transformer:
    num_layers: 8
    d_model: 1024
```

---

## Next Steps

1. Start with the small config on a subset of your data
2. Monitor metrics and adjust hyperparameters
3. Scale up batch size and model size as needed
4. Use distributed training for very large datasets

For questions, see the main [README.md](README.md) or open an issue.
