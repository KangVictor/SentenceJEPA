"""
Profile training to identify bottlenecks.

Usage:
    python scripts/profile_training.py --config configs/base.yaml
"""

import argparse
import yaml
import torch
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HierarchicalSentenceJEPA
from data import SentenceJEPACollator


def check_gpu():
    """Check GPU availability and usage."""
    print("="*60)
    print("GPU Information")
    print("="*60)

    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Current allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Current reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("✗ CUDA not available - training on CPU!")

    return torch.cuda.is_available()


def profile_model(config):
    """Profile model to check memory usage."""
    print("\n" + "="*60)
    print("Model Profiling")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = HierarchicalSentenceJEPA(
        sentence_encoder_name=config['model']['sentence_encoder']['model_name'],
        sentence_encoder_frozen=config['model']['sentence_encoder']['frozen'],
        d_model=config['model']['paragraph_transformer']['d_model'],
        nhead=config['model']['paragraph_transformer']['nhead'],
        num_layers=config['model']['paragraph_transformer']['num_layers'],
        projection_output_dim=config['model']['projection']['output_dim'],
    )

    model = model.to(device)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"Memory after model load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model, device


def profile_batch(model, device, batch_size, num_sentences=5, seq_length=32):
    """Profile a single batch."""
    print("\n" + "="*60)
    print(f"Batch Profiling (batch_size={batch_size})")
    print("="*60)

    # Create dummy batch
    input_ids = torch.randint(0, 1000, (batch_size, num_sentences, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, num_sentences, seq_length).to(device)
    sentence_mask = torch.ones(batch_size, num_sentences).to(device)
    mask_idx = torch.randint(0, num_sentences, (batch_size,)).to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated(0) / 1e9

    # Forward pass
    start = time.time()
    with torch.no_grad():
        z_pred, z_target = model(input_ids, attention_mask, sentence_mask, mask_idx)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    forward_time = time.time() - start

    if torch.cuda.is_available():
        end_mem = torch.cuda.memory_allocated(0) / 1e9
        mem_used = end_mem - start_mem
        print(f"Memory used by batch: {mem_used:.2f} GB")
        print(f"Total memory allocated: {end_mem:.2f} GB")

    print(f"Forward pass time: {forward_time*1000:.2f} ms")
    print(f"Throughput: {batch_size/forward_time:.2f} samples/sec")

    return forward_time


def recommend_batch_size(config):
    """Recommend optimal batch size."""
    print("\n" + "="*60)
    print("Batch Size Recommendations")
    print("="*60)

    if not torch.cuda.is_available():
        print("Running on CPU - keep batch size small (8-16)")
        return

    device = torch.device('cuda')

    # Create model
    model = HierarchicalSentenceJEPA(
        sentence_encoder_name=config['model']['sentence_encoder']['model_name'],
        sentence_encoder_frozen=config['model']['sentence_encoder']['frozen'],
        d_model=config['model']['paragraph_transformer']['d_model'],
        nhead=config['model']['paragraph_transformer']['nhead'],
        num_layers=config['model']['paragraph_transformer']['num_layers'],
        projection_output_dim=config['model']['projection']['output_dim'],
    ).to(device)

    # Try different batch sizes
    batch_sizes = [8, 16, 32, 64, 128]

    print("Testing different batch sizes...")
    results = []

    for bs in batch_sizes:
        try:
            torch.cuda.empty_cache()

            # Create dummy batch
            input_ids = torch.randint(0, 1000, (bs, 5, 32)).to(device)
            attention_mask = torch.ones(bs, 5, 32).to(device)
            sentence_mask = torch.ones(bs, 5).to(device)
            mask_idx = torch.randint(0, 5, (bs,)).to(device)

            # Forward + backward
            z_pred, z_target = model(input_ids, attention_mask, sentence_mask, mask_idx)
            loss = (z_pred - z_target).pow(2).sum()
            loss.backward()

            torch.cuda.synchronize()

            mem_used = torch.cuda.memory_allocated(0) / 1e9
            results.append((bs, mem_used, True))
            print(f"  Batch size {bs:3d}: {mem_used:.2f} GB ✓")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Batch size {bs:3d}: OOM ✗")
                results.append((bs, None, False))
                break
            else:
                raise

    # Find optimal
    valid_results = [(bs, mem) for bs, mem, valid in results if valid]
    if valid_results:
        max_bs = max(bs for bs, _ in valid_results)
        print(f"\n✓ Maximum batch size: {max_bs}")
        print(f"✓ Recommended: {max_bs} (for training with gradients)")
        print(f"\nUpdate your config:")
        print(f"  training:")
        print(f"    batch_size: {max_bs}")


def main():
    parser = argparse.ArgumentParser(description='Profile training performance')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Check GPU
    has_gpu = check_gpu()

    # Profile model
    model, device = profile_model(config)

    # Profile batch with current config
    current_batch_size = config['training']['batch_size']
    profile_batch(model, device, current_batch_size)

    # Recommend optimal batch size
    if has_gpu:
        recommend_batch_size(config)

    print("\n" + "="*60)
    print("Profiling Complete")
    print("="*60)


if __name__ == "__main__":
    main()
