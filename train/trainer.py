"""
Trainer for Sentence JEPA model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Optional
from pathlib import Path
import math
from tqdm import tqdm
import json

from losses import combined_loss
from losses.sigreg import SIGReg
from .evaluation import evaluate_retrieval


class Trainer:
    """
    Trainer for Hierarchical Sentence JEPA model.

    Handles:
    - Training loop with AdamW optimizer
    - Different learning rates for sentence encoder vs rest
    - Cosine scheduler with warmup
    - Gradient clipping
    - Logging and checkpointing
    - Evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        # Optimizer config
        lr_sentence_encoder: float = 1e-5,
        lr_rest: float = 1e-4,
        weight_decay: float = 0.01,
        # Scheduler config
        num_epochs: int = 20,
        warmup_steps: int = 500,
        # Loss config
        lambda_sigreg: float = 0.1,
        num_projections: int = 32,
        projection_dim: int = 128,
        # Training config
        gradient_clip: float = 1.0,
        device: torch.device = None,
        # Logging
        log_every: int = 50,
        eval_every: int = 500,
        save_every: int = 1000,
        checkpoint_dir: str = "./checkpoints",
        # Evaluation
        recall_k: list = [1, 5, 10],
    ):
        """
        Args:
            model: HierarchicalSentenceJEPA model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            lr_sentence_encoder: Learning rate for sentence encoder
            lr_rest: Learning rate for other parameters
            weight_decay: Weight decay for AdamW
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for scheduler
            lambda_sigreg: Weight for SIGReg loss
            num_projections: Number of projections for SIGReg
            projection_dim: Dimension of projections for SIGReg
            gradient_clip: Gradient clipping value
            device: Device to train on
            log_every: Log every N steps
            eval_every: Evaluate every N steps
            save_every: Save checkpoint every N steps
            checkpoint_dir: Directory for checkpoints
            recall_k: K values for Recall@K metric
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Loss
        self.lambda_sigreg = lambda_sigreg
        self.sigreg_module = SIGReg(
            embedding_dim=model.target_head.net[-1].out_features,  # Output dim of projection head
            num_projections=num_projections,
            projection_dim=projection_dim,
        ).to(self.device)

        # Optimizer with parameter groups
        self.optimizer = self._create_optimizer(
            lr_sentence_encoder=lr_sentence_encoder,
            lr_rest=lr_rest,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.total_steps = num_epochs * len(train_dataloader)
        self.scheduler = self._create_scheduler()

        # Training config
        self.gradient_clip = gradient_clip

        # Logging
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.recall_k = recall_k

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_recall = 0.0

        print(f"Trainer initialized on device: {self.device}")
        print(f"Total training steps: {self.total_steps}")
        print(f"Warmup steps: {self.warmup_steps}")

    def _create_optimizer(
        self,
        lr_sentence_encoder: float,
        lr_rest: float,
        weight_decay: float,
    ) -> AdamW:
        """Create optimizer with parameter groups."""
        # Separate parameters for sentence encoder vs rest
        sentence_encoder_params = []
        rest_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'sentence_encoder' in name:
                sentence_encoder_params.append(param)
            else:
                rest_params.append(param)

        param_groups = [
            {'params': sentence_encoder_params, 'lr': lr_sentence_encoder},
            {'params': rest_params, 'lr': lr_rest},
        ]

        print(f"Sentence encoder params: {sum(p.numel() for p in sentence_encoder_params):,}")
        print(f"Rest params: {sum(p.numel() for p in rest_params):,}")

        return AdamW(param_groups, weight_decay=weight_decay)

    def _create_scheduler(self) -> LambdaLR:
        """Create cosine scheduler with warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Cosine decay
                progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Run training loop."""
        print("\n=== Starting Training ===\n")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self._train_epoch()

            # Evaluate at end of epoch
            if self.val_dataloader is not None:
                self._evaluate()

        print("\n=== Training Complete ===\n")

    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        epoch_loss = 0.0
        epoch_jepa_loss = 0.0
        epoch_sigreg_loss = 0.0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch+1}/{self.num_epochs}")

        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            sentence_mask = batch['sentence_mask'].to(self.device)
            mask_idx = batch['mask_idx'].to(self.device)

            # Forward pass
            z_pred, z_target = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence_mask=sentence_mask,
                mask_idx=mask_idx,
            )

            # Compute loss
            loss, loss_dict = combined_loss(
                pred_embeddings=z_pred,
                target_embeddings=z_target,
                sigreg_module=self.sigreg_module,
                lambda_sigreg=self.lambda_sigreg,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()
            self.scheduler.step()

            # Update stats
            epoch_loss += loss_dict['total']
            epoch_jepa_loss += loss_dict['jepa']
            epoch_sigreg_loss += loss_dict['sigreg']
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'jepa': f"{loss_dict['jepa']:.4f}",
                'sig': f"{loss_dict['sigreg']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # Logging
            if self.global_step % self.log_every == 0:
                self._log(loss_dict)

            # Evaluation
            if self.val_dataloader is not None and self.global_step % self.eval_every == 0:
                self._evaluate()
                self.model.train()

            # Checkpointing
            if self.global_step % self.save_every == 0:
                self._save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

        # Epoch summary
        num_batches = len(self.train_dataloader)
        print(f"\nEpoch {self.epoch+1} Summary:")
        print(f"  Avg Loss: {epoch_loss / num_batches:.4f}")
        print(f"  Avg JEPA Loss: {epoch_jepa_loss / num_batches:.4f}")
        print(f"  Avg SIGReg Loss: {epoch_sigreg_loss / num_batches:.4f}")

    def _log(self, loss_dict: Dict):
        """Log training metrics."""
        # In a real implementation, you'd log to tensorboard/wandb
        pass

    def _evaluate(self):
        """Run evaluation."""
        print("\n--- Evaluation ---")
        metrics = evaluate_retrieval(
            model=self.model,
            dataloader=self.val_dataloader,
            device=self.device,
            k_values=self.recall_k,
            max_batches=None,
        )

        print(f"Step {self.global_step} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Save best model
        if 'recall@1' in metrics and metrics['recall@1'] > self.best_recall:
            self.best_recall = metrics['recall@1']
            self._save_checkpoint("best_model.pt")
            print(f"  New best model! Recall@1: {self.best_recall:.4f}")

        print()

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_recall': self.best_recall,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_recall = checkpoint['best_recall']
        print(f"Checkpoint loaded from: {path}")


if __name__ == "__main__":
    print("Trainer module ready!")
