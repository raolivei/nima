"""
Training utilities and trainer class for LLM training.

This module provides a comprehensive training framework with:
- Training loops
- Gradient accumulation
- Mixed precision training
- Checkpointing
- Learning rate scheduling
- Logging and monitoring
"""

import os
import time
import math
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training hyperparameters
    num_epochs: int = 10
    learning_rate: float = 3e-4
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Optimization
    optimizer_type: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"  # cosine, linear, constant
    min_lr: float = 0.0
    
    # Mixed precision training
    use_mixed_precision: bool = True
    
    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 3
    checkpoint_dir: str = "experiments/checkpoints"
    
    # Evaluation
    eval_steps: int = 500
    eval_accumulation_steps: int = 1
    
    # Logging
    logging_steps: int = 100
    log_dir: str = "experiments/logs"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seed
    seed: int = 42


class Trainer:
    """
    Trainer class for training language models.
    
    Handles training loops, optimization, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        callbacks: Optional[list] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            optimizer: Optimizer (will be created if not provided)
            lr_scheduler: Learning rate scheduler (will be created if not provided)
            callbacks: List of callback functions
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.callbacks = callbacks or []
        
        # Set device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
            
        # Create learning rate scheduler if not provided
        if lr_scheduler is None:
            self.lr_scheduler = self._create_scheduler()
        else:
            self.lr_scheduler = lr_scheduler
            
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.training_logs = []
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        # Separate parameters with and without weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        if self.config.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
        elif self.config.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
        elif self.config.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.config.optimizer_type}")
            
        return optimizer
        
    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler based on config."""
        num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        num_training_steps = num_training_steps // self.config.gradient_accumulation_steps
        
        if self.config.lr_scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - self.config.warmup_steps,
                eta_min=self.config.min_lr
            )
        elif self.config.lr_scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr / self.config.learning_rate,
                total_iters=num_training_steps - self.config.warmup_steps
            )
        elif self.config.lr_scheduler_type == "constant":
            return torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=num_training_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.config.lr_scheduler_type}")
            
    def _get_lr(self) -> float:
        """Get current learning rate with warmup."""
        if self.global_step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (self.global_step / self.config.warmup_steps)
        else:
            return self.optimizer.param_groups[0]['lr']
            
    def train(self) -> Dict[str, Any]:
        """
        Run training loop.
        
        Returns:
            Dictionary with training statistics
        """
        print("=" * 80)
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Total training steps: {len(self.train_dataloader) * self.config.num_epochs}")
        print("=" * 80)
        
        total_start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training epoch
            train_loss = self.train_epoch()
            
            # Evaluation
            eval_loss = None
            if self.eval_dataloader is not None:
                eval_loss = self.evaluate()
                
            epoch_time = time.time() - epoch_start_time
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if eval_loss is not None:
                print(f"  Eval Loss: {eval_loss:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(f"epoch_{epoch + 1}")
            
        total_time = time.time() - total_start_time
        
        print("\n" + "=" * 80)
        print(f"Training completed in {total_time:.2f}s")
        print("=" * 80)
        
        return {
            'total_time': total_time,
            'final_train_loss': train_loss,
            'final_eval_loss': eval_loss,
            'training_logs': self.training_logs
        }
        
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_mixed_precision):
                loss = self.training_step(batch)
                loss = loss / self.config.gradient_accumulation_steps
                
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                # Learning rate scheduling (after warmup)
                if self.global_step >= self.config.warmup_steps:
                    self.lr_scheduler.step()
                    
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_training_step(loss.item() * self.config.gradient_accumulation_steps)
                    
                # Evaluation
                if (self.eval_dataloader is not None and 
                    self.global_step % self.config.eval_steps == 0):
                    eval_loss = self.evaluate()
                    self.model.train()
                    
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
                    
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
        return total_loss / num_batches
        
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss tensor
        """
        outputs = self.model(batch['input_ids'], attention_mask=batch.get('attention_mask'))
        
        # Calculate loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))
        
        return loss
        
    def evaluate(self) -> float:
        """
        Evaluate the model.
        
        Returns:
            Average evaluation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                        
                with autocast(enabled=self.config.use_mixed_precision):
                    loss = self.training_step(batch)
                    
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        
        # Log evaluation
        print(f"\nEvaluation at step {self.global_step}: Loss = {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < self.best_eval_loss:
            self.best_eval_loss = avg_loss
            self.save_checkpoint("best_model")
            print(f"  New best model saved! Loss: {avg_loss:.4f}")
            
        return avg_loss
        
    def _log_training_step(self, loss: float):
        """Log training step information."""
        lr = self._get_lr()
        log_entry = {
            'step': self.global_step,
            'epoch': self.epoch,
            'loss': loss,
            'learning_rate': lr
        }
        self.training_logs.append(log_entry)
        
        print(f"Step {self.global_step}: Loss = {loss:.4f}, LR = {lr:.2e}")
        
    def save_checkpoint(self, checkpoint_name: str):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_name: Name for the checkpoint
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{checkpoint_name}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
            'best_eval_loss': self.best_eval_loss
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Manage checkpoint limits
        self._cleanup_checkpoints()
        
    def _cleanup_checkpoints(self):
        """Remove old checkpoints if exceeding limit."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            [f for f in checkpoint_dir.glob("step_*.pt")],
            key=lambda x: x.stat().st_mtime
        )
        
        # Keep only the most recent checkpoints
        if len(checkpoints) > self.config.save_total_limit:
            for old_checkpoint in checkpoints[:-self.config.save_total_limit]:
                old_checkpoint.unlink()
                print(f"Removed old checkpoint: {old_checkpoint}")
                
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Resuming from epoch {self.epoch}, step {self.global_step}")
