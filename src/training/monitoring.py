"""
Enhanced training utilities for Nima.

Includes:
- Early stopping
- Learning rate scheduling
- Metrics tracking and visualization
- TensorBoard/W&B integration
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    enabled: bool = True
    patience: int = 5
    min_delta: float = 0.001
    metric: str = "val_loss"
    mode: str = "min"  # 'min' for loss, 'max' for accuracy


class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric stops improving.
    
    Args:
        patience: Number of evaluations to wait for improvement
        min_delta: Minimum change to qualify as improvement
        metric: Name of metric to monitor
        mode: 'min' for metrics where lower is better, 'max' for metrics where higher is better
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        metric: str = "val_loss",
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        logger.info(f"EarlyStopping initialized: patience={patience}, metric={metric}, mode={mode}")
    
    def __call__(self, current_score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current metric value
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False
        
        # Check improvement based on mode
        if self.mode == 'min':
            improved = current_score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = current_score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            logger.info(f"✓ {self.metric} improved to {current_score:.6f}")
        else:
            self.counter += 1
            logger.info(
                f"✗ {self.metric} did not improve from {self.best_score:.6f} "
                f"(patience: {self.counter}/{self.patience})"
            )
            
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered! Best {self.metric}: {self.best_score:.6f} at epoch {self.best_epoch}")
                self.early_stop = True
        
        return self.early_stop


class MetricsTracker:
    """
    Track and visualize training metrics.
    """
    
    def __init__(self, metrics: List[str], save_dir: Optional[str] = None):
        """
        Initialize metrics tracker.
        
        Args:
            metrics: List of metric names to track
            save_dir: Directory to save plots
        """
        self.metrics = {name: [] for name in metrics}
        self.steps = []
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def add(self, step: int, **metrics):
        """Add metrics for a step."""
        self.steps.append(step)
        for name, value in metrics.items():
            if name in self.metrics:
                self.metrics[name].append(value)
    
    def get_history(self, metric: str) -> List[float]:
        """Get history of a metric."""
        return self.metrics.get(metric, [])
    
    def get_latest(self, metric: str) -> Optional[float]:
        """Get latest value of a metric."""
        history = self.get_history(metric)
        return history[-1] if history else None
    
    def plot_metrics(
        self,
        metric_pairs: Optional[List[List[str]]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot metrics over time.
        
        Args:
            metric_pairs: List of metric groups to plot together
            save_path: Path to save the plot
        """
        if not metric_pairs:
            # Plot all metrics individually
            metric_pairs = [[name] for name in self.metrics.keys()]
        
        n_plots = len(metric_pairs)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        for ax, metrics_group in zip(axes, metric_pairs):
            for metric_name in metrics_group:
                if metric_name in self.metrics and self.metrics[metric_name]:
                    ax.plot(self.steps[:len(self.metrics[metric_name])], 
                           self.metrics[metric_name], 
                           label=metric_name, 
                           marker='o', 
                           markersize=3,
                           linewidth=2)
            
            ax.set_xlabel('Steps', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title(f"{' vs '.join(metrics_group)}", fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved metrics plot to: {save_path}")
        
        return fig
    
    def plot_loss_curves(self, save_path: Optional[str] = None):
        """Plot training and validation loss curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        if 'train_loss' in self.metrics and self.metrics['train_loss']:
            ax1.plot(self.steps[:len(self.metrics['train_loss'])], 
                    self.metrics['train_loss'], 
                    label='Training Loss', 
                    color='blue', 
                    linewidth=2)
        
        if 'val_loss' in self.metrics and self.metrics['val_loss']:
            # Validation might have fewer points
            val_steps = self.steps[:len(self.metrics['val_loss'])]
            ax1.plot(val_steps, 
                    self.metrics['val_loss'], 
                    label='Validation Loss', 
                    color='red', 
                    linewidth=2,
                    marker='o',
                    markersize=5)
        
        ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Perplexity plot
        if 'train_perplexity' in self.metrics and self.metrics['train_perplexity']:
            ax2.plot(self.steps[:len(self.metrics['train_perplexity'])], 
                    self.metrics['train_perplexity'], 
                    label='Training Perplexity', 
                    color='blue', 
                    linewidth=2)
        
        if 'val_perplexity' in self.metrics and self.metrics['val_perplexity']:
            val_steps = self.steps[:len(self.metrics['val_perplexity'])]
            ax2.plot(val_steps, 
                    self.metrics['val_perplexity'], 
                    label='Validation Perplexity', 
                    color='red', 
                    linewidth=2,
                    marker='o',
                    markersize=5)
        
        ax2.set_xlabel('Steps', fontsize=12)
        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.set_title('Training and Validation Perplexity', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved loss curves to: {save_path}")
        
        return fig
    
    def print_summary(self):
        """Print summary of metrics."""
        print("\n" + "=" * 80)
        print("METRICS SUMMARY")
        print("=" * 80)
        
        for metric_name, values in self.metrics.items():
            if values:
                print(f"{metric_name:30s}: {values[-1]:.6f} (latest)")
                print(f"{'':30s}  Best: {min(values) if 'loss' in metric_name else max(values):.6f}")
        
        print("=" * 80 + "\n")


class TensorBoardLogger:
    """Simple TensorBoard logger wrapper."""
    
    def __init__(self, log_dir: str, enabled: bool = True):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.writer = None
        
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
                logger.info(f"TensorBoard logging enabled: {log_dir}")
            except ImportError:
                logger.warning("tensorboard not installed. Install with: pip install tensorboard")
                self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram of values."""
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        if self.enabled and self.writer:
            self.writer.add_text(tag, text, step)
    
    def close(self):
        """Close the writer."""
        if self.enabled and self.writer:
            self.writer.close()


class WandBLogger:
    """Weights & Biases logger wrapper."""
    
    def __init__(
        self,
        project: str,
        name: str,
        config: Optional[Dict] = None,
        enabled: bool = False,
        **kwargs
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            name: Run name
            config: Configuration dictionary
            enabled: Whether logging is enabled
            **kwargs: Additional arguments for wandb.init
        """
        self.enabled = enabled
        self.run = None
        
        if enabled:
            try:
                import wandb
                self.run = wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    **kwargs
                )
                logger.info(f"W&B logging enabled: {project}/{name}")
            except ImportError:
                logger.warning("wandb not installed. Install with: pip install wandb")
                self.enabled = False
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if self.enabled and self.run:
            import wandb
            wandb.log(metrics, step=step)
    
    def finish(self):
        """Finish the run."""
        if self.enabled and self.run:
            import wandb
            wandb.finish()


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    loss: float
) -> Dict[str, float]:
    """
    Compute various metrics for evaluation.
    
    Args:
        predictions: Model predictions (logits)
        labels: Ground truth labels
        loss: Computed loss value
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Loss and perplexity
    metrics['loss'] = loss
    metrics['perplexity'] = np.exp(loss)
    
    # Accuracy
    pred_tokens = predictions.argmax(dim=-1)
    mask = labels != -100  # Ignore padding
    correct = (pred_tokens == labels) & mask
    metrics['accuracy'] = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
    
    return metrics
