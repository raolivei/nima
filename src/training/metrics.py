"""
Training metrics and monitoring utilities.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np


class MetricsTracker:
    """
    Track and log training metrics.
    """
    
    def __init__(self, log_dir: str = "experiments/logs"):
        """
        Initialize metrics tracker.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.step_times = []
        self.start_time = time.time()
        
    def log(self, metrics: Dict[str, float], step: int):
        """
        Log metrics for a step.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Current training step
        """
        for key, value in metrics.items():
            self.metrics[key].append({
                'step': step,
                'value': value,
                'timestamp': time.time() - self.start_time
            })
            
    def log_step_time(self, duration: float):
        """Log step execution time."""
        self.step_times.append(duration)
        
    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> float:
        """
        Get average of a metric.
        
        Args:
            metric_name: Name of metric
            last_n: Number of last values to average (None for all)
            
        Returns:
            Average value
        """
        if metric_name not in self.metrics:
            return 0.0
            
        values = [m['value'] for m in self.metrics[metric_name]]
        if last_n is not None:
            values = values[-last_n:]
            
        return np.mean(values) if values else 0.0
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        summary = {}
        
        for metric_name, values in self.metrics.items():
            vals = [m['value'] for m in values]
            summary[metric_name] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'min': np.min(vals),
                'max': np.max(vals),
                'last': vals[-1] if vals else 0.0,
                'count': len(vals)
            }
            
        # Add timing information
        if self.step_times:
            summary['step_time'] = {
                'mean': np.mean(self.step_times),
                'std': np.std(self.step_times),
                'total': sum(self.step_times)
            }
            
        summary['total_time'] = time.time() - self.start_time
        
        return summary
        
    def save(self, filename: str = "metrics.json"):
        """
        Save metrics to file.
        
        Args:
            filename: Output filename
        """
        save_path = self.log_dir / filename
        
        # Convert metrics to serializable format
        metrics_dict = {
            'metrics': dict(self.metrics),
            'summary': self.get_summary(),
            'step_times': self.step_times
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
            
        print(f"Metrics saved to {save_path}")
        
    def load(self, filename: str = "metrics.json"):
        """
        Load metrics from file.
        
        Args:
            filename: Input filename
        """
        load_path = self.log_dir / filename
        
        with open(load_path, 'r') as f:
            metrics_dict = json.load(f)
            
        self.metrics = defaultdict(list, metrics_dict['metrics'])
        self.step_times = metrics_dict.get('step_times', [])
        
        print(f"Metrics loaded from {load_path}")


class ProgressPrinter:
    """
    Print training progress in a formatted way.
    """
    
    def __init__(
        self,
        total_steps: int,
        metrics_to_track: List[str] = None,
        print_frequency: int = 10
    ):
        """
        Initialize progress printer.
        
        Args:
            total_steps: Total number of training steps
            metrics_to_track: List of metric names to display
            print_frequency: How often to print (in steps)
        """
        self.total_steps = total_steps
        self.metrics_to_track = metrics_to_track or ['loss', 'lr']
        self.print_frequency = print_frequency
        self.start_time = time.time()
        
    def print_step(self, step: int, metrics: Dict[str, float]):
        """
        Print progress for a step.
        
        Args:
            step: Current step
            metrics: Dictionary of metrics
        """
        if step % self.print_frequency != 0:
            return
            
        elapsed_time = time.time() - self.start_time
        steps_per_sec = step / elapsed_time if elapsed_time > 0 else 0
        eta = (self.total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
        
        # Build progress string
        progress_pct = (step / self.total_steps) * 100
        progress_bar = self._create_progress_bar(progress_pct)
        
        print(f"\rStep {step}/{self.total_steps} {progress_bar} ", end='')
        
        # Print metrics
        metric_strs = []
        for metric_name in self.metrics_to_track:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, float):
                    metric_strs.append(f"{metric_name}={value:.4f}")
                else:
                    metric_strs.append(f"{metric_name}={value}")
                    
        print(f"{' '.join(metric_strs)} | {steps_per_sec:.2f} steps/s | ETA: {self._format_time(eta)}", end='')
        
    def _create_progress_bar(self, progress_pct: float, width: int = 30) -> str:
        """Create a progress bar string."""
        filled = int(width * progress_pct / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {progress_pct:.1f}%"
        
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
            
    def print_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Print summary at epoch end."""
        print(f"\n\nEpoch {epoch} completed:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric direction
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        
    def __call__(self, metric_value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric_value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = metric_value < (self.best_value - self.min_delta)
        else:
            improved = metric_value > (self.best_value + self.min_delta)
            
        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\nEarly stopping triggered after {self.counter} epochs without improvement")
                
        return self.early_stop
        
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.early_stop = False


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity value
    """
    return np.exp(loss)


def calculate_tokens_per_second(
    num_tokens: int,
    elapsed_time: float
) -> float:
    """
    Calculate tokens processed per second.
    
    Args:
        num_tokens: Number of tokens processed
        elapsed_time: Time elapsed in seconds
        
    Returns:
        Tokens per second
    """
    return num_tokens / elapsed_time if elapsed_time > 0 else 0.0


def estimate_training_time(
    current_step: int,
    total_steps: int,
    elapsed_time: float
) -> Dict[str, float]:
    """
    Estimate remaining training time.
    
    Args:
        current_step: Current training step
        total_steps: Total training steps
        elapsed_time: Time elapsed so far
        
    Returns:
        Dictionary with time estimates
    """
    if current_step == 0:
        return {
            'remaining_seconds': 0.0,
            'total_seconds': 0.0,
            'steps_per_second': 0.0
        }
        
    steps_per_second = current_step / elapsed_time
    remaining_steps = total_steps - current_step
    remaining_seconds = remaining_steps / steps_per_second
    total_seconds = elapsed_time + remaining_seconds
    
    return {
        'remaining_seconds': remaining_seconds,
        'total_seconds': total_seconds,
        'steps_per_second': steps_per_second
    }
